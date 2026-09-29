#!/usr/bin/env python3
"""Pullback: optimize an activation path to track a probe trajectory; compare to Mh.

Follows Wurgaft et al. §3.3 / App. A.8 and the mountain-car variant in App. C.3,
adapted to V-JEPA: behavior is the frozen linear probe (sincos for direction,
scalar for speed), not a token simplex.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import LBFGS

from common import ARTIFACTS, ensure_dirs, load_split, theta_to_sincos, train_probe_c11
from metrics import circular_mae_degrees, mae, sincos_to_degrees
from part2_splines import (
    L_STAR,
    N_PCA,
    N_WAYPOINTS,
    ActivationSpline,
    build_paths,
    circular_lerp,
    decode_probe,
    fit_spline,
    label_to_u,
    load_acts,
    path_us,
    u_to_label,
)

SEED = 0
N_OPT_DIM = 32  # paper A.8 restricts pullback to first 32 PCs
SMOOTH_WEIGHT = 1e-3
MAX_LBFGS_STEPS = 60


def claimed_along(spline: ActivationSpline, path, kind: str) -> np.ndarray:
    us = path_us(spline, path, kind)
    return np.array([u_to_label(spline, u, kind) for u in us], dtype=np.float64)


def target_outputs(claimed: np.ndarray, kind: str) -> np.ndarray:
    if kind == "direction":
        return theta_to_sincos(claimed)
    return claimed.astype(np.float32)[:, None]


def pca_basis(spline: ActivationSpline) -> tuple[np.ndarray, np.ndarray]:
    """Return (components (n_comp, d), mean (d,))."""
    return spline.pca.components_.astype(np.float64), spline.pca.mean_.astype(np.float64)


def decode_from_z(
    z: torch.Tensor,
    leftover: torch.Tensor,
    components: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """z: (K, n_pca) or (n_pca,), leftover: (N, d). Returns (K, N, d) or (N, d)."""
    # recon = z @ components + mean
    if z.ndim == 1:
        recon = z @ components + mean  # (d,)
        return leftover + recon  # broadcast (N, d)
    recon = z @ components + mean  # (K, d)
    return leftover[None, :, :] + recon[:, None, :]  # (K, N, d)


def distance_to_curve(Z: np.ndarray, spline: ActivationSpline, n_samples: int = 1024) -> np.ndarray:
    if spline.periodic:
        grid = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    else:
        grid = np.linspace(float(spline.u.min()), float(spline.u.max()), n_samples)
    curve = spline.eval_pca(grid)
    d2 = ((Z[:, None, :] - curve[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(d2.min(axis=1))


def build_reference_paths(spline: ActivationSpline, path, kind: str) -> dict[str, np.ndarray]:
    """Return PCA waypoint arrays for geometric (Mh) and chord."""
    us = path_us(spline, path, kind)
    ts = np.linspace(0.0, 1.0, N_WAYPOINTS)
    z0 = spline.eval_pca(path.u0)
    z1 = spline.eval_pca(path.u1)
    geo = np.stack([spline.eval_pca(u) for u in us], axis=0)
    chord = np.stack([(1.0 - t) * z0 + t * z1 for t in ts], axis=0)
    return {"geometric": geo, "chord": chord}


@torch.no_grad()
def path_probe_metrics(
    Z: np.ndarray,
    *,
    spline: ActivationSpline,
    leftover_np: np.ndarray,
    probe,
    claimed: np.ndarray,
    kind: str,
    device: torch.device,
) -> dict:
    components, mean = pca_basis(spline)
    leftover = leftover_np
    errs_c, errs_t, decoded = [], [], []
    # y_test needed for truth — pass via leftover's matching; we compute claimed only here
    for k, z in enumerate(Z):
        X = leftover + (z @ components + mean).astype(np.float32)
        pred = decode_probe(probe, X.astype(np.float32), kind, device)
        if kind == "direction":
            decoded.append(float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(pred))), np.mean(np.cos(np.radians(pred))))) % 360))
            errs_c.append(circular_mae_degrees(pred, np.full(len(pred), claimed[k])))
        else:
            decoded.append(float(np.mean(pred)))
            errs_c.append(mae(np.full(len(pred), claimed[k]), pred))
    return {
        "decoded_mean": decoded,
        "err_to_claimed": errs_c,
        "final_err_to_claimed": float(errs_c[-1]),
        "mid_err_to_claimed": float(errs_c[len(errs_c) // 2]),
        "mean_err_to_claimed": float(np.mean(errs_c)),
    }


def optimize_pullback(
    *,
    spline: ActivationSpline,
    path,
    kind: str,
    leftover_np: np.ndarray,
    probe,
    claimed: np.ndarray,
    device: torch.device,
    n_opt_dim: int = N_OPT_DIM,
) -> dict:
    refs = build_reference_paths(spline, path, kind)
    chord = refs["chord"]
    geo = refs["geometric"]
    n_pca = spline.pca.n_components_
    n_opt = min(n_opt_dim, n_pca)
    fixed_tail = chord[:, n_opt:]  # held at chord (paper holds remaining PCs)

    components_np, mean_np = pca_basis(spline)
    components = torch.tensor(components_np, dtype=torch.float64, device=device)
    mean = torch.tensor(mean_np, dtype=torch.float64, device=device)
    leftover = torch.tensor(leftover_np, dtype=torch.float64, device=device)
    target = torch.tensor(target_outputs(claimed, kind), dtype=torch.float64, device=device)  # (K, out)

    W = probe.weight.detach().to(device=device, dtype=torch.float64)  # (out, d)
    b = probe.bias.detach().to(device=device, dtype=torch.float64)

    # Optimize only first n_opt dims of each waypoint; init at chord
    z_opt = torch.tensor(chord[:, :n_opt], dtype=torch.float64, device=device, requires_grad=True)
    z_tail = torch.tensor(fixed_tail, dtype=torch.float64, device=device)

    opt = LBFGS([z_opt], lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

    loss_hist = []

    def closure():
        opt.zero_grad(set_to_none=True)
        z_full = torch.cat([z_opt, z_tail], dim=1)  # (K, n_pca)
        X = decode_from_z(z_full, leftover, components, mean)  # (K, N, d)
        # probe: (K, N, out)
        pred = torch.einsum("od,knd->kno", W, X) + b
        # match target broadcast over clips
        tgt = target[:, None, :]  # (K, 1, out)
        loss = torch.mean((pred - tgt) ** 2)
        # mild path smoothness in the optimized coords
        if z_opt.shape[0] > 1:
            loss = loss + SMOOTH_WEIGHT * torch.mean((z_opt[1:] - z_opt[:-1]) ** 2)
        loss.backward()
        return loss

    prev = None
    for step in range(MAX_LBFGS_STEPS):
        loss = opt.step(closure)
        val = float(loss.detach().cpu())
        loss_hist.append(val)
        if prev is not None and abs(prev - val) / max(abs(prev), 1e-8) < 1e-3:
            break
        prev = val
        if step % 5 == 0 or step == MAX_LBFGS_STEPS - 1:
            print(f"  LBFGS step {step}: loss={val:.6f}")

    with torch.no_grad():
        z_full = torch.cat([z_opt, z_tail], dim=1).cpu().numpy()

    d_pull = distance_to_curve(z_full, spline)
    d_chord = distance_to_curve(chord, spline)
    d_geo = distance_to_curve(geo, spline)

    # Recovery fraction vs chord→geometric gap (paper mountain-car style)
    gap = float(d_chord.mean() - d_geo.mean())
    recovery = None
    if gap > 1e-8:
        recovery = float((d_chord.mean() - d_pull.mean()) / gap)

    metrics = path_probe_metrics(
        z_full,
        spline=spline,
        leftover_np=leftover_np,
        probe=probe,
        claimed=claimed,
        kind=kind,
        device=device,
    )
    metrics_chord = path_probe_metrics(
        chord,
        spline=spline,
        leftover_np=leftover_np,
        probe=probe,
        claimed=claimed,
        kind=kind,
        device=device,
    )
    metrics_geo = path_probe_metrics(
        geo,
        spline=spline,
        leftover_np=leftover_np,
        probe=probe,
        claimed=claimed,
        kind=kind,
        device=device,
    )

    return {
        "n_opt_dim": n_opt,
        "n_steps": len(loss_hist),
        "loss_hist": loss_hist,
        "final_loss": loss_hist[-1] if loss_hist else None,
        "distance_to_Mh": {
            "pullback_mean": float(d_pull.mean()),
            "pullback_median": float(np.median(d_pull)),
            "chord_mean": float(d_chord.mean()),
            "geometric_mean": float(d_geo.mean()),
            "recovery_fraction": recovery,
            "per_waypoint_pullback": d_pull.tolist(),
            "per_waypoint_chord": d_chord.tolist(),
        },
        "probe": {
            "pullback": metrics,
            "chord": metrics_chord,
            "geometric": metrics_geo,
        },
        "waypoints_pca": {
            "pullback": z_full.tolist(),
            "chord": chord.tolist(),
            "geometric": geo.tolist(),
        },
    }


def plot_pullback(spline: ActivationSpline, result: dict, path_name: str, claimed: np.ndarray, kind: str, out: Path):
    Zp = np.asarray(result["waypoints_pca"]["pullback"])
    Zc = np.asarray(result["waypoints_pca"]["chord"])
    Zg = np.asarray(result["waypoints_pca"]["geometric"])

    fig = plt.figure(figsize=(11, 4.6))
    ax3 = fig.add_subplot(121, projection="3d")
    if spline.periodic:
        grid = np.linspace(0, 1, 200, endpoint=False)
    else:
        grid = np.linspace(float(spline.u.min()), float(spline.u.max()), 200)
    curve = spline.eval_pca(grid)
    ax3.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#0f766e", lw=1.8, alpha=0.7, label="Mh")
    ax3.plot(Zg[:, 0], Zg[:, 1], Zg[:, 2], color="#0f766e", lw=2.4, marker="o", ms=3, label="geometric")
    ax3.plot(Zc[:, 0], Zc[:, 1], Zc[:, 2], color="#9f1239", lw=2.0, ls="--", marker="x", ms=3, label="chord")
    ax3.plot(Zp[:, 0], Zp[:, 1], Zp[:, 2], color="#1d4ed8", lw=2.2, marker="D", ms=3, label="pullback")
    ax3.set_title(f"{kind} pullback PCA — {path_name}")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(122)
    ts = np.linspace(0, 1, len(claimed))
    for name, color, ls in (
        ("geometric", "#0f766e", "-"),
        ("chord", "#9f1239", "--"),
        ("pullback", "#1d4ed8", "-"),
    ):
        dec = result["probe"][name]["decoded_mean"]
        ax.plot(ts, dec, color=color, ls=ls, marker="o", ms=3, label=name)
    ax.plot(ts, claimed, color="#142028", ls=":", label="claimed")
    ax.set_xlabel("t")
    ax.set_ylabel("decoded mean")
    ax.set_title("probe readout along path")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_distance_bars(results: dict, out: Path):
    names = list(results.keys())
    pull = [results[n]["distance_to_Mh"]["pullback_mean"] for n in names]
    chord = [results[n]["distance_to_Mh"]["chord_mean"] for n in names]
    geo = [results[n]["distance_to_Mh"]["geometric_mean"] for n in names]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - w, geo, w, label="geometric (on Mh)", color="#0f766e")
    ax.bar(x, pull, w, label="pullback", color="#1d4ed8")
    ax.bar(x + w, chord, w, label="chord init", color="#9f1239")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("mean distance to Mh (PCA)")
    ax.set_title("Does pullback recover the activation spline?")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_kind(kind: str, device: torch.device) -> dict:
    print(f"\n=== pullback {kind} ===")
    d = load_acts(kind)
    split = load_split(kind)
    train_ids = set(split["train"])
    test_ids = set(split["test"])
    L = L_STAR
    X_all = d["layers"][:, L]
    y_all = d["targets"]
    motion = d["motion"]

    if kind == "direction":
        fit_mask = np.array(
            [int(cid) in train_ids and m == "velocity" for cid, m in zip(d["ids"], motion)],
            dtype=bool,
        )
        X_fit, y_fit = X_all[fit_mask], y_all[fit_mask]
    else:
        train_idx = np.array([i for i, cid in enumerate(d["ids"]) if int(cid) in train_ids])
        X_fit, y_fit = X_all[train_idx], y_all[train_idx]

    test_idx = np.array([i for i, cid in enumerate(d["ids"]) if int(cid) in test_ids])
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    spline = fit_spline(X_fit, y_fit, kind=kind)
    leftover = spline.leftover(X_test)

    if kind == "direction":
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        probe = train_probe_c11(X_test, theta_to_sincos(y_test), kind=kind, device=device)
    else:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        probe = train_probe_c11(X_test, y_test, kind=kind, device=device)

    paths = build_paths(spline, kind, y_fit if kind == "direction" else y_all)
    # Focus: direction half-turn + short arc; speed one path for contrast
    if kind == "direction":
        paths = paths  # both
    else:
        paths = paths[:1]

    out = {"kind": kind, "L_star": L, "n_pca": int(spline.pca.n_components_), "paths": {}}
    for path in paths:
        print(f"path {path.name}: {path.label0} → {path.label1}")
        claimed = claimed_along(spline, path, kind)
        result = optimize_pullback(
            spline=spline,
            path=path,
            kind=kind,
            leftover_np=leftover,
            probe=probe,
            claimed=claimed,
            device=device,
        )
        dinfo = result["distance_to_Mh"]
        print(
            f"  dist→Mh: geometric={dinfo['geometric_mean']:.3f}, "
            f"pullback={dinfo['pullback_mean']:.3f}, chord={dinfo['chord_mean']:.3f}, "
            f"recovery={dinfo['recovery_fraction']}"
        )
        print(
            f"  mid MAE→claimed: geo={result['probe']['geometric']['mid_err_to_claimed']:.3f}, "
            f"pull={result['probe']['pullback']['mid_err_to_claimed']:.3f}, "
            f"chord={result['probe']['chord']['mid_err_to_claimed']:.3f}"
        )
        plot_pullback(
            spline,
            result,
            path.name,
            claimed,
            kind,
            ARTIFACTS / "figures" / f"part2_pullback_{path.name}.png",
        )
        # drop bulky waypoint lists from summary json? keep — useful. maybe strip for size
        slim = {k: v for k, v in result.items() if k != "waypoints_pca"}
        slim["path"] = {
            "name": path.name,
            "label0": path.label0,
            "label1": path.label1,
            "note": path.note,
        }
        # keep first 3 PCs of waypoints for plotting reproducibility
        slim["waypoints_pca_pc3"] = {
            key: np.asarray(result["waypoints_pca"][key])[:, :3].tolist()
            for key in ("pullback", "chord", "geometric")
        }
        out["paths"][path.name] = slim
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kinds", nargs="+", default=["direction", "speed"])
    args = ap.parse_args()
    device = torch.device(args.device)
    ensure_dirs()
    (ARTIFACTS / "figures").mkdir(parents=True, exist_ok=True)

    summary = {
        "method": {
            "n_waypoints": N_WAYPOINTS,
            "n_opt_dim": N_OPT_DIM,
            "smooth_weight": SMOOTH_WEIGHT,
            "max_lbfgs_steps": MAX_LBFGS_STEPS,
            "behavior": "frozen linear probe on unsteered test",
            "init": "chord in PCA subspace",
        },
        "variables": {},
    }
    for kind in args.kinds:
        summary["variables"][kind] = run_kind(kind, device)

    # distance bar across all paths
    flat = {}
    for kind, block in summary["variables"].items():
        for name, res in block["paths"].items():
            flat[name] = res
    if flat:
        plot_distance_bars(flat, ARTIFACTS / "figures" / "part2_pullback_distances.png")

    out = ARTIFACTS / "results" / "part2_pullback.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
