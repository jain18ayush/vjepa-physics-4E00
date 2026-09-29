#!/usr/bin/env python3
"""Tube-not-curve: spline spine + residual normal bundle vs multi-probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import ARTIFACTS, ensure_dirs, load_split, theta_to_sincos, train_probe_c11
from metrics import circular_mae_degrees, sincos_to_degrees
from part2_splines import (
    L_STAR,
    N_WAYPOINTS,
    ActivationSpline,
    build_paths,
    circular_mean_degrees,
    decode_probe,
    fit_spline,
    load_acts,
    path_us,
    u_to_label,
)
from steer import ProbeWeights, steer as multiprobe_steer

SEED = 0
K_TUBE = [0, 1, 2, 5, 10, 20]
N_CURVE_SAMPLES = 1024


def pca_mats(spline: ActivationSpline):
    return spline.pca.components_.astype(np.float64), spline.pca.mean_.astype(np.float64)


def nearest_on_spline(Z: np.ndarray, spline: ActivationSpline, n_samples: int = N_CURVE_SAMPLES):
    if spline.periodic:
        grid = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    else:
        grid = np.linspace(float(spline.u.min()), float(spline.u.max()), n_samples)
    curve = spline.eval_pca(grid)
    # tangents via finite diff on the same grid
    tang = np.gradient(curve, axis=0)
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
    d2 = ((Z[:, None, :] - curve[None, :, :]) ** 2).sum(axis=-1)
    idx = d2.argmin(axis=1)
    return grid[idx], curve[idx], tang[idx], np.sqrt(d2.min(axis=1))


def fit_tube_basis(spline: ActivationSpline, X_fit: np.ndarray) -> dict:
    Z = spline.project(X_fit)
    u_star, z_star, tang, resid_norm = nearest_on_spline(Z, spline)
    r = Z - z_star
    # remove tangential component
    r_perp = r - (np.sum(r * tang, axis=1, keepdims=True) * tang)
    # PCA on perpendicular residuals
    # center
    mu = r_perp.mean(axis=0)
    Rc = r_perp - mu
    # SVD
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    var = (S**2) / max(len(Rc) - 1, 1)
    total = float(var.sum()) + 1e-12
    tang_energy = float(np.mean(np.sum((r - r_perp) ** 2, axis=1)))
    perp_energy = float(np.mean(np.sum(r_perp**2, axis=1)))
    return {
        "basis": Vt,  # (n_comp, n_pca), rows are components
        "mean_residual": mu,
        "variance": var,
        "variance_explained_ratio": (var / total).tolist(),
        "cumulative_var_ratio": np.cumsum(var / total).tolist(),
        "tangential_energy_mean": tang_energy,
        "perpendicular_energy_mean": perp_energy,
        "residual_norm_median": float(np.median(resid_norm)),
        "residual_norm_mean": float(np.mean(resid_norm)),
    }


def solve_tube_offset(
    *,
    z_spine: np.ndarray,
    V_k: np.ndarray,
    leftover: np.ndarray,
    probe,
    theta_star_deg: float,
    components: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Solve c so mean probe over clips hits (sin,cos) at z_spine + V_k c.

    Uses the linear probe: pred = W (leftover + mean + (z_spine + V c) @ components) + b
    Average over clips → only the shared z terms matter for the mean target.
    Actually each clip has different leftover; we match each clip then average c,
    or match the mean prediction. Match each clip's probe to target and average c
    via stacked lstsq (same as multi-probe with shared V).
    """
    # W: (2, d), we want W @ x_i + b = t for all i
    # x_i = leftover_i + mean + (z_spine + V c) @ components
    # W @ leftover_i + W @ mean + W @ components.T @ z_spine + (W @ components.T @ V) c + b = t
    W = probe.weight.detach().cpu().numpy().astype(np.float64)  # (2, d)
    b = probe.bias.detach().cpu().numpy().astype(np.float64)
    t = np.array(
        [math.sin(math.radians(theta_star_deg)), math.cos(math.radians(theta_star_deg))],
        dtype=np.float64,
    )
    WC = W @ components.T  # (2, n_pca)
    # V_k as (n_pca, k)
    V = np.asarray(V_k, dtype=np.float64)
    if V.size == 0:
        return np.zeros(0, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, None]
    if V.shape[0] != components.shape[0] and V.shape[1] == components.shape[0]:
        V = V.T
    A = WC @ V  # (2, k)
    N = leftover.shape[0]
    rhs = np.concatenate(
        [t - b - W @ leftover[i] - WC @ z_spine for i in range(N)],
        axis=0,
    )
    A_big = np.tile(A, (N, 1))
    c, *_ = np.linalg.lstsq(A_big, rhs, rcond=None)
    return c.astype(np.float64)


def decode_z_batch(z: np.ndarray, leftover: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    recon = z @ components + mean
    return (leftover + recon).astype(np.float32)


def evaluate_waypoints(
    Z_waypoints: np.ndarray,
    *,
    leftover: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    probe,
    claimed: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict:
    rows = []
    for t_i, (z, claim) in enumerate(zip(Z_waypoints, claimed)):
        X = decode_z_batch(z, leftover, components, mean)
        pred = decode_probe(probe, X, "direction", device)
        rows.append(
            {
                "t": float(t_i / (len(Z_waypoints) - 1)),
                "claimed_label": float(claim),
                "decoded_mean": circular_mean_degrees(pred),
                "err_to_claimed": circular_mae_degrees(pred, np.full(len(pred), claim)),
                "err_to_truth": circular_mae_degrees(pred, y_test),
            }
        )
    return {
        "waypoints": rows,
        "mid_err_to_claimed": float(rows[len(rows) // 2]["err_to_claimed"]),
        "final_err_to_target": float(rows[-1]["err_to_claimed"]),
        "final_err_to_truth": float(rows[-1]["err_to_truth"]),
        "mean_err_to_claimed": float(np.mean([r["err_to_claimed"] for r in rows])),
    }


def load_nullspace_probes(device: torch.device) -> list[ProbeWeights]:
    pt = torch.load(
        ARTIFACTS / "probes" / f"direction_velocity_L{L_STAR}_nullspace.pt",
        map_location="cpu",
        weights_only=False,
    )
    return [ProbeWeights(p["weight"], p["bias"]).to(device) for p in pt["probes"]]


def nullspace_tube_basis(probes: list[ProbeWeights], components: np.ndarray, tangent: np.ndarray, k: int) -> np.ndarray:
    """Map probe weight rows into PCA space, QR, remove tangent, take first k."""
    # weight: (2, d) per probe → stack → (2n, d)
    rows = []
    for p in probes:
        w = p.weight.detach().cpu().numpy().astype(np.float64)
        rows.append(w)
    W = np.concatenate(rows, axis=0)  # (2n, d)
    # components: (n_pca, d). Direction δ in full space → components @ δ in PCA.
    Wp = (components @ W.T).T  # (2n, n_pca)
    # remove tangent
    t = tangent / (np.linalg.norm(tangent) + 1e-12)
    Wp = Wp - (Wp @ t)[:, None] * t[None, :]
    # QR on transpose to get orthonormal frame in n_pca dims spanning rows
    Q, R = np.linalg.qr(Wp.T, mode="reduced")  # Q: (n_pca, rank)
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.abs([R])
    keep = diag >= 1e-6
    Q = Q[:, keep]
    k = min(k, Q.shape[1])
    return Q[:, :k]  # (n_pca, k)


def run_path(
    *,
    spline: ActivationSpline,
    path,
    tube: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe,
    probes: list[ProbeWeights],
    device: torch.device,
) -> dict:
    components, mean = pca_mats(spline)
    leftover = spline.leftover(X_test)
    us = path_us(spline, path, "direction")
    claimed = np.array([u_to_label(spline, u, "direction") for u in us])
    ts = np.linspace(0.0, 1.0, N_WAYPOINTS)
    z0 = spline.eval_pca(path.u0)
    z1 = spline.eval_pca(path.u1)
    # tangent at target for nullspace tube
    _, _, tang_grid, _ = nearest_on_spline(z1[None, :], spline)
    tangent = tang_grid[0]

    geo = np.stack([spline.eval_pca(u) for u in us], axis=0)
    chord = np.stack([(1 - t) * z0 + t * z1 for t in ts], axis=0)

    out = {
        "path": {"name": path.name, "label0": path.label0, "label1": path.label1},
        "spline_only": evaluate_waypoints(
            geo, leftover=leftover, components=components, mean=mean, probe=probe, claimed=claimed, y_test=y_test, device=device
        ),
        "chord": evaluate_waypoints(
            chord, leftover=leftover, components=components, mean=mean, probe=probe, claimed=claimed, y_test=y_test, device=device
        ),
        "tube_residual_pca": {},
        "tube_nullspace": {},
        "multiprobe": {},
    }

    theta_star = float(path.label1)
    basis_rows = tube["basis"]  # (n_comp, n_pca)

    for k in K_TUBE:
        # residual-PCA tube
        if k == 0:
            V = np.zeros((components.shape[0], 0))
            c = np.zeros(0)
        else:
            V = basis_rows[:k].T  # (n_pca, k)
            c = solve_tube_offset(
                z_spine=z1,
                V_k=V,
                leftover=leftover,
                probe=probe,
                theta_star_deg=theta_star,
                components=components,
                mean=mean,
            )
        # along-path: s(u(t)) + t * V c
        Z_along = np.stack([geo[i] + ts[i] * (V @ c if k else 0) for i in range(len(ts))], axis=0)
        # endpoint-only: mid path pure spline, last point tube-corrected
        Z_end = geo.copy()
        Z_end[-1] = z1 + (V @ c if k else 0)

        out["tube_residual_pca"][str(k)] = {
            "along_path": evaluate_waypoints(
                Z_along, leftover=leftover, components=components, mean=mean, probe=probe, claimed=claimed, y_test=y_test, device=device
            ),
            "endpoint_only": evaluate_waypoints(
                Z_end, leftover=leftover, components=components, mean=mean, probe=probe, claimed=claimed, y_test=y_test, device=device
            ),
            "c_norm": float(np.linalg.norm(c)) if k else 0.0,
        }

        # nullspace tube
        Vn = nullspace_tube_basis(probes, components, tangent, k) if k else np.zeros((components.shape[0], 0))
        cn = (
            solve_tube_offset(
                z_spine=z1,
                V_k=Vn,
                leftover=leftover,
                probe=probe,
                theta_star_deg=theta_star,
                components=components,
                mean=mean,
            )
            if k
            else np.zeros(0)
        )
        Z_along_n = np.stack([geo[i] + ts[i] * (Vn @ cn if k else 0) for i in range(len(ts))], axis=0)
        Z_end_n = geo.copy()
        Z_end_n[-1] = z1 + (Vn @ cn if k else 0)
        out["tube_nullspace"][str(k)] = {
            "along_path": evaluate_waypoints(
                Z_along_n, leftover=leftover, components=components, mean=mean, probe=probe, claimed=claimed, y_test=y_test, device=device
            ),
            "endpoint_only": evaluate_waypoints(
                Z_end_n, leftover=leftover, components=components, mean=mean, probe=probe, claimed=claimed, y_test=y_test, device=device
            ),
            "c_norm": float(np.linalg.norm(cn)) if k else 0.0,
            "rank": int(Vn.shape[1]),
        }
        print(
            f"  k={k}: resid-PCA final={out['tube_residual_pca'][str(k)]['along_path']['final_err_to_target']:.2f} "
            f"mid={out['tube_residual_pca'][str(k)]['along_path']['mid_err_to_claimed']:.2f} | "
            f"null final={out['tube_nullspace'][str(k)]['along_path']['final_err_to_target']:.2f}"
        )

    # multi-probe toward endpoint
    for n in [1, 5, 20]:
        if n > len(probes):
            continue
        probes_n = probes[:n]
        steered = np.zeros_like(X_test)
        thr = math.radians(theta_star)
        for i in range(len(X_test)):
            x = torch.from_numpy(X_test[i]).to(device)
            steered[i] = multiprobe_steer(x, probes_n, thr).detach().cpu().numpy()
        pred = decode_probe(probe, steered, "direction", device)
        out["multiprobe"][str(n)] = {
            "final_err_to_target": circular_mae_degrees(pred, np.full(len(pred), theta_star)),
            "final_err_to_truth": circular_mae_degrees(pred, y_test),
            "decoded_mean": circular_mean_degrees(pred),
        }
    return out


def plot_tube_curves(path_result: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ks = [int(k) for k in path_result["tube_residual_pca"].keys()]
    # final error vs k
    ax = axes[0]
    ax.plot(ks, [path_result["tube_residual_pca"][str(k)]["along_path"]["final_err_to_target"] for k in ks], "o-", color="#0f766e", label="spine+tube (resid PCA)")
    ax.plot(ks, [path_result["tube_nullspace"][str(k)]["along_path"]["final_err_to_target"] for k in ks], "s-", color="#1d4ed8", label="spine+tube (nullspace)")
    ax.plot(ks, [path_result["tube_residual_pca"][str(k)]["along_path"]["mid_err_to_claimed"] for k in ks], "o--", color="#0f766e", alpha=0.5, label="mid-path (resid)")
    for n, style in [("1", ":"), ("5", "--"), ("20", "-")]:
        if n in path_result["multiprobe"]:
            ax.axhline(
                path_result["multiprobe"][n]["final_err_to_target"],
                color="#9f1239",
                ls=style,
                alpha=0.8,
                label=f"multi-probe n={n}",
            )
    ax.axhline(path_result["spline_only"]["final_err_to_target"], color="#142028", ls=":", label="spline only final")
    ax.set_xlabel("tube dimensionality k")
    ax.set_ylabel("circular MAE (°)")
    ax.set_title(f"{path_result['path']['name']}: endpoint vs tube rank")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # decode vs t for selected k
    ax = axes[1]
    claimed = [w["claimed_label"] for w in path_result["spline_only"]["waypoints"]]
    ts = [w["t"] for w in path_result["spline_only"]["waypoints"]]
    ax.plot(ts, claimed, "k:", label="claimed")
    ax.plot(ts, [w["decoded_mean"] for w in path_result["spline_only"]["waypoints"]], color="#142028", label="spline only")
    for k, color in [(0, "#94a3b8"), (5, "#0f766e"), (20, "#1d4ed8")]:
        rows = path_result["tube_residual_pca"][str(k)]["along_path"]["waypoints"]
        ax.plot(ts, [w["decoded_mean"] for w in rows], color=color, marker="o", ms=3, label=f"spine+tube k={k}")
    ax.plot(ts, [w["decoded_mean"] for w in path_result["chord"]["waypoints"]], color="#9f1239", ls="--", label="chord")
    ax.set_xlabel("t")
    ax.set_ylabel("decoded mean (°)")
    ax.set_title("readout along path")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_tube_variance(tube: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    cum = tube["cumulative_var_ratio"][:40]
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#0f766e")
    ax.axhline(0.9, color="#9f1239", ls="--", label="90%")
    ax.set_xlabel("tube PCA component")
    ax.set_ylabel("cumulative variance of r_⊥")
    ax.set_title(
        f"Tube residual PCA (perp energy={tube['perpendicular_energy_mean']:.2f}, "
        f"tang={tube['tangential_energy_mean']:.2f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ensure_dirs()

    d = load_acts("direction")
    split = load_split("direction")
    train_ids, test_ids = set(split["train"]), set(split["test"])
    X_all, y_all, motion = d["layers"][:, L_STAR], d["targets"], d["motion"]
    fit_mask = np.array(
        [int(cid) in train_ids and m == "velocity" for cid, m in zip(d["ids"], motion)],
        dtype=bool,
    )
    test_idx = np.array([i for i, cid in enumerate(d["ids"]) if int(cid) in test_ids])
    X_fit, y_fit = X_all[fit_mask], y_all[fit_mask]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    print("fitting spline + tube basis...")
    spline = fit_spline(X_fit, y_fit, kind="direction")
    tube = fit_tube_basis(spline, X_fit)
    print(
        f"residual median={tube['residual_norm_median']:.3f}, "
        f"perp energy={tube['perpendicular_energy_mean']:.3f}, "
        f"cumvar@5={tube['cumulative_var_ratio'][4]:.3f}, "
        f"cumvar@20={tube['cumulative_var_ratio'][19]:.3f}"
    )
    plot_tube_variance(tube, ARTIFACTS / "figures" / "part2_tube_variance.png")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    probe = train_probe_c11(X_test, theta_to_sincos(y_test), kind="direction", device=device)
    probes = load_nullspace_probes(device)
    paths = build_paths(spline, "direction", y_fit)

    summary = {
        "L_star": L_STAR,
        "tube_stats": {k: v for k, v in tube.items() if k != "basis"},
        "paths": {},
    }
    # don't store full basis in json
    for path in paths:
        print(f"\n=== {path.name} ===")
        result = run_path(
            spline=spline,
            path=path,
            tube=tube,
            X_test=X_test,
            y_test=y_test,
            probe=probe,
            probes=probes,
            device=device,
        )
        plot_tube_curves(result, ARTIFACTS / "figures" / f"part2_tube_{path.name}.png")
        summary["paths"][path.name] = result

    # cast numpy scalars in tube_stats
    def _to_py(o):
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_py(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    summary = _to_py(summary)
    out = ARTIFACTS / "results" / "part2_tube.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
