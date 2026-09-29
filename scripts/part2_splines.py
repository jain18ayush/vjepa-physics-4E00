#!/usr/bin/env python3
"""Part 2: fit activation splines and compare spline vs chord vs multi-probe steering."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA

from common import (
    ARTIFACTS,
    ensure_dirs,
    load_split,
    score_probe,
    theta_to_sincos,
    train_probe_c11,
)
from metrics import circular_mae_degrees, mae, sincos_to_degrees
from steer import ProbeWeights, steer as multiprobe_steer

L_STAR = 5
N_PCA = 64
N_WAYPOINTS = 21
SEED = 0


@dataclass
class PathSpec:
    name: str
    u0: float
    u1: float
    label0: float
    label1: float
    note: str


def load_acts(dataset: str) -> dict:
    data = np.load(ARTIFACTS / "activations" / f"{dataset}.npz", allow_pickle=True)
    return {
        "ids": data["ids"].astype(np.int64),
        "layers": data["layers"].astype(np.float32),
        "targets": data["targets"].astype(np.float32),
        "motion": data["motion"].astype(str),
    }


def nearest_label(labels: np.ndarray, value: float) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    return float(labels[np.argmin(np.abs(labels - value))])


def circular_lerp(a: float, b: float, t: float) -> float:
    """Interpolate on the circle in [0, 1) along the shorter arc."""
    a = a % 1.0
    b = b % 1.0
    du = b - a
    if du > 0.5:
        du -= 1.0
    if du < -0.5:
        du += 1.0
    return (a + t * du) % 1.0


def circular_mean_degrees(deg: np.ndarray) -> float:
    rad = np.radians(np.asarray(deg, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0)


class ActivationSpline:
    """1D cubic spline through label centroids in a PCA subspace."""

    def __init__(
        self,
        *,
        pca: PCA,
        u: np.ndarray,
        centroids_pca: np.ndarray,
        labels: np.ndarray,
        periodic: bool,
        mean_full: np.ndarray,
    ):
        self.pca = pca
        self.u = np.asarray(u, dtype=np.float64)
        self.centroids_pca = np.asarray(centroids_pca, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=np.float64)
        self.periodic = periodic
        self.mean_full = np.asarray(mean_full, dtype=np.float64)
        bc = "periodic" if periodic else "natural"
        # Periodic CubicSpline requires first==last sample.
        if periodic:
            u_fit = np.concatenate([self.u, [1.0]])
            c_fit = np.vstack([self.centroids_pca, self.centroids_pca[0]])
        else:
            u_fit = self.u
            c_fit = self.centroids_pca
        self.spline = CubicSpline(u_fit, c_fit, axis=0, bc_type=bc)

    def eval_pca(self, u: np.ndarray | float) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        if self.periodic:
            u = u % 1.0
        out = self.spline(u)
        return out.astype(np.float64)

    def decode_full(self, z_pca: np.ndarray, leftover: np.ndarray) -> np.ndarray:
        """Inverse-PCA the subspace coords and add the off-subspace leftover."""
        z_pca = np.asarray(z_pca, dtype=np.float64)
        leftover = np.asarray(leftover, dtype=np.float64)
        recon = self.pca.inverse_transform(z_pca)
        # leftover already includes the PCA mean outside the top components,
        # so we add the reconstructed subspace part relative to the train mean.
        # Standard: x = leftover_off + recon, where leftover_off = x - P P^+ (x - mean) - mean
        # equivalently leftover = x - (recon - mean) wait...
        # We define leftover = x - pca.inverse_transform(pca.transform(x))
        # then x_hat = leftover + pca.inverse_transform(z_new)
        return (leftover + recon).astype(np.float32)

    def project(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def leftover(self, X: np.ndarray) -> np.ndarray:
        Z = self.pca.transform(X)
        return (X.astype(np.float64) - self.pca.inverse_transform(Z)).astype(np.float32)

    def residual_to_curve(self, X: np.ndarray, n_samples: int = 512) -> np.ndarray:
        """Distance in PCA space from each point to nearest sampled spline point."""
        Z = self.project(X)
        if self.periodic:
            grid = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        else:
            grid = np.linspace(float(self.u.min()), float(self.u.max()), n_samples)
        curve = self.eval_pca(grid)
        # (N, S)
        d2 = ((Z[:, None, :] - curve[None, :, :]) ** 2).sum(axis=-1)
        return np.sqrt(d2.min(axis=1))


def fit_spline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    kind: str,
) -> ActivationSpline:
    pca = PCA(n_components=min(N_PCA, X_train.shape[0] - 1, X_train.shape[1]), random_state=SEED)
    Z = pca.fit_transform(X_train)
    labels = np.unique(y_train)
    centroids = []
    kept_labels = []
    for lab in labels:
        mask = np.isclose(y_train, lab)
        if mask.sum() == 0:
            continue
        centroids.append(Z[mask].mean(axis=0))
        kept_labels.append(float(lab))
    centroids = np.asarray(centroids, dtype=np.float64)
    kept_labels = np.asarray(kept_labels, dtype=np.float64)

    if kind == "direction":
        # Intrinsic coordinate on the circle: label/360, already ordered.
        order = np.argsort(kept_labels)
        kept_labels = kept_labels[order]
        centroids = centroids[order]
        u = (kept_labels % 360.0) / 360.0
        # Ensure strictly increasing u in [0, 1); drop duplicate u if any.
        _, uniq = np.unique(np.round(u, 10), return_index=True)
        u = u[uniq]
        centroids = centroids[uniq]
        kept_labels = kept_labels[uniq]
        periodic = True
    else:
        order = np.argsort(kept_labels)
        kept_labels = kept_labels[order]
        centroids = centroids[order]
        lo, hi = float(kept_labels[0]), float(kept_labels[-1])
        u = (kept_labels - lo) / (hi - lo) if hi > lo else np.zeros_like(kept_labels)
        periodic = False

    return ActivationSpline(
        pca=pca,
        u=u,
        centroids_pca=centroids,
        labels=kept_labels,
        periodic=periodic,
        mean_full=X_train.mean(axis=0),
    )


def label_to_u(spline: ActivationSpline, label: float, kind: str) -> float:
    if kind == "direction":
        return (float(label) % 360.0) / 360.0
    lo, hi = float(spline.labels[0]), float(spline.labels[-1])
    if hi <= lo:
        return 0.0
    return float(np.clip((label - lo) / (hi - lo), 0.0, 1.0))


def u_to_label(spline: ActivationSpline, u: float, kind: str) -> float:
    if kind == "direction":
        return float((u % 1.0) * 360.0)
    lo, hi = float(spline.labels[0]), float(spline.labels[-1])
    return float(lo + u * (hi - lo))


def path_us(spline: ActivationSpline, path: PathSpec, kind: str) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, N_WAYPOINTS)
    if kind == "direction":
        return np.array([circular_lerp(path.u0, path.u1, t) for t in ts])
    return path.u0 + ts * (path.u1 - path.u0)


def claimed_labels(spline: ActivationSpline, path: PathSpec, kind: str) -> np.ndarray:
    us = path_us(spline, path, kind)
    return np.array([u_to_label(spline, u, kind) for u in us])


def build_paths(spline: ActivationSpline, kind: str, y_all: np.ndarray) -> list[PathSpec]:
    uniq = np.unique(y_all)
    if kind == "speed":
        a, b = nearest_label(uniq, 0.5), nearest_label(uniq, 3.5)
        return [
            PathSpec(
                name="speed_0.5_to_3.5",
                u0=label_to_u(spline, a, kind),
                u1=label_to_u(spline, b, kind),
                label0=a,
                label1=b,
                note="interior magnitudes",
            )
        ]
    if kind == "acceleration":
        a, b = nearest_label(uniq, 1.0), nearest_label(uniq, 8.0)
        return [
            PathSpec(
                name="accel_1_to_8",
                u0=label_to_u(spline, a, kind),
                u1=label_to_u(spline, b, kind),
                label0=a,
                label1=b,
                note="interior magnitudes",
            )
        ]
    # direction
    a, b = nearest_label(uniq, 0.0), nearest_label(uniq, 180.0)
    c, d = nearest_label(uniq, 20.0), nearest_label(uniq, 340.0)
    return [
        PathSpec(
            name="dir_0_to_180",
            u0=label_to_u(spline, a, kind),
            u1=label_to_u(spline, b, kind),
            label0=a,
            label1=b,
            note="half turn; chord crosses interior",
        ),
        PathSpec(
            name="dir_20_to_340",
            u0=label_to_u(spline, c, kind),
            u1=label_to_u(spline, d, kind),
            label0=c,
            label1=d,
            note="short arc across 0°",
        ),
    ]


@torch.no_grad()
def decode_probe(probe, X: np.ndarray, kind: str, device: torch.device) -> np.ndarray:
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    pred = probe(xt).detach().cpu().numpy()
    if kind == "direction":
        return sincos_to_degrees(pred)
    return pred.reshape(-1)


def evaluate_path(
    *,
    spline: ActivationSpline,
    path: PathSpec,
    X_test: np.ndarray,
    y_test: np.ndarray,
    motion_test: np.ndarray | None,
    probe,
    kind: str,
    device: torch.device,
) -> dict:
    leftover = spline.leftover(X_test)
    us = path_us(spline, path, kind)
    claimed = claimed_labels(spline, path, kind)
    ts = np.linspace(0.0, 1.0, N_WAYPOINTS)

    z0 = spline.eval_pca(path.u0)
    z1 = spline.eval_pca(path.u1)

    methods = {}
    for method in ("spline", "chord"):
        decoded_mean = []
        err_to_claimed = []
        err_to_truth = []
        subset_rows = []
        for t_i, (t, u, claim) in enumerate(zip(ts, us, claimed)):
            if method == "spline":
                z = spline.eval_pca(u)
            else:
                z = (1.0 - t) * z0 + t * z1
            # Broadcast one subspace point + per-clip leftover
            Z = np.broadcast_to(z, (len(X_test), z.shape[0]))
            X_s = spline.decode_full(Z, leftover)
            pred = decode_probe(probe, X_s, kind, device)
            if kind == "direction":
                decoded_mean.append(circular_mean_degrees(pred))
                err_to_claimed.append(circular_mae_degrees(pred, np.full(len(pred), claim)))
                err_to_truth.append(circular_mae_degrees(pred, y_test))
            else:
                decoded_mean.append(float(np.mean(pred)))
                err_to_claimed.append(mae(np.full(len(pred), claim), pred))
                err_to_truth.append(mae(y_test, pred))

            row = {
                "t": float(t),
                "u": float(u),
                "claimed_label": float(claim),
                "decoded_mean": float(decoded_mean[-1]),
                "err_to_claimed": float(err_to_claimed[-1]),
                "err_to_truth": float(err_to_truth[-1]),
            }
            if motion_test is not None:
                for name, mask in (
                    ("velocity", motion_test == "velocity"),
                    ("acceleration", motion_test == "acceleration"),
                ):
                    if mask.sum() == 0:
                        continue
                    if kind == "direction":
                        row[f"err_to_claimed_{name}"] = circular_mae_degrees(
                            pred[mask], np.full(mask.sum(), claim)
                        )
                        row[f"err_to_truth_{name}"] = circular_mae_degrees(pred[mask], y_test[mask])
                    else:
                        row[f"err_to_claimed_{name}"] = mae(np.full(mask.sum(), claim), pred[mask])
                        row[f"err_to_truth_{name}"] = mae(y_test[mask], pred[mask])
            subset_rows.append(row)
        methods[method] = {
            "waypoints": subset_rows,
            "final_err_to_target": float(err_to_claimed[-1]),
            "final_err_to_truth": float(err_to_truth[-1]),
            "final_decoded_mean": float(decoded_mean[-1]),
        }
    return methods


def load_nullspace_probes(L_star: int, device: torch.device) -> list[ProbeWeights]:
    path = ARTIFACTS / "probes" / f"direction_velocity_L{L_star}_nullspace.pt"
    pt = torch.load(path, map_location="cpu", weights_only=False)
    return [ProbeWeights(p["weight"], p["bias"]).to(device) for p in pt["probes"]]


def evaluate_multiprobe(
    *,
    X_test: np.ndarray,
    y_test: np.ndarray,
    motion_test: np.ndarray,
    probe,
    probes: list[ProbeWeights],
    theta_star: float,
    device: torch.device,
    n_values: list[int] | None = None,
) -> dict:
    n_values = n_values or [1, 5, 20]
    n_values = [n for n in n_values if n <= len(probes)]
    theta_star_rad = math.radians(theta_star)
    out = {"theta_star_degrees": theta_star, "by_n": []}
    # n=0 baseline
    pred0 = decode_probe(probe, X_test, "direction", device)
    out["by_n"].append(
        {
            "n": 0,
            "mae_to_target": circular_mae_degrees(pred0, np.full(len(pred0), theta_star)),
            "mae_to_truth": circular_mae_degrees(pred0, y_test),
            "decoded_mean": circular_mean_degrees(pred0),
        }
    )
    for n in n_values:
        probes_n = probes[:n]
        steered = np.zeros_like(X_test)
        for i in range(len(X_test)):
            x = torch.from_numpy(X_test[i]).to(device)
            steered[i] = multiprobe_steer(x, probes_n, theta_star_rad).detach().cpu().numpy()
        pred = decode_probe(probe, steered, "direction", device)
        row = {
            "n": n,
            "mae_to_target": circular_mae_degrees(pred, np.full(len(pred), theta_star)),
            "mae_to_truth": circular_mae_degrees(pred, y_test),
            "decoded_mean": circular_mean_degrees(pred),
        }
        for name, mask in (
            ("velocity", motion_test == "velocity"),
            ("acceleration", motion_test == "acceleration"),
        ):
            if mask.sum() == 0:
                continue
            row[f"mae_to_target_{name}"] = circular_mae_degrees(
                pred[mask], np.full(mask.sum(), theta_star)
            )
            row[f"mae_to_truth_{name}"] = circular_mae_degrees(pred[mask], y_test[mask])
        out["by_n"].append(row)
        print(f"  multiprobe n={n} → {theta_star:g}°: MAE→target={row['mae_to_target']:.2f}")
    return out


def plot_pca_overlay(spline: ActivationSpline, X: np.ndarray, y: np.ndarray, kind: str, path: PathSpec, out: Path):
    Z = spline.project(X)
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, cmap="turbo", s=8, alpha=0.55)
    if spline.periodic:
        grid = np.linspace(0, 1, 200, endpoint=False)
    else:
        grid = np.linspace(float(spline.u.min()), float(spline.u.max()), 200)
    curve = spline.eval_pca(grid)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#0f766e", lw=2.4, label="spline")
    z0 = spline.eval_pca(path.u0)
    z1 = spline.eval_pca(path.u1)
    ax.plot(
        [z0[0], z1[0]],
        [z0[1], z1[1]],
        [z0[2], z1[2]],
        color="#9f1239",
        lw=2.0,
        ls="--",
        label="chord",
    )
    ax.scatter([z0[0], z1[0]], [z0[1], z1[1]], [z0[2], z1[2]], c="#142028", s=40)
    ax.set_title(f"{kind} L{L_STAR} PCA — {path.name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="label")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_decode_curves(kind: str, path: PathSpec, methods: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for method, color in (("spline", "#0f766e"), ("chord", "#9f1239")):
        rows = methods[method]["waypoints"]
        ts = [r["t"] for r in rows]
        claimed = [r["claimed_label"] for r in rows]
        decoded = [r["decoded_mean"] for r in rows]
        err_c = [r["err_to_claimed"] for r in rows]
        err_t = [r["err_to_truth"] for r in rows]
        axes[0].plot(ts, decoded, color=color, marker="o", ms=3, label=f"{method} decode")
        if method == "spline":
            axes[0].plot(ts, claimed, color="#142028", ls=":", label="claimed label")
        axes[1].plot(ts, err_c, color=color, marker="o", ms=3, label=f"{method} → claimed")
        axes[1].plot(ts, err_t, color=color, marker="x", ms=3, ls="--", label=f"{method} → truth")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("decoded mean")
    axes[0].set_title(f"{kind}: {path.name}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("error" + (" (°)" if kind == "direction" else ""))
    axes[1].set_title("error along path")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_comparison_bar(summary: dict, out: Path):
    """Final error-to-target bars for all paths + multiprobe."""
    labels = []
    spline_vals = []
    chord_vals = []
    for var, block in summary["variables"].items():
        for path in block["paths"]:
            labels.append(path["name"])
            spline_vals.append(path["methods"]["spline"]["final_err_to_target"])
            chord_vals.append(path["methods"]["chord"]["final_err_to_target"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, spline_vals, w, label="spline", color="#0f766e")
    ax.bar(x + w / 2, chord_vals, w, label="chord", color="#9f1239")
    # multiprobe markers for direction paths
    mp = summary.get("multiprobe_same_target", {})
    for i, name in enumerate(labels):
        if name in mp:
            ax.scatter(
                [i],
                [mp[name]["by_n"][-1]["mae_to_target"]],
                marker="D",
                s=55,
                color="#1d4ed8",
                zorder=5,
                label="multi-probe n=20" if i == 0 or "multi-probe n=20" not in ax.get_legend_handles_labels()[1] else None,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("final error to path target")
    ax.set_title("Spline vs chord vs multi-probe (direction only)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_variable(kind: str, device: torch.device) -> dict:
    print(f"\n=== {kind} ===")
    d = load_acts(kind)
    split = load_split(kind)
    train_ids = set(split["train"])
    test_ids = set(split["test"])
    assert train_ids.isdisjoint(test_ids)

    L = L_STAR
    train_idx = np.array([i for i, cid in enumerate(d["ids"]) if int(cid) in train_ids], dtype=np.int64)
    test_idx = np.array([i for i, cid in enumerate(d["ids"]) if int(cid) in test_ids], dtype=np.int64)

    X_all = d["layers"][:, L]
    y_all = d["targets"]
    motion = d["motion"]

    if kind == "direction":
        # Fit on train / velocity only
        fit_mask = np.array(
            [int(cid) in train_ids and m == "velocity" for cid, m in zip(d["ids"], motion)],
            dtype=bool,
        )
        X_fit = X_all[fit_mask]
        y_fit = y_all[fit_mask]
    else:
        X_fit = X_all[train_idx]
        y_fit = y_all[train_idx]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    motion_test = motion[test_idx] if kind == "direction" else None

    spline = fit_spline(X_fit, y_fit, kind=kind)
    var_explained = float(spline.pca.explained_variance_ratio_.sum())
    print(
        f"PCA comps={spline.pca.n_components_}, var={var_explained:.4f}, "
        f"centroids={len(spline.labels)}, fit_n={len(X_fit)}"
    )

    train_res = spline.residual_to_curve(X_fit)
    test_res = spline.residual_to_curve(X_test)
    print(
        f"residual to curve: train median={np.median(train_res):.3f}, "
        f"test median={np.median(test_res):.3f}"
    )

    # Eval probe on unsteered test only
    if kind == "direction":
        y_probe = theta_to_sincos(y_test)
    else:
        y_probe = y_test
    probe = train_probe_c11(X_test, y_probe, kind=kind, device=device)
    base_metrics = score_probe(probe, X_test, y_test, kind=kind, device=device)
    print("eval probe unsteered test:", base_metrics)

    paths = build_paths(spline, kind, y_fit if kind == "direction" else y_all)
    path_results = []
    for path in paths:
        print(f"path {path.name}: {path.label0} → {path.label1}")
        methods = evaluate_path(
            spline=spline,
            path=path,
            X_test=X_test,
            y_test=y_test,
            motion_test=motion_test,
            probe=probe,
            kind=kind,
            device=device,
        )
        for mname, m in methods.items():
            print(
                f"  {mname}: final→target={m['final_err_to_target']:.3f}, "
                f"final→truth={m['final_err_to_truth']:.3f}, "
                f"decoded={m['final_decoded_mean']:.3f}"
            )
        plot_pca_overlay(
            spline,
            X_fit if kind != "direction" else X_all[fit_mask],
            y_fit,
            kind,
            path,
            ARTIFACTS / "figures" / f"part2_pca_{path.name}.png",
        )
        plot_decode_curves(
            kind,
            path,
            methods,
            ARTIFACTS / "figures" / f"part2_decode_{path.name}.png",
        )
        path_results.append(
            {
                **asdict(path),
                "methods": methods,
            }
        )

    # Accelerating direction transfer: evaluate first path on accel-only test
    transfer = None
    if kind == "direction":
        accel_mask = motion_test == "acceleration"
        if accel_mask.any():
            path = paths[0]
            methods = evaluate_path(
                spline=spline,
                path=path,
                X_test=X_test[accel_mask],
                y_test=y_test[accel_mask],
                motion_test=motion_test[accel_mask],
                probe=probe,
                kind=kind,
                device=device,
            )
            transfer = {
                "path": path.name,
                "n": int(accel_mask.sum()),
                "spline_final_err_to_target": methods["spline"]["final_err_to_target"],
                "chord_final_err_to_target": methods["chord"]["final_err_to_target"],
            }
            print("accel-only transfer:", transfer)

    return {
        "kind": kind,
        "L_star": L,
        "n_pca": int(spline.pca.n_components_),
        "variance_explained": var_explained,
        "n_centroids": int(len(spline.labels)),
        "labels": spline.labels.tolist(),
        "fit_n": int(len(X_fit)),
        "test_n": int(len(X_test)),
        "residual_train_median": float(np.median(train_res)),
        "residual_test_median": float(np.median(test_res)),
        "residual_train_mean": float(np.mean(train_res)),
        "residual_test_mean": float(np.mean(test_res)),
        "eval_probe_unsteered": base_metrics,
        "paths": path_results,
        "accel_transfer": transfer,
        "pca_components_for_plot": 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ensure_dirs()
    (ARTIFACTS / "figures").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "results").mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "L_star": L_STAR,
        "n_pca": N_PCA,
        "n_waypoints": N_WAYPOINTS,
        "variables": {},
        "part1_multiprobe_quote_90": None,
        "multiprobe_same_target": {},
    }

    # Quote Part 1 multi-probe → 90°
    steer_path = ARTIFACTS / "results" / "steering.json"
    if steer_path.exists():
        part1 = json.loads(steer_path.read_text())
        quote = []
        for row in part1["table"]:
            if row["n"] in (0, 1, 5, 20) and "all" in row["subsets"]:
                quote.append(
                    {
                        "n": row["n"],
                        "mae_to_target": row["subsets"]["all"]["mae_to_target"],
                        "mae_to_truth": row["subsets"]["all"]["mae_to_truth"],
                    }
                )
        summary["part1_multiprobe_quote_90"] = {
            "theta_star_degrees": part1["theta_star_degrees"],
            "eval_probe_unsteered": part1["eval_probe_unsteered"],
            "rows": quote,
        }

    for kind in ("speed", "acceleration", "direction"):
        summary["variables"][kind] = run_variable(kind, device)

    # Fair multi-probe comparison toward each direction path endpoint
    print("\n=== multi-probe same-target (direction) ===")
    d = load_acts("direction")
    split = load_split("direction")
    test_ids = set(split["test"])
    test_idx = np.array([i for i, cid in enumerate(d["ids"]) if int(cid) in test_ids], dtype=np.int64)
    X_test = d["layers"][test_idx, L_STAR]
    y_test = d["targets"][test_idx]
    motion_test = d["motion"][test_idx]
    y_probe = theta_to_sincos(y_test)
    probe = train_probe_c11(X_test, y_probe, kind="direction", device=device)
    probes = load_nullspace_probes(L_STAR, device)
    for path in summary["variables"]["direction"]["paths"]:
        mp = evaluate_multiprobe(
            X_test=X_test,
            y_test=y_test,
            motion_test=motion_test,
            probe=probe,
            probes=probes,
            theta_star=float(path["label1"]),
            device=device,
        )
        summary["multiprobe_same_target"][path["name"]] = mp

    plot_comparison_bar(summary, ARTIFACTS / "figures" / "part2_comparison_bar.png")

    out = ARTIFACTS / "results" / "part2_splines.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
