#!/usr/bin/env python3
"""Tube-why campaign: fiber, nuisance, copies, radius/belief, conditional spine, layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from common import ARTIFACTS, ensure_dirs, load_split, theta_to_sincos, train_probe_c11
from metrics import circular_mae_degrees, sincos_to_degrees
from part2_splines import L_STAR, ActivationSpline, fit_spline, load_acts
from part2_tube import fit_tube_basis, nearest_on_spline, pca_mats, load_nullspace_probes
from steer import ProbeWeights

SEED = 0
LAYERS = [0, 5, 12, 23]
EPS_GRID = np.linspace(-3.0, 3.0, 13)


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


def load_meta() -> dict[int, dict]:
    path = ARTIFACTS / "meta" / "direction_meta.json"
    if not path.exists():
        # fallback: derive what we can from npz only
        return {}
    rows = json.loads(path.read_text())
    return {int(r["id"]): r for r in rows}


def decode_from_z(z_pca, leftover, components, mean):
    recon = z_pca @ components + mean
    return (leftover + recon).astype(np.float32)


def fit_eval_probe(X, y, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    return train_probe_c11(X, theta_to_sincos(y), kind="direction", device=device)


def ridge_r2(X, y, cv=5):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pred = np.zeros_like(y)
    for j in range(y.shape[1]):
        model = Ridge(alpha=1.0)
        pred[:, j] = cross_val_predict(model, Xs, y[:, j], cv=min(cv, len(X) // 2 or 2))
    return float(r2_score(y, pred, multioutput="uniform_average"))


def subspace_principal_angles_deg(A: np.ndarray, B: np.ndarray) -> list[float]:
    """A, B as orthonormal bases with shape (d, k)."""
    if A.size == 0 or B.size == 0:
        return []
    # thin QR
    Qa, _ = np.linalg.qr(A, mode="reduced")
    Qb, _ = np.linalg.qr(B, mode="reduced")
    _, s, _ = np.linalg.svd(Qa.T @ Qb)
    s = np.clip(s, 0, 1)
    return [float(np.degrees(np.arccos(si))) for si in s]


# ---------- Step 1a: fiber test ----------
def step_fiber(spline, tube, X_fit, y_fit, leftover_fit, probe, device, components, mean):
    Z = spline.project(X_fit)
    u_star, z_star, tang, _ = nearest_on_spline(Z, spline)
    # pick a mid-circle angle near 90°
    target_u = 90.0 / 360.0
    idx = int(np.argmin(np.abs(((u_star - target_u + 0.5) % 1.0) - 0.5)))
    z0 = z_star[idx]
    tvec = tang[idx]
    leftover = leftover_fit[idx : idx + 1]
    y0 = float(y_fit[idx])

    basis = tube["basis"]  # (n, n_pca)
    results = {"anchor_theta": y0, "anchor_u": float(u_star[idx]), "curves": {}}

    # spine direction (unit tangent)
    dirs = {"tangent": tvec}
    for k in range(3):
        dirs[f"tube_pc{k+1}"] = basis[k]

    for name, v in dirs.items():
        v = v / (np.linalg.norm(v) + 1e-12)
        dec, err = [], []
        for eps in EPS_GRID:
            z = z0 + eps * v
            X = decode_from_z(z, leftover, components, mean)
            pred = probe(torch.from_numpy(X).to(device)).detach().cpu().numpy()
            deg = float(sincos_to_degrees(pred)[0])
            dec.append(deg)
            err.append(circular_mae_degrees(np.array([deg]), np.array([y0])))
        # sensitivity: deg change per unit eps near 0
        mid = len(EPS_GRID) // 2
        sens = abs(dec[mid + 1] - dec[mid - 1]) / (EPS_GRID[mid + 1] - EPS_GRID[mid - 1])
        results["curves"][name] = {
            "eps": EPS_GRID.tolist(),
            "decoded_deg": dec,
            "err_to_anchor": err,
            "sensitivity_deg_per_eps": float(sens),
        }
    return results


# ---------- Step 1b: nuisance audit ----------
def step_nuisance(spline, tube, X_fit, y_fit, ids_fit, motion_fit, start_xy_fit, meta_by_id):
    Z = spline.project(X_fit)
    u_star, z_star, tang, rho = nearest_on_spline(Z, spline)
    r = Z - z_star
    r_perp = r - (np.sum(r * tang, axis=1, keepdims=True) * tang)
    # tube scores
    mu = tube["mean_residual"]
    scores = (r_perp - mu) @ tube["basis"][:10].T  # (N, 10)
    spine_feat = np.stack([np.sin(2 * np.pi * u_star), np.cos(2 * np.pi * u_star)], axis=1)

    speed = np.array(
        [meta_by_id[int(i)]["speed_mps"] if int(i) in meta_by_id else np.nan for i in ids_fit]
    )
    accel = np.array(
        [
            meta_by_id[int(i)]["acceleration_mps2"] if int(i) in meta_by_id else np.nan
            for i in ids_fit
        ]
    )
    # if meta missing, skip those targets
    targets = {
        "theta_sincos": theta_to_sincos(y_fit),
        "start_x": start_xy_fit[:, 0],
        "start_y": start_xy_fit[:, 1],
        "start_xy": start_xy_fit,
        "speed_mps": speed,
        "acceleration_mps2": accel,
    }
    out = {"r2_spine": {}, "r2_tube10": {}, "r2_spine_plus_tube10": {}, "motion_acc": {}}

    for name, y in targets.items():
        y = np.asarray(y, dtype=np.float64)
        if np.isnan(y).any():
            mask = ~np.isnan(y) if y.ndim == 1 else ~np.isnan(y).any(axis=1)
            if mask.sum() < 50:
                continue
            Xs, Xt, Xp, yy = spine_feat[mask], scores[mask], np.hstack([spine_feat[mask], scores[mask]]), y[mask]
        else:
            Xs, Xt, Xp, yy = spine_feat, scores, np.hstack([spine_feat, scores]), y
        out["r2_spine"][name] = ridge_r2(Xs, yy)
        out["r2_tube10"][name] = ridge_r2(Xt, yy)
        out["r2_spine_plus_tube10"][name] = ridge_r2(Xp, yy)

    # motion classification
    y_m = (motion_fit == "acceleration").astype(int)
    for label, X in ("spine", spine_feat), ("tube10", scores), ("both", np.hstack([spine_feat, scores])):
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=1000)
        pred = cross_val_predict(clf, Xs, y_m, cv=5)
        out["motion_acc"][label] = float(accuracy_score(y_m, pred))

    out["rho_mean"] = float(np.mean(rho))
    out["rho_std"] = float(np.std(rho))
    return out, scores, spine_feat, rho, u_star, r_perp


# ---------- Step 2: copy alignment ----------
def step_copy_alignment(tube, probes, components, tangent_ref):
    # nullspace probe planes in PCA
    rows = []
    for p in probes:
        w = p.weight.detach().cpu().numpy().astype(np.float64)  # (2, d)
        rows.append(w)
    W = np.concatenate(rows, axis=0)  # (2n, d)
    Wp = (components @ W.T).T  # (2n, n_pca)
    t = tangent_ref / (np.linalg.norm(tangent_ref) + 1e-12)
    Wp = Wp - (Wp @ t)[:, None] * t[None, :]

    tube_basis = tube["basis"][:20].T  # (n_pca, 20)
    out = {"angles_to_tube20_by_probe_pair": [], "cum_overlap_energy": []}
    # each probe is a 2-row plane
    for i in range(len(probes)):
        plane = Wp[2 * i : 2 * i + 2].T  # (n_pca, 2)
        # remove near-zero
        if np.linalg.norm(plane) < 1e-8:
            continue
        angles = subspace_principal_angles_deg(plane, tube_basis)
        # overlap: || P_tube plane || / ||plane||
        Q, _ = np.linalg.qr(tube_basis, mode="reduced")
        proj = Q @ (Q.T @ plane)
        overlap = float(np.linalg.norm(proj) / (np.linalg.norm(plane) + 1e-12))
        out["angles_to_tube20_by_probe_pair"].append(
            {"probe_index": i, "principal_angles_deg": angles, "overlap_fraction": overlap}
        )

    # energy of successive probe planes captured by first k tube PCs
    for k in [1, 2, 5, 10, 20]:
        Q = tube["basis"][:k].T
        Q, _ = np.linalg.qr(Q, mode="reduced")
        fracs = []
        for i in range(min(20, len(probes))):
            plane = Wp[2 * i : 2 * i + 2].T
            proj = Q @ (Q.T @ plane)
            fracs.append(float(np.linalg.norm(proj) ** 2 / (np.linalg.norm(plane) ** 2 + 1e-12)))
        out["cum_overlap_energy"].append({"k": k, "mean_overlap_sq": float(np.mean(fracs)), "per_probe": fracs})
    return out


# ---------- Step 3: radius + belief ----------
def belief_softmax(z, centroids, tau=0.5):
    # z: (n_pca,), centroids: (C, n_pca)
    d2 = np.sum((centroids - z[None, :]) ** 2, axis=1)
    logits = -d2 / tau
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return p


def step_radius_belief(spline, tube, X_fit, y_fit, scores, rho, u_star, probe, device, components, mean, leftover_fit):
    # probe confidence: ||sincos pred||
    X = X_fit
    pred = probe(torch.from_numpy(X.astype(np.float32)).to(device)).detach().cpu().numpy()
    conf = np.linalg.norm(pred, axis=1)
    pred_deg = sincos_to_degrees(pred)
    err = np.abs((pred_deg - y_fit + 180) % 360 - 180)

    # belief entropy at each point's PCA coord
    centroids = spline.centroids_pca
    labels = spline.labels
    Z = spline.project(X_fit)
    ents = []
    belief_mode_err = []
    for z, yt in zip(Z, y_fit):
        p = belief_softmax(z, centroids, tau=0.5)
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        ents.append(ent)
        mode = float(labels[int(np.argmax(p))])
        belief_mode_err.append(abs((mode - yt + 180) % 360 - 180))
    ents = np.asarray(ents)
    belief_mode_err = np.asarray(belief_mode_err)

    def corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    out = {
        "corr_rho_confidence": corr(rho, conf),
        "corr_rho_probe_err": corr(rho, err),
        "corr_rho_belief_entropy": corr(rho, ents),
        "corr_rho_belief_mode_err": corr(rho, belief_mode_err),
        "corr_tube_pc1_confidence": corr(scores[:, 0], conf),
        "corr_tube_pc1_entropy": corr(scores[:, 0], ents),
        "mean_entropy": float(ents.mean()),
        "mean_rho": float(rho.mean()),
    }

    # fiber belief: at fixed u near 90°, move along tube PC1 and measure entropy + mode
    target_u = 90.0 / 360.0
    idx = int(np.argmin(np.abs(((u_star - target_u + 0.5) % 1.0) - 0.5)))
    z0 = spline.eval_pca(u_star[idx])
    v = tube["basis"][0]
    v = v / (np.linalg.norm(v) + 1e-12)
    fiber = []
    for eps in EPS_GRID:
        z = z0 + eps * v
        p = belief_softmax(z, centroids, tau=0.5)
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        mode = float(labels[int(np.argmax(p))])
        leftover = leftover_fit[idx : idx + 1]
        X = decode_from_z(z, leftover, components, mean)
        pr = probe(torch.from_numpy(X).to(device)).detach().cpu().numpy()
        deg = float(sincos_to_degrees(pr)[0])
        fiber.append(
            {
                "eps": float(eps),
                "entropy": ent,
                "belief_mode": mode,
                "probe_deg": deg,
            }
        )
    out["fiber_belief_tube_pc1"] = fiber
    out["_arrays_for_plots"] = {
        "rho": rho,
        "conf": conf,
        "err": err,
        "ents": ents,
    }
    return out


# ---------- Step 4: conditional spine ----------
def step_conditional(X_fit, y_fit, start_xy_fit, motion_fit, meta_by_id, ids_fit):
    # regress out start position / speed from activations, refit spline, measure tube energy
    def tube_energy_after(X):
        sp = fit_spline(X, y_fit, kind="direction")
        tb = fit_tube_basis(sp, X)
        return {
            "perp_energy": tb["perpendicular_energy_mean"],
            "tang_energy": tb["tangential_energy_mean"],
            "resid_median": tb["residual_norm_median"],
            "cumvar5": tb["cumulative_var_ratio"][4],
            "cumvar20": tb["cumulative_var_ratio"][19],
        }

    def residualize(X, features):
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim == 1:
            feats = feats[:, None]
        mask = ~np.isnan(feats).any(axis=1)
        Xs = StandardScaler().fit_transform(feats[mask])
        X_res = np.array(X, dtype=np.float64, copy=True)
        for j in range(X.shape[1]):
            reg = Ridge(alpha=1.0).fit(Xs, X[mask, j])
            X_res[mask, j] = X[mask, j] - reg.predict(Xs)
        return X_res.astype(np.float32)

    baseline = tube_energy_after(X_fit)
    after_xy = tube_energy_after(residualize(X_fit, start_xy_fit))

    speed = np.array(
        [meta_by_id[int(i)]["speed_mps"] if int(i) in meta_by_id else np.nan for i in ids_fit]
    )
    after_speed = tube_energy_after(residualize(X_fit, speed))
    after_xy_speed = tube_energy_after(
        residualize(X_fit, np.hstack([start_xy_fit, speed[:, None]]))
    )

    # within-angle energy (no spine)
    X_within = np.zeros_like(X_fit, dtype=np.float64)
    for lab in np.unique(y_fit):
        m = np.isclose(y_fit, lab)
        X_within[m] = X_fit[m] - X_fit[m].mean(axis=0)
    within_energy = float(np.mean(np.sum(X_within**2, axis=1)))

    return {
        "baseline": baseline,
        "after_regress_out_start_xy": after_xy,
        "after_regress_out_speed": after_speed,
        "after_regress_out_xy_and_speed": after_xy_speed,
        "perp_energy_ratio_after_xy": after_xy["perp_energy"] / (baseline["perp_energy"] + 1e-12),
        "perp_energy_ratio_after_speed": after_speed["perp_energy"] / (baseline["perp_energy"] + 1e-12),
        "perp_energy_ratio_after_xy_speed": after_xy_speed["perp_energy"]
        / (baseline["perp_energy"] + 1e-12),
        "within_angle_activation_energy": within_energy,
        "n_velocity": int((motion_fit == "velocity").sum()),
        "n_acceleration_in_fit": int((motion_fit == "acceleration").sum()),
    }


# ---------- Step 5: layer filmstrip ----------
def step_layers(d, fit_mask, device):
    rows = []
    X_all = d["layers"]
    y = d["targets"]
    for L in LAYERS:
        X = X_all[fit_mask, L]
        yf = y[fit_mask]
        sp = fit_spline(X, yf, kind="direction")
        tb = fit_tube_basis(sp, X)
        # fiber sensitivity at this layer
        components, mean = pca_mats(sp)
        leftover = sp.leftover(X)
        probe = fit_eval_probe(X, yf, device)
        tube = tb
        Z = sp.project(X)
        u_star, z_star, tang, rho = nearest_on_spline(Z, sp)
        idx = int(np.argmin(np.abs(((u_star - 0.25 + 0.5) % 1.0) - 0.5)))  # ~90°
        z0 = z_star[idx]
        tvec = tang[idx]
        v_tube = tb["basis"][0]
        leftover_i = leftover[idx : idx + 1]
        y0 = float(yf[idx])

        def sens(v):
            v = v / (np.linalg.norm(v) + 1e-12)
            deg = []
            for eps in (-0.5, 0.5):
                Xd = decode_from_z(z0 + eps * v, leftover_i, components, mean)
                pr = probe(torch.from_numpy(Xd).to(device)).detach().cpu().numpy()
                deg.append(float(sincos_to_degrees(pr)[0]))
            return abs(deg[1] - deg[0])

        rows.append(
            {
                "layer": L,
                "pca_var": float(sp.pca.explained_variance_ratio_.sum()),
                "perp_energy": tb["perpendicular_energy_mean"],
                "tang_energy": tb["tangential_energy_mean"],
                "resid_median": tb["residual_norm_median"],
                "cumvar5": tb["cumulative_var_ratio"][4],
                "cumvar20": tb["cumulative_var_ratio"][19],
                "fiber_sens_tangent": float(sens(tvec)),
                "fiber_sens_tube_pc1": float(sens(v_tube)),
                "rho_mean": float(rho.mean()),
            }
        )
        print(
            f"  L{L}: perp={rows[-1]['perp_energy']:.2f} "
            f"sens_T={rows[-1]['fiber_sens_tangent']:.2f} sens_tube={rows[-1]['fiber_sens_tube_pc1']:.2f}"
        )
    return rows


# ---------- plots ----------
def plot_fiber(fiber, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, color in (
        ("tangent", "#9f1239"),
        ("tube_pc1", "#0f766e"),
        ("tube_pc2", "#1d4ed8"),
        ("tube_pc3", "#d89a1a"),
    ):
        c = fiber["curves"][name]
        ax.plot(c["eps"], c["err_to_anchor"], color=color, marker="o", ms=3, label=f"{name} (sens={c['sensitivity_deg_per_eps']:.2f})")
    ax.set_xlabel("ε along unit direction")
    ax.set_ylabel("circular error to anchor θ (°)")
    ax.set_title(f"Fiber test at θ≈{fiber['anchor_theta']:.1f}°")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_nuisance(nuis, out):
    keys = [k for k in ("theta_sincos", "start_x", "start_y", "speed_mps", "acceleration_mps2") if k in nuis["r2_spine"]]
    x = np.arange(len(keys))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - w, [nuis["r2_spine"][k] for k in keys], w, label="spine (sin/cos u)", color="#142028")
    ax.bar(x, [nuis["r2_tube10"][k] for k in keys], w, label="tube PC1–10", color="#0f766e")
    ax.bar(x + w, [nuis["r2_spine_plus_tube10"][k] for k in keys], w, label="spine+tube", color="#1d4ed8")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15, ha="right")
    ax.set_ylabel("cross-val R²")
    ax.set_title("Nuisance audit: what do spine vs tube predict?")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_copy(copy, out):
    overlaps = [r["overlap_fraction"] for r in copy["angles_to_tube20_by_probe_pair"][:41]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(overlaps, color="#0f766e")
    axes[0].set_xlabel("nullspace probe index")
    axes[0].set_ylabel("overlap with tube20")
    axes[0].set_title("Probe plane ⊂ tube?")
    axes[0].grid(True, alpha=0.3)
    ks = [r["k"] for r in copy["cum_overlap_energy"]]
    means = [r["mean_overlap_sq"] for r in copy["cum_overlap_energy"]]
    axes[1].plot(ks, means, "o-", color="#1d4ed8")
    axes[1].set_xlabel("tube PCs kept")
    axes[1].set_ylabel("mean squared overlap (first 20 probes)")
    axes[1].set_title("Copy energy captured by tube prefix")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_radius(radius, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    arr = radius["_arrays_for_plots"]
    axes[0].scatter(arr["rho"], arr["ents"], s=8, alpha=0.35, c="#0f766e")
    axes[0].set_xlabel("tube radius ρ")
    axes[0].set_ylabel("belief entropy")
    axes[0].set_title(f"ρ vs entropy (r={radius['corr_rho_belief_entropy']:.2f})")
    axes[0].grid(True, alpha=0.3)
    fiber = radius["fiber_belief_tube_pc1"]
    axes[1].plot([f["eps"] for f in fiber], [f["entropy"] for f in fiber], color="#0f766e", label="entropy")
    ax2 = axes[1].twinx()
    ax2.plot([f["eps"] for f in fiber], [f["probe_deg"] for f in fiber], color="#9f1239", label="probe θ")
    axes[1].set_xlabel("ε along tube PC1")
    axes[1].set_ylabel("belief entropy")
    ax2.set_ylabel("probe deg")
    axes[1].set_title("Belief along tube fiber")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_layers(layer_rows, out):
    Ls = [r["layer"] for r in layer_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(Ls, [r["perp_energy"] for r in layer_rows], "o-", color="#0f766e", label="perp energy")
    axes[0].plot(Ls, [r["tang_energy"] for r in layer_rows], "s--", color="#9f1239", label="tang energy")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("energy")
    axes[0].set_title("Tube thickness over depth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(Ls, [r["fiber_sens_tangent"] for r in layer_rows], "o-", color="#9f1239", label="sens tangent")
    axes[1].plot(Ls, [r["fiber_sens_tube_pc1"] for r in layer_rows], "s-", color="#0f766e", label="sens tube PC1")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("Δθ for Δε=1 (from ±0.5)")
    axes[1].set_title("Fiber test over depth")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ensure_dirs()
    (ARTIFACTS / "figures").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "results").mkdir(parents=True, exist_ok=True)

    meta_by_id = load_meta()
    print(f"meta rows: {len(meta_by_id)}")

    d = load_acts("direction")
    split = load_split("direction")
    train_ids = set(split["train"])
    fit_mask = np.array(
        [int(cid) in train_ids and m == "velocity" for cid, m in zip(d["ids"], d["motion"])],
        dtype=bool,
    )
    X_fit = d["layers"][fit_mask, L_STAR]
    y_fit = d["targets"][fit_mask]
    ids_fit = d["ids"][fit_mask]
    motion_fit = d["motion"][fit_mask]
    if d.get("start_xy") is not None:
        start_xy_all = d["start_xy"]
    else:
        start_xy_all = np.array(
            [
                [meta_by_id[int(i)]["start_x"], meta_by_id[int(i)]["start_y"]]
                if int(i) in meta_by_id
                else [np.nan, np.nan]
                for i in d["ids"]
            ],
            dtype=np.float32,
        )
    start_xy_fit = start_xy_all[fit_mask]

    print("fitting L5 spline + tube...")
    spline = fit_spline(X_fit, y_fit, kind="direction")
    tube = fit_tube_basis(spline, X_fit)
    components, mean = pca_mats(spline)
    leftover_fit = spline.leftover(X_fit)
    probe = fit_eval_probe(X_fit, y_fit, device)
    probes = load_nullspace_probes(device)

    # Nuisance audit on *all train* clips (both motions), using velocity-fit spine/tube
    train_mask = np.array([int(cid) in train_ids for cid in d["ids"]], dtype=bool)
    X_tr = d["layers"][train_mask, L_STAR]
    y_tr = d["targets"][train_mask]
    ids_tr = d["ids"][train_mask]
    motion_tr = d["motion"][train_mask]
    start_xy_tr = start_xy_all[train_mask]

    # mean tangent at 90° for copy alignment
    Z = spline.project(X_fit)
    u_star, z_star, tang, rho = nearest_on_spline(Z, spline)
    idx90 = int(np.argmin(np.abs(((u_star - 0.25 + 0.5) % 1.0) - 0.5)))
    tangent_ref = tang[idx90]

    print("\n=== Step 1: fiber + nuisance ===")
    fiber = step_fiber(spline, tube, X_fit, y_fit, leftover_fit, probe, device, components, mean)
    for name, c in fiber["curves"].items():
        print(f"  {name}: sensitivity={c['sensitivity_deg_per_eps']:.3f} deg/eps")
    nuis, scores, spine_feat, rho, u_star, r_perp = step_nuisance(
        spline, tube, X_tr, y_tr, ids_tr, motion_tr, start_xy_tr, meta_by_id
    )
    print("  R² theta_sincos spine/tube", nuis["r2_spine"].get("theta_sincos"), nuis["r2_tube10"].get("theta_sincos"))
    print("  R² start_xy spine/tube", nuis["r2_spine"].get("start_xy"), nuis["r2_tube10"].get("start_xy"))
    print("  R² speed spine/tube", nuis["r2_spine"].get("speed_mps"), nuis["r2_tube10"].get("speed_mps"))
    print("  motion acc", nuis["motion_acc"])
    plot_fiber(fiber, ARTIFACTS / "figures" / "tube_why_fiber.png")
    plot_nuisance(nuis, ARTIFACTS / "figures" / "tube_why_nuisance.png")

    print("\n=== Step 2: copy alignment ===")
    copy = step_copy_alignment(tube, probes, components, tangent_ref)
    print("  first 5 probe overlaps:", [round(r["overlap_fraction"], 3) for r in copy["angles_to_tube20_by_probe_pair"][:5]])
    print("  tube k capture:", [(r["k"], round(r["mean_overlap_sq"], 3)) for r in copy["cum_overlap_energy"]])
    plot_copy(copy, ARTIFACTS / "figures" / "tube_why_copy.png")

    print("\n=== Step 3: radius + belief ===")
    leftover_tr = spline.leftover(X_tr)
    radius = step_radius_belief(
        spline, tube, X_tr, y_tr, scores, rho, u_star, probe, device, components, mean, leftover_tr
    )
    print(
        f"  corr rho-entropy={radius['corr_rho_belief_entropy']:.3f} "
        f"rho-conf={radius['corr_rho_confidence']:.3f} rho-err={radius['corr_rho_probe_err']:.3f}"
    )
    plot_radius(radius, ARTIFACTS / "figures" / "tube_why_radius.png")
    # strip arrays from json
    radius_save = {k: v for k, v in radius.items() if k != "_arrays_for_plots"}

    print("\n=== Step 4: conditional spine ===")
    cond = step_conditional(X_fit, y_fit, start_xy_fit, motion_fit, meta_by_id, ids_fit)
    print(
        f"  perp energy baseline={cond['baseline']['perp_energy']:.3f}\n"
        f"  after start_xy={cond['after_regress_out_start_xy']['perp_energy']:.3f} "
        f"(ratio {cond['perp_energy_ratio_after_xy']:.3f})\n"
        f"  after speed={cond['after_regress_out_speed']['perp_energy']:.3f} "
        f"(ratio {cond['perp_energy_ratio_after_speed']:.3f})\n"
        f"  after xy+speed={cond['after_regress_out_xy_and_speed']['perp_energy']:.3f} "
        f"(ratio {cond['perp_energy_ratio_after_xy_speed']:.3f})"
    )

    print("\n=== Step 5: layer filmstrip ===")
    layers = step_layers(d, fit_mask, device)
    plot_layers(layers, ARTIFACTS / "figures" / "tube_why_layers.png")

    summary = {
        "L_star": L_STAR,
        "n_fit": int(fit_mask.sum()),
        "tube_stats": {k: v for k, v in tube.items() if k not in ("basis", "mean_residual", "variance")},
        "step1_fiber": fiber,
        "step1_nuisance": nuis,
        "step2_copy": copy,
        "step3_radius_belief": radius_save,
        "step4_conditional": cond,
        "step5_layers": layers,
    }
    out = ARTIFACTS / "results" / "tube_why_campaign.json"
    out.write_text(json.dumps(_to_py(summary), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
