#!/usr/bin/env python3
"""Step 2: layer-wise linear probes and layer selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from common import (
    ARTIFACTS,
    N_LAYERS,
    REPO_ID,
    ensure_dirs,
    group_kfold_indices,
    load_split,
    make_split,
    mask_ids,
    start_group,
    stratified_kfold_indices,
    theta_to_sincos,
    train_probe,
    train_probe_appendix_b_best,
    score_probe,
)
from metrics import theta_to_sincos as _  # noqa: F401


def load_acts(dataset: str):
    path = ARTIFACTS / "activations" / f"{dataset}.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "ids": data["ids"].astype(np.int64),
        "layers": data["layers"].astype(np.float32),
        "targets": data["targets"].astype(np.float32),
        "motion": data["motion"].astype(str),
        "start_xy": data["start_xy"].astype(np.float32),
        "pooling": str(data["pooling"]),
    }


def run_series(
    name: str,
    kind: str,
    X_all: np.ndarray,
    targets: np.ndarray,
    ids: np.ndarray,
    split: dict,
    groups: np.ndarray,
    bins: np.ndarray,
    device: torch.device,
) -> dict:
    """X_all: (N, 24, 1024) already filtered to the series subset."""
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])
    # indices within this filtered array
    idx_trainval = np.array(
        [i for i, cid in enumerate(ids) if int(cid) in train_ids or int(cid) in val_ids],
        dtype=np.int64,
    )
    assert not any(int(cid) in test_ids for cid in ids[idx_trainval])

    folds, fold_method = group_kfold_indices(groups[idx_trainval], 5)
    if not folds:
        folds, fold_method = stratified_kfold_indices(bins[idx_trainval], 5)

    cv_r2 = np.zeros((N_LAYERS, 5), dtype=np.float64)
    cv_err = np.zeros((N_LAYERS, 5), dtype=np.float64)
    err_key = "circular_mae" if kind == "direction" else "mae"

    for fold_i, (tr_rel, va_rel) in enumerate(tqdm(folds, desc=f"cv {name}", leave=False)):
        tr = idx_trainval[tr_rel]
        va = idx_trainval[va_rel]
        for ell in range(N_LAYERS):
            Xtr = X_all[tr, ell]
            Xva = X_all[va, ell]
            ytr = targets[tr]
            yva = targets[va]
            y_train = theta_to_sincos(ytr) if kind == "direction" else ytr
            # sweep inside fold
            best_r2 = -1e9
            best_err = 0.0
            for lr in [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]:
                for wd in [0.01, 0.1, 0.4, 0.8]:
                    probe = train_probe(Xtr, y_train, lr=lr, weight_decay=wd, epochs=100, device=device)
                    m = score_probe(probe, Xva, yva, kind=kind, device=device)
                    if m["r2"] > best_r2:
                        best_r2 = m["r2"]
                        best_err = m[err_key]
            cv_r2[ell, fold_i] = best_r2
            cv_err[ell, fold_i] = best_err

    # train-only refit / val for layer selection
    idx_train = np.array([i for i, cid in enumerate(ids) if int(cid) in train_ids], dtype=np.int64)
    idx_val = np.array([i for i, cid in enumerate(ids) if int(cid) in val_ids], dtype=np.int64)
    val_r2 = np.zeros(N_LAYERS)
    val_err = np.zeros(N_LAYERS)
    best_hps = []
    for ell in tqdm(range(N_LAYERS), desc=f"val {name}", leave=False):
        probe, hp, metrics = train_probe_appendix_b_best(
            X_all[idx_train, ell],
            targets[idx_train],
            X_all[idx_val, ell],
            targets[idx_val],
            kind=kind,
            device=device,
        )
        val_r2[ell] = metrics["r2"]
        val_err[ell] = metrics[err_key]
        best_hps.append(hp)

    return {
        "name": name,
        "kind": kind,
        "fold_method": fold_method,
        "cv_r2_mean": cv_r2.mean(axis=1).tolist(),
        "cv_r2_std": cv_r2.std(axis=1).tolist(),
        "cv_err_mean": cv_err.mean(axis=1).tolist(),
        "cv_err_std": cv_err.std(axis=1).tolist(),
        "val_r2": val_r2.tolist(),
        "val_err": val_err.tolist(),
        "best_hps": best_hps,
        "n_trainval": int(len(idx_trainval)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
    }


def choose_layer(dir_vel_val_r2, speed_val_r2, accel_val_r2) -> dict:
    r = np.asarray(dir_vel_val_r2, dtype=np.float64)
    r0 = float(r[0])
    L_peak = int(np.argmax(r))
    rmax = float(r[L_peak])
    thresh = (r0 + rmax) / 2.0
    L_emerg = int(np.argmax(r >= thresh))  # first True
    # np.argmax on bool gives first True; if all False, 0 — handle:
    if not np.any(r >= thresh):
        L_emerg = L_peak
    else:
        L_emerg = int(np.where(r >= thresh)[0][0])

    speed_ok = float(speed_val_r2[L_emerg]) >= 0.5
    accel_ok = float(accel_val_r2[L_emerg]) >= 0.5
    L_star = L_emerg if (speed_ok and accel_ok) else L_peak
    return {
        "r0": r0,
        "rmax": rmax,
        "L_peak": L_peak,
        "L_emerg": L_emerg,
        "L_star": int(L_star),
        "speed_at_emerg": float(speed_val_r2[L_emerg]),
        "accel_at_emerg": float(accel_val_r2[L_emerg]),
        "used_peak_fallback": not (speed_ok and accel_ok),
    }


def plot_layer_curves(series_list: list[dict], layer_choice: dict, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {
        "direction_all": "#1d4ed8",
        "direction_velocity": "#2563eb",
        "direction_acceleration": "#7c3aed",
        "speed": "#0f766e",
        "acceleration": "#9f1239",
    }
    for s in series_list:
        x = np.arange(N_LAYERS)
        mean = np.asarray(s["cv_r2_mean"])
        std = np.asarray(s["cv_r2_std"])
        c = colors.get(s["name"], "#333")
        axes[0].plot(x, mean, label=s["name"], color=c)
        axes[0].fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
        err = np.asarray(s["cv_err_mean"])
        err_std = np.asarray(s["cv_err_std"])
        axes[1].plot(x, err, label=s["name"], color=c)
        axes[1].fill_between(x, err - err_std, err + err_std, color=c, alpha=0.15)

    L = layer_choice["L_star"]
    for ax in axes:
        ax.axvline(L, color="#d89a1a", ls="--", label=f"L*={L}")
        ax.axvspan(7, 9, color="#d89a1a", alpha=0.12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("R² (5-fold CV on train+val)")
    axes[1].set_ylabel("Error (circ-MAE° / MAE)")
    axes[1].set_xlabel("Layer (block index 0–23)")
    axes[0].set_title(
        f"Layer-wise probes | pooling=mean_spacetime | {REPO_ID}\n"
        f"Test held out; L* from velocity-direction val R² (r0={layer_choice['r0']:.3f}, "
        f"L_emerg={layer_choice['L_emerg']}, L_peak={layer_choice['L_peak']})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ensure_dirs()

    # ensure splits exist
    for ds in ["direction", "speed", "acceleration"]:
        make_split(ds)

    series_out = []
    # Direction: three subsets
    dir_data = load_acts("direction")
    split_d = load_split("direction")
    groups_d = np.array([start_group(xy) for xy in dir_data["start_xy"]])
    bins_d = np.array([f"{t:.6f}|{m}" for t, m in zip(dir_data["targets"], dir_data["motion"])])

    for subset, mask_fn in [
        ("direction_all", lambda m: np.ones(len(m), dtype=bool)),
        ("direction_velocity", lambda m: m == "velocity"),
        ("direction_acceleration", lambda m: m == "acceleration"),
    ]:
        mask = mask_fn(dir_data["motion"])
        idx = np.where(mask)[0]
        series_out.append(
            run_series(
                subset,
                "direction",
                dir_data["layers"][idx],
                dir_data["targets"][idx],
                dir_data["ids"][idx],
                split_d,
                groups_d[idx],
                bins_d[idx],
                device,
            )
        )

    # Speed / acceleration
    for ds in ["speed", "acceleration"]:
        data = load_acts(ds)
        split = load_split(ds)
        groups = np.array([start_group(xy) for xy in data["start_xy"]])
        bins = np.array([f"{t:.8f}" for t in data["targets"]])
        series_out.append(
            run_series(
                ds,
                "scalar",
                data["layers"],
                data["targets"],
                data["ids"],
                split,
                groups,
                bins,
                device,
            )
        )

    dir_vel = next(s for s in series_out if s["name"] == "direction_velocity")
    speed = next(s for s in series_out if s["name"] == "speed")
    accel = next(s for s in series_out if s["name"] == "acceleration")
    layer_choice = choose_layer(dir_vel["val_r2"], speed["val_r2"], accel["val_r2"])

    # pooling ablation check
    r0 = layer_choice["r0"]
    need_ablation = r0 > 0.5 and layer_choice["L_emerg"] in (0, 1)
    layer_choice["need_temporal_last_ablation"] = bool(need_ablation)

    results = {
        "repo": REPO_ID,
        "pooling": dir_data["pooling"],
        "layer_choice": layer_choice,
        "series": series_out,
        "splits": {
            "direction": {k: len(v) if isinstance(v, list) else v for k, v in split_d.items()},
            "speed": {k: len(v) if isinstance(v, list) else v for k, v in load_split("speed").items()},
            "acceleration": {
                k: len(v) if isinstance(v, list) else v for k, v in load_split("acceleration").items()
            },
        },
    }
    out_json = ARTIFACTS / "results" / "layer_probes.json"
    out_json.write_text(json.dumps(results, indent=2))
    plot_layer_curves(series_out, layer_choice, ARTIFACTS / "figures" / "layer_curves.png")
    print(json.dumps(layer_choice, indent=2))
    print(f"wrote {out_json}")
    if need_ablation:
        print("WARNING: velocity direction readable at L0; temporal_last ablation recommended per plan.")


if __name__ == "__main__":
    main()
