#!/usr/bin/env python3
"""Step 3: iterative nullspace probing at L*."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    ARTIFACTS,
    ensure_dirs,
    load_split,
    project_out,
    score_probe,
    theta_to_sincos,
    train_probe_c11,
)
from metrics import random_baseline_mae


def load_acts(dataset: str):
    data = np.load(ARTIFACTS / "activations" / f"{dataset}.npz", allow_pickle=True)
    return {
        "ids": data["ids"].astype(np.int64),
        "layers": data["layers"].astype(np.float32),
        "targets": data["targets"].astype(np.float32),
        "motion": data["motion"].astype(str),
    }


def load_L_star() -> int:
    path = ARTIFACTS / "results" / "layer_probes.json"
    return int(json.loads(path.read_text())["layer_choice"]["L_star"])


def should_stop(kind: str, metrics: dict, baseline_mae: float | None) -> bool:
    if kind == "direction":
        return metrics["r2"] < 0.1 or metrics["circular_mae"] > 80.0
    return metrics["r2"] < 0.05 or metrics["mae"] > 0.9 * baseline_mae


def run_nullspace(
    name: str,
    kind: str,
    X: np.ndarray,
    targets: np.ndarray,
    ids: np.ndarray,
    split: dict,
    device: torch.device,
    max_iters: int,
) -> dict:
    train_set = set(split["train"])
    val_set = set(split["val"])
    test_set = set(split["test"])
    tr = np.array([i for i, cid in enumerate(ids) if int(cid) in train_set], dtype=np.int64)
    va = np.array([i for i, cid in enumerate(ids) if int(cid) in val_set], dtype=np.int64)
    assert not any(int(cid) in test_set for cid in ids[tr])
    assert not any(int(cid) in test_set for cid in ids[va])

    X_tr = X[tr].copy()
    X_va = X[va].copy()
    y_tr = targets[tr]
    y_va = targets[va]
    y_train_fit = theta_to_sincos(y_tr) if kind == "direction" else y_tr

    baseline = None
    if kind != "direction":
        baseline = random_baseline_mae(y_tr, y_va)

    kept = []
    history = []
    hit_cap = True
    for k in range(max_iters):
        probe = train_probe_c11(X_tr, y_train_fit if kind == "direction" else y_tr, kind=kind, device=device)
        metrics = score_probe(probe, X_va, y_va, kind=kind, device=device)
        history.append({"k": k, **metrics})
        print(f"  [{name}] k={k} {metrics}")
        if should_stop(kind, metrics, baseline):
            hit_cap = False
            break
        # keep probe, then project
        kept.append(
            {
                "weight": probe.weight.detach().cpu().clone(),
                "bias": probe.bias.detach().cpu().clone(),
                "metrics": metrics,
            }
        )
        X_tr = project_out(X_tr, probe.weight)
        X_va = project_out(X_va, probe.weight)
        # y_train_fit for direction stays same labels; features changed
    else:
        # completed all iterations without stop — last probe at cap was scored but
        # if we didn't break, the last iteration's probe was kept only if not stop.
        # Loop for-else: hit_cap True means never stopped.
        pass

    dim = (2 * len(kept)) if kind == "direction" else len(kept)
    return {
        "name": name,
        "kind": kind,
        "n_kept": len(kept),
        "dimensionality": dim,
        "history": history,
        "hit_iteration_cap": hit_cap and len(history) >= max_iters,
        "baseline_mae": baseline,
        "probes": kept,
    }


def sawtooth_visible(history: list[dict]) -> bool:
    """Heuristic: look for paired rise after drop in direction R²."""
    if len(history) < 4:
        return False
    r2 = [h["r2"] for h in history[: min(20, len(history))]]
    rises_after_drop = 0
    for i in range(1, len(r2) - 1):
        if r2[i] < r2[i - 1] - 0.02 and r2[i + 1] > r2[i] + 0.01:
            rises_after_drop += 1
    return rises_after_drop >= 2


def plot_nullspace(runs: list[dict], L_star: int, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        xs = [h["k"] for h in run["history"]]
        r2 = [h["r2"] for h in run["history"]]
        if run["kind"] == "direction":
            err = [h["circular_mae"] for h in run["history"]]
            axes[1].set_ylabel("circular MAE (deg)")
        else:
            err = [h["mae"] for h in run["history"]]
            axes[1].set_ylabel("MAE")
        axes[0].plot(xs, r2, marker="o", ms=3, label=run["name"])
        axes[1].plot(xs, err, marker="o", ms=3, label=run["name"])
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("probe iteration k")
    axes[0].set_ylabel("val R²")
    axes[0].set_title(f"Nullspace R² at L*={L_star} (train fit / val score; test unused)")
    axes[1].set_title("Nullspace error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--L-star", type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device)
    ensure_dirs()
    L_star = args.L_star if args.L_star is not None else load_L_star()
    print(f"L* = {L_star}")

    runs_meta = []
    saved_probes = {}

    # Direction subsets
    d = load_acts("direction")
    split_d = load_split("direction")
    for name, motion_filter in [
        ("direction_velocity", "velocity"),
        ("direction_acceleration", "acceleration"),
        ("direction_all", None),
    ]:
        if motion_filter is None:
            idx = np.arange(len(d["ids"]))
        else:
            idx = np.where(d["motion"] == motion_filter)[0]
        print(f"nullspace {name} n={len(idx)}")
        run = run_nullspace(
            name,
            "direction",
            d["layers"][idx, L_star],
            d["targets"][idx],
            d["ids"][idx],
            split_d,
            device,
            max_iters=80,
        )
        run["sawtooth_visible"] = sawtooth_visible(run["history"])
        saved_probes[name] = run.pop("probes")
        runs_meta.append(run)

    for ds in ["speed", "acceleration"]:
        data = load_acts(ds)
        split = load_split(ds)
        print(f"nullspace {ds}")
        run = run_nullspace(
            ds,
            "scalar",
            data["layers"][:, L_star],
            data["targets"],
            data["ids"],
            split,
            device,
            max_iters=40,
        )
        run["sawtooth_visible"] = False
        saved_probes[ds] = run.pop("probes")
        runs_meta.append(run)

    # persist velocity probes for steering
    vel_probes = saved_probes["direction_velocity"]
    torch.save(
        {
            "L_star": L_star,
            "probes": [{"weight": p["weight"], "bias": p["bias"], "metrics": p["metrics"]} for p in vel_probes],
            "n_kept": len(vel_probes),
        },
        ARTIFACTS / "probes" / f"direction_velocity_L{L_star}_nullspace.pt",
    )
    # also save all
    torch.save(
        {k: [{"weight": p["weight"], "bias": p["bias"], "metrics": p["metrics"]} for p in v] for k, v in saved_probes.items()},
        ARTIFACTS / "probes" / f"all_L{L_star}_nullspace.pt",
    )

    out = {
        "L_star": L_star,
        "runs": runs_meta,
        "quoted_dimensionality": next(r["dimensionality"] for r in runs_meta if r["name"] == "direction_velocity"),
        "direction_velocity_sawtooth": next(r["sawtooth_visible"] for r in runs_meta if r["name"] == "direction_velocity"),
    }
    (ARTIFACTS / "results" / "nullspace.json").write_text(json.dumps(out, indent=2))
    plot_nullspace(runs_meta, L_star, ARTIFACTS / "figures" / "nullspace.png")
    print(json.dumps({k: out[k] for k in out if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
