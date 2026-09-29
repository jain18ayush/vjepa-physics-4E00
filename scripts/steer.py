#!/usr/bin/env python3
"""Step 4: multi-probe subspace steering for direction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import (
    ARTIFACTS,
    LinearProbe,
    ensure_dirs,
    load_split,
    score_probe,
    theta_to_sincos,
    train_probe_c11,
)
from metrics import circular_mae_degrees, sincos_to_degrees


def load_acts(dataset: str):
    data = np.load(ARTIFACTS / "activations" / f"{dataset}.npz", allow_pickle=True)
    return {
        "ids": data["ids"].astype(np.int64),
        "layers": data["layers"].astype(np.float32),
        "targets": data["targets"].astype(np.float32),
        "motion": data["motion"].astype(str),
    }


def load_L_star() -> int:
    return int(json.loads((ARTIFACTS / "results" / "layer_probes.json").read_text())["layer_choice"]["L_star"])


class ProbeWeights(nn.Module):
    """Thin wrapper matching LinearProbe interface for saved weights."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        return self.linear(x)


def steer(x: torch.Tensor, probes_n: list[ProbeWeights], theta_star_rad: float) -> torch.Tensor:
    """Steer a single vector x (d,) toward theta* using probes' weight subspace."""
    M = torch.cat([p.weight.T for p in probes_n], dim=1)  # (d, 2n)
    V, _ = torch.linalg.qr(M, mode="reduced")
    x_perp = x - V @ (V.T @ x)
    t = torch.tensor(
        [math.sin(theta_star_rad), math.cos(theta_star_rad)],
        device=x.device,
        dtype=x.dtype,
    )
    rows, rhs = [], []
    for p in probes_n:
        rows.append(p.weight @ V)
        rhs.append(t - p.bias - p.weight @ x_perp)
    A = torch.cat(rows, dim=0)
    b = torch.cat(rhs, dim=0)
    c_star = torch.linalg.lstsq(A, b).solution
    return V @ c_star + x_perp


@torch.no_grad()
def sanity_steering_probes(x_star: torch.Tensor, probes_n: list[ProbeWeights], tol: float = 0.05) -> dict:
    errs = []
    target = torch.tensor([1.0, 0.0], device=x_star.device)  # sin90=1, cos90=0
    for p in probes_n:
        pred = p(x_star.unsqueeze(0))[0]
        errs.append(float(torch.mean((pred - target) ** 2)))
    return {"mean_mse_to_target_sincos": float(np.mean(errs)), "ok": float(np.mean(errs)) < tol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--L-star", type=int, default=None)
    ap.add_argument("--theta-star", type=float, default=90.0)
    args = ap.parse_args()
    device = torch.device(args.device)
    ensure_dirs()
    L_star = args.L_star if args.L_star is not None else load_L_star()
    theta_star = args.theta_star
    theta_star_rad = math.radians(theta_star)

    d = load_acts("direction")
    split = load_split("direction")
    test_ids = set(split["test"])
    assert not any(int(i) in set(split["train"]) for i in split["test"])

    test_idx = np.array([i for i, cid in enumerate(d["ids"]) if int(cid) in test_ids], dtype=np.int64)
    X_test = d["layers"][test_idx, L_star]
    y_test = d["targets"][test_idx]
    motion_test = d["motion"][test_idx]

    # evaluation probe on unsteered test only
    y_fit = theta_to_sincos(y_test)
    eval_probe = train_probe_c11(X_test, y_fit, kind="direction", device=device)
    eval_metrics = score_probe(eval_probe, X_test, y_test, kind="direction", device=device)
    print("eval probe on unsteered test:", eval_metrics)

    # load velocity nullspace probes
    pt = torch.load(
        ARTIFACTS / "probes" / f"direction_velocity_L{L_star}_nullspace.pt",
        map_location="cpu",
        weights_only=False,
    )
    probes = [ProbeWeights(p["weight"], p["bias"]).to(device) for p in pt["probes"]]
    print(f"loaded {len(probes)} steering probes")

    n_values = [1, 2, 5, 10, 15, 20]
    if len(probes) not in n_values:
        n_values.append(len(probes))
    n_values = sorted({n for n in n_values if 1 <= n <= len(probes)})

    # baseline n=0
    results = []
    subsets = {
        "all": np.ones(len(test_idx), dtype=bool),
        "velocity": motion_test == "velocity",
        "acceleration": motion_test == "acceleration",
    }

    def eval_table(X_s: np.ndarray):
        pred = eval_probe(torch.from_numpy(X_s).to(device)).detach().cpu().numpy()
        pred_deg = sincos_to_degrees(pred)
        out = {}
        for name, mask in subsets.items():
            if mask.sum() == 0:
                continue
            out[name] = {
                "mae_to_target": circular_mae_degrees(pred_deg[mask], np.full(mask.sum(), theta_star)),
                "mae_to_truth": circular_mae_degrees(pred_deg[mask], y_test[mask]),
                "n": int(mask.sum()),
            }
        return out

    baseline = eval_table(X_test)
    results.append({"n": 0, "subsets": baseline})

    sanity = None
    for n in n_values:
        probes_n = probes[:n]
        steered = np.zeros_like(X_test)
        for i in range(len(X_test)):
            x = torch.from_numpy(X_test[i]).to(device)
            steered[i] = steer(x, probes_n, theta_star_rad).detach().cpu().numpy()
        if n == n_values[-1]:
            x0 = torch.from_numpy(X_test[0]).to(device)
            x0s = torch.from_numpy(steered[0]).to(device)
            sanity = sanity_steering_probes(x0s, probes_n)
            print("sanity at full n:", sanity)
        results.append({"n": n, "subsets": eval_table(steered)})
        print(f"n={n}", results[-1]["subsets"]["all"])

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for subset, style in [("all", "-"), ("velocity", "--"), ("acceleration", ":")]:
        ns = [r["n"] for r in results if subset in r["subsets"]]
        to_t = [r["subsets"][subset]["mae_to_target"] for r in results if subset in r["subsets"]]
        to_true = [r["subsets"][subset]["mae_to_truth"] for r in results if subset in r["subsets"]]
        ax.plot(ns, to_t, style, marker="o", label=f"{subset}: MAE→{theta_star:g}°")
        ax.plot(ns, to_true, style, marker="x", label=f"{subset}: MAE→truth")
    ax.set_xlabel("number of orthogonal probes in steering basis")
    ax.set_ylabel("circular MAE (degrees) on held-out test")
    ax.set_title(
        f"Multi-probe steering at L*={L_star} → {theta_star:g}°\n"
        f"Steering probes from train (velocity); eval probe fit on unsteered test only"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "figures" / "steering.png", dpi=160)
    plt.close(fig)

    out = {
        "L_star": L_star,
        "theta_star_degrees": theta_star,
        "eval_probe_unsteered": eval_metrics,
        "n_steering_probes_available": len(probes),
        "sanity_full_n": sanity,
        "table": results,
    }
    (ARTIFACTS / "results" / "steering.json").write_text(json.dumps(out, indent=2))
    print("wrote steering results")


if __name__ == "__main__":
    main()
