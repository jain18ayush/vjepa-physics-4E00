#!/usr/bin/env python3
"""Unit tests from PART1_PLAN.md checks 3–5."""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

from metrics import circular_mae_degrees, sincos_to_degrees, theta_to_sincos
from common import LinearProbe, project_out
from steer import ProbeWeights, steer


def test_circular_mae():
    true = np.array([0.0, 10.0, 350.0])
    pred_perfect = true.copy()
    assert circular_mae_degrees(pred_perfect, true) < 1e-6
    pred_180 = (true + 180.0) % 360.0
    assert abs(circular_mae_degrees(pred_180, true) - 180.0) < 1e-6
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(np.linspace(0, 360, 1000, endpoint=False))
    true_u = np.linspace(0, 360, 1000, endpoint=False)
    mae = circular_mae_degrees(shuffled, true_u)
    assert 70 < mae < 110, mae
    print("circular MAE ok", mae)


def test_nullspace_projection():
    X = np.zeros((5, 4), dtype=np.float32)
    X[:, 0] = 1.0
    X[:, 1] = 2.0
    w = torch.zeros(1, 4)
    w[0, 0] = 1.0
    Xp = project_out(X, w)
    assert np.allclose(Xp[:, 0], 0, atol=1e-5)
    assert np.allclose(Xp[:, 1], 2.0, atol=1e-5)
    print("nullspace projection ok")


def test_steering_toy():
    # 4-D space; two orthogonal probes reading sin/cos-like targets from dims 0 and 1
    p1 = ProbeWeights(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), torch.zeros(2))
    p2 = ProbeWeights(torch.tensor([[0, 0, 1.0, 0], [0, 0, 0, 1.0]]), torch.zeros(2))
    x = torch.tensor([0.2, -0.3, 0.5, -0.7])
    x_star = steer(x, [p1, p2], math.radians(90))
    # both probes should predict (sin90, cos90)=(1,0)
    for p in [p1, p2]:
        pred = p(x_star.unsqueeze(0))[0]
        assert torch.allclose(pred, torch.tensor([1.0, 0.0]), atol=1e-4), pred
    # component orthogonal to both weight rows: originally none left in 4D if both probes span all dims
    # Use a component only in a direction outside the first probe: with both probes spanning R^4,
    # everything is in subspace. Instead check first probe alone leaves dims 2,3 unchanged.
    x2 = steer(x, [p1], math.radians(90))
    assert torch.allclose(x2[2:], x[2:], atol=1e-5)
    pred1 = p1(x2.unsqueeze(0))[0]
    assert torch.allclose(pred1, torch.tensor([1.0, 0.0]), atol=1e-4)
    print("steering toy ok")


if __name__ == "__main__":
    test_circular_mae()
    test_nullspace_projection()
    test_steering_toy()
    print("all unit tests passed")
