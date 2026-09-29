"""Shared metrics for Part 1 probes."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def circular_mae_degrees(pred_deg: np.ndarray, true_deg: np.ndarray) -> float:
    """Mean absolute circular error in degrees. Wraps through 0/360."""
    pred = np.asarray(pred_deg, dtype=np.float64)
    true = np.asarray(true_deg, dtype=np.float64)
    err = (pred - true + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(err)))


def sincos_to_degrees(sincos: np.ndarray) -> np.ndarray:
    """Convert (N, 2) [sin, cos] predictions to degrees in [0, 360)."""
    s = np.asarray(sincos, dtype=np.float64)
    deg = np.degrees(np.arctan2(s[:, 0], s[:, 1])) % 360.0
    return deg


def theta_to_sincos(theta_degrees: np.ndarray) -> np.ndarray:
    """Convert degrees to (N, 2) [sin, cos] in radians."""
    rad = np.radians(np.asarray(theta_degrees, dtype=np.float64))
    return np.stack([np.sin(rad), np.cos(rad)], axis=1).astype(np.float32)


def r2_multi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
        yp = yp.reshape(-1, 1)
    return float(r2_score(yt, yp, multioutput="uniform_average"))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def random_baseline_mae(y_train: np.ndarray, y_eval: np.ndarray) -> float:
    mean = float(np.mean(y_train))
    return mae(y_eval, np.full_like(np.asarray(y_eval, dtype=np.float64), mean))


def direction_scores(y_true_sincos: np.ndarray, y_pred_sincos: np.ndarray, true_deg: np.ndarray):
    r2 = r2_multi(y_true_sincos, y_pred_sincos)
    pred_deg = sincos_to_degrees(y_pred_sincos)
    cmae = circular_mae_degrees(pred_deg, true_deg)
    return {"r2": r2, "circular_mae": cmae}


def scalar_scores(y_true: np.ndarray, y_pred: np.ndarray):
    return {"r2": r2_multi(y_true, y_pred), "mae": mae(y_true, y_pred)}
