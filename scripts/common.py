"""Data loading, splits, and probe training helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, StratifiedKFold

from metrics import (  # noqa: F401 — re-exported for other scripts
    direction_scores,
    random_baseline_mae,
    scalar_scores,
    theta_to_sincos,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPO_ID = "facebook/vjepa2-vitl-fpc64-256"
N_LAYERS = 24
SEED = 0

LR_SWEEP = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
WD_SWEEP = [0.01, 0.1, 0.4, 0.8]


def start_group(xy) -> str:
    x, y = float(xy[0]), float(xy[1])
    return f"{round(x * 2) / 2:.1f}_{round(y * 2) / 2:.1f}"


def load_manifest(dataset: str) -> list[dict]:
    base = DATA_ROOT / dataset
    rows = []
    with open(base / "manifest.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            meta = json.loads((base / rec["metadata"]).read_text())
            video = base / rec["video"]
            rows.append(
                {
                    "id": int(rec["id"]),
                    "video": str(video),
                    "metadata": meta,
                    "motion": meta["motion"],
                    "theta_degrees": float(meta["theta_degrees"]),
                    "start_xy": [float(meta["start_position_xy_m"][0]), float(meta["start_position_xy_m"][1])],
                    "speed_mps": float(meta["speed_mps"]),
                    "acceleration_mps2": float(meta["acceleration_mps2"]),
                    "magnitude": float(meta["magnitude"]) if "magnitude" in meta else None,
                }
            )
    rows.sort(key=lambda r: r["id"])
    return rows


def target_for(dataset: str, row: dict) -> float:
    if dataset == "direction":
        return row["theta_degrees"]
    if dataset == "speed":
        return row["magnitude"] if row["magnitude"] is not None else row["speed_mps"]
    if dataset == "acceleration":
        return row["magnitude"] if row["magnitude"] is not None else row["acceleration_mps2"]
    raise ValueError(dataset)


def label_bin(dataset: str, row: dict) -> str:
    if dataset == "direction":
        return f"{row['theta_degrees']:.6f}|{row['motion']}"
    return f"{target_for(dataset, row):.8f}"


def make_split(dataset: str, rows: list[dict] | None = None) -> dict[str, list[int]]:
    rows = rows or load_manifest(dataset)
    by_bin: dict[str, list[dict]] = {}
    for r in rows:
        by_bin.setdefault(label_bin(dataset, r), []).append(r)

    rng = np.random.default_rng(SEED)
    train, val, test = [], [], []
    for bin_rows in by_bin.values():
        ids = [r["id"] for r in bin_rows]
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(0.20 * n))) if n >= 3 else (1 if n >= 2 else 0)
        n_val = max(1, int(round(0.10 * n))) if n >= 5 else (1 if n - n_test >= 2 else 0)
        # ensure train is non-empty when possible
        while n_test + n_val >= n and n_val > 0:
            n_val -= 1
        while n_test + n_val >= n and n_test > 1:
            n_test -= 1
        test.extend(ids[:n_test])
        val.extend(ids[n_test : n_test + n_val])
        train.extend(ids[n_test + n_val :])

    split = {
        "seed": SEED,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }
    out = ARTIFACTS / "splits" / f"{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2))
    return split


def load_split(dataset: str) -> dict[str, list[int]]:
    path = ARTIFACTS / "splits" / f"{dataset}.json"
    if path.exists():
        return json.loads(path.read_text())
    return make_split(dataset)


def decode_video_rgb(path: str, n_frames: int = 16) -> np.ndarray:
    """Return uint8 array (T, H, W, C) RGB with exactly n_frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < n_frames:
        raise RuntimeError(f"{path}: got {len(frames)} frames, need {n_frames}")
    if len(frames) > n_frames:
        # take first n_frames in temporal order (clips are exactly 16)
        frames = frames[:n_frames]
    arr = np.stack(frames, axis=0)
    if arr.shape[0] != n_frames:
        raise RuntimeError(f"{path}: unexpected shape {arr.shape}")
    return arr


def process_video(processor, frames_thwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """frames: (T, H, W, C) uint8 -> pixel_values_videos (1, T, C, H, W) on device."""
    # transformers AutoVideoProcessor expects list of frames or (T,C,H,W) / (T,H,W,C)
    video = list(frames_thwc)  # list of HWC
    batch = processor(video, return_tensors="pt")
    pixels = batch["pixel_values_videos"]
    if pixels.ndim != 5:
        raise RuntimeError(f"unexpected pixels shape {tuple(pixels.shape)}")
    t = pixels.shape[1] if pixels.shape[2] == 3 else pixels.shape[2]
    # HF VJEPA2 typically: (B, T, C, H, W) or (B, C, T, H, W)
    if pixels.shape[2] == 3:
        # (B, T, C, H, W)
        if pixels.shape[1] != 16:
            raise RuntimeError(f"processor changed T to {pixels.shape[1]}, need 16")
    elif pixels.shape[1] == 3:
        if pixels.shape[2] != 16:
            raise RuntimeError(f"processor changed T to {pixels.shape[2]}, need 16")
    else:
        raise RuntimeError(f"cannot interpret pixels shape {tuple(pixels.shape)}")
    return pixels.to(device)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    device: torch.device | None = None,
) -> LinearProbe:
    device = device or torch.device("cpu")
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    out_dim = y.shape[1]
    probe = LinearProbe(X.shape[1], out_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    yt = torch.from_numpy(y).to(device)
    probe.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = probe(xt)
        loss = torch.mean((pred - yt) ** 2)
        loss.backward()
        opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def predict_probe(probe: LinearProbe, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    device = device or next(probe.parameters()).device
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    out = probe(xt).detach().cpu().numpy()
    if out.shape[1] == 1:
        return out[:, 0]
    return out


def score_probe(
    probe: LinearProbe,
    X: np.ndarray,
    targets: np.ndarray,
    *,
    kind: str,
    device: torch.device | None = None,
) -> dict[str, float]:
    pred = predict_probe(probe, X, device=device)
    if kind == "direction":
        y = theta_to_sincos(targets)
        return direction_scores(y, pred, targets)
    return scalar_scores(targets, pred)


def train_probe_c11(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kind: str,
    device: torch.device | None = None,
) -> LinearProbe:
    epochs = 100 if kind == "direction" else 50
    return train_probe(X, y, lr=1e-3, weight_decay=1e-4, epochs=epochs, device=device)


def train_probe_appendix_b_best(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    targets_va: np.ndarray,
    *,
    kind: str,
    device: torch.device | None = None,
) -> tuple[LinearProbe, dict[str, float], dict[str, float]]:
    """Sweep LR/WD; return best probe on val R² plus its hyperparams and scores."""
    best_probe = None
    best_score = -1e9
    best_hp: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    y_train = theta_to_sincos(y_tr) if kind == "direction" else y_tr
    for lr in LR_SWEEP:
        for wd in WD_SWEEP:
            probe = train_probe(X_tr, y_train, lr=lr, weight_decay=wd, epochs=100, device=device)
            metrics = score_probe(probe, X_va, targets_va, kind=kind, device=device)
            if metrics["r2"] > best_score:
                best_score = metrics["r2"]
                best_probe = probe
                best_hp = {"lr": lr, "weight_decay": wd}
                best_metrics = metrics
    assert best_probe is not None
    return best_probe, best_hp, best_metrics


def group_kfold_indices(groups: np.ndarray, n_splits: int = 5) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    unique = np.unique(groups)
    if len(unique) < n_splits:
        return [], "too_few_groups"
    gkf = GroupKFold(n_splits=n_splits)
    # dummy X/y
    X = np.zeros((len(groups), 1))
    y = np.zeros(len(groups))
    try:
        folds = list(gkf.split(X, y, groups))
        return folds, "GroupKFold"
    except ValueError:
        return [], "GroupKFold_failed"


def stratified_kfold_indices(bins: np.ndarray, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    X = np.zeros((len(bins), 1))
    return list(skf.split(X, bins)), "StratifiedKFold"


def index_map(ids: np.ndarray) -> dict[int, int]:
    return {int(i): k for k, i in enumerate(ids)}


def mask_ids(all_ids: np.ndarray, keep: set[int] | list[int]) -> np.ndarray:
    keep_set = set(int(x) for x in keep)
    return np.array([i for i, cid in enumerate(all_ids) if int(cid) in keep_set], dtype=np.int64)


def project_out(X: np.ndarray, weight: torch.Tensor) -> np.ndarray:
    """X: (N, d), weight: (n_out, d). Project out row space of weight."""
    w = weight.detach().cpu().numpy().astype(np.float64).T  # (d, n_out)
    Q, R = np.linalg.qr(w, mode="reduced")
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.abs(R)
    keep = diag >= 1e-6
    if not np.any(keep):
        return X.copy()
    Q = Q[:, keep]
    X64 = X.astype(np.float64)
    return (X64 - X64 @ Q @ Q.T).astype(np.float32)


def ensure_dirs():
    for sub in ["splits", "activations", "probes", "figures", "results"]:
        (ARTIFACTS / sub).mkdir(parents=True, exist_ok=True)
