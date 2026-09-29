#!/usr/bin/env python3
"""Step 1: extract mean-pooled V-JEPA 2 layer activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor

from common import (
    ARTIFACTS,
    DATA_ROOT,
    N_LAYERS,
    REPO_ID,
    decode_video_rgb,
    ensure_dirs,
    load_manifest,
    process_video,
    target_for,
)


def one_clip_assert(model, processor, device, sample_path: str) -> dict:
    frames = decode_video_rgb(sample_path, 16)
    pixels = process_video(processor, frames, device)
    assert pixels.shape[0] == 1
    # Normalize layout to (B, T, C, H, W) for reporting
    if pixels.shape[1] == 16 and pixels.shape[2] == 3:
        t_frames = 16
    elif pixels.shape[1] == 3 and pixels.shape[2] == 16:
        t_frames = 16
    else:
        raise AssertionError(f"bad pixels shape {tuple(pixels.shape)}")

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(
                    pixel_values_videos=pixels,
                    output_hidden_states=True,
                    skip_predictor=True,
                )
        else:
            out = model(
                pixel_values_videos=pixels,
                output_hidden_states=True,
                skip_predictor=True,
            )

    hs = out.hidden_states
    assert len(hs) == 25, f"expected 25 hidden states, got {len(hs)}"
    shapes = []
    for ell in range(N_LAYERS):
        h = hs[1 + ell]
        assert h.shape[0] == 1 and h.shape[-1] == 1024, tuple(h.shape)
        assert h.shape[1] == 2048, f"layer {ell}: tokens={h.shape[1]}, want 2048"
        shapes.append(tuple(h.shape))
    info = {
        "sample_path": sample_path,
        "n_frames_in": 16,
        "pixels_shape": list(pixels.shape),
        "n_hidden_states": len(hs),
        "layer0_shape": list(shapes[0]),
        "t_frames_ok": t_frames == 16,
        "tokens_ok": True,
    }
    return info


@torch.inference_mode()
def extract_batch(model, processor, paths, device, use_amp: bool):
    videos = []
    for p in paths:
        frames = decode_video_rgb(p, 16)
        videos.append(list(frames))
    batch = processor(videos, return_tensors="pt")
    pixels = batch["pixel_values_videos"].to(device)
    # enforce T=16
    if pixels.shape[1] == 3:
        assert pixels.shape[2] == 16, pixels.shape
    else:
        assert pixels.shape[1] == 16, pixels.shape

    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(
                pixel_values_videos=pixels,
                output_hidden_states=True,
                skip_predictor=True,
            )
    else:
        out = model(
            pixel_values_videos=pixels,
            output_hidden_states=True,
            skip_predictor=True,
        )

    hs = out.hidden_states
    B = pixels.shape[0]
    pooled = np.zeros((B, N_LAYERS, 1024), dtype=np.float32)
    for ell in range(N_LAYERS):
        h = hs[1 + ell].float()  # (B, tokens, 1024)
        pooled[:, ell, :] = h.mean(dim=1).cpu().numpy()
    return pooled


def extract_dataset(dataset: str, batch_size: int, device: torch.device, resume: bool):
    ensure_dirs()
    rows = load_manifest(dataset)
    out_path = ARTIFACTS / "activations" / f"{dataset}.npz"
    ckpt_path = ARTIFACTS / "activations" / f"{dataset}.partial.npz"

    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    targets = np.array([target_for(dataset, r) for r in rows], dtype=np.float32)
    motion = np.array([r["motion"] for r in rows])
    start_xy = np.array([r["start_xy"] for r in rows], dtype=np.float32)
    paths = [r["video"] for r in rows]
    N = len(rows)
    layers = np.zeros((N, N_LAYERS, 1024), dtype=np.float32)
    done = np.zeros(N, dtype=bool)

    if resume and ckpt_path.exists():
        part = np.load(ckpt_path, allow_pickle=True)
        layers = part["layers"]
        done = part["done"].astype(bool)
        print(f"resuming {dataset}: {done.sum()}/{N} done")

    print(f"loading model {REPO_ID} ...")
    processor = AutoVideoProcessor.from_pretrained(REPO_ID)
    model = AutoModel.from_pretrained(REPO_ID, attn_implementation="sdpa").to(device).eval()

    assert_info = one_clip_assert(model, processor, device, paths[0])
    (ARTIFACTS / "results").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "results" / f"assert_{dataset}.json").write_text(json.dumps(assert_info, indent=2))
    print("one-clip assert ok:", assert_info)

    use_amp = device.type == "cuda"
    pending = [i for i in range(N) if not done[i]]
    for start in tqdm(range(0, len(pending), batch_size), desc=f"extract {dataset}"):
        idx = pending[start : start + batch_size]
        batch_paths = [paths[i] for i in idx]
        try:
            pooled = extract_batch(model, processor, batch_paths, device, use_amp)
        except Exception as e:
            # fall back to single-clip for this batch
            print(f"batch failed ({e}); falling back to single clips")
            pooled = np.zeros((len(idx), N_LAYERS, 1024), dtype=np.float32)
            for j, p in enumerate(batch_paths):
                pooled[j] = extract_batch(model, processor, [p], device, use_amp)[0]
        for j, i in enumerate(idx):
            layers[i] = pooled[j]
            done[i] = True
        if (start // batch_size) % 20 == 0:
            np.savez_compressed(ckpt_path, layers=layers, done=done, ids=ids)

    assert done.all(), f"incomplete extraction for {dataset}"
    np.savez_compressed(
        out_path,
        ids=ids,
        layers=layers,
        targets=targets,
        motion=motion,
        start_xy=start_xy,
        pooling=np.array("mean_spacetime"),
        repo=np.array(REPO_ID),
    )
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"wrote {out_path} shape={layers.shape}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["direction", "speed", "acceleration", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    datasets = ["direction", "speed", "acceleration"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        extract_dataset(ds, args.batch_size, device, args.resume)


if __name__ == "__main__":
    main()
