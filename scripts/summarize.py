#!/usr/bin/env python3
"""Step 5: assemble summary.json from intermediate results."""

from __future__ import annotations

import json
from pathlib import Path

from common import ARTIFACTS, REPO_ID, ensure_dirs, load_split


def main():
    ensure_dirs()
    layer = json.loads((ARTIFACTS / "results" / "layer_probes.json").read_text())
    null = json.loads((ARTIFACTS / "results" / "nullspace.json").read_text())
    steer = json.loads((ARTIFACTS / "results" / "steering.json").read_text())
    asserts = {}
    for ds in ["direction", "speed", "acceleration"]:
        p = ARTIFACTS / "results" / f"assert_{ds}.json"
        if p.exists():
            asserts[ds] = json.loads(p.read_text())

    splits = {}
    for ds in ["direction", "speed", "acceleration"]:
        sp = load_split(ds)
        splits[ds] = {k: (len(v) if isinstance(v, list) else v) for k, v in sp.items()}

    series_val = {s["name"]: s["val_r2"] for s in layer["series"]}
    summary = {
        "repo": REPO_ID,
        "pooling": layer.get("pooling", "mean_spacetime"),
        "seed": 0,
        "splits": splits,
        "layer_choice": layer["layer_choice"],
        "per_layer_val_r2": series_val,
        "nullspace": {
            "L_star": null["L_star"],
            "quoted_dimensionality_direction_velocity": null["quoted_dimensionality"],
            "direction_velocity_sawtooth": null["direction_velocity_sawtooth"],
            "runs": [
                {
                    "name": r["name"],
                    "n_kept": r["n_kept"],
                    "dimensionality": r["dimensionality"],
                    "hit_iteration_cap": r["hit_iteration_cap"],
                    "final_history": r["history"][-1] if r["history"] else None,
                    "sawtooth_visible": r.get("sawtooth_visible"),
                }
                for r in null["runs"]
            ],
        },
        "steering": {
            "L_star": steer["L_star"],
            "theta_star_degrees": steer["theta_star_degrees"],
            "eval_probe_unsteered": steer["eval_probe_unsteered"],
            "sanity_full_n": steer["sanity_full_n"],
            "table": steer["table"],
        },
        "one_clip_asserts": asserts,
        "fallbacks": {
            "need_temporal_last_ablation": layer["layer_choice"].get("need_temporal_last_ablation", False),
            "used_peak_fallback": layer["layer_choice"].get("used_peak_fallback", False),
            "fold_methods": {s["name"]: s["fold_method"] for s in layer["series"]},
        },
        "figures": [
            "artifacts/figures/layer_curves.png",
            "artifacts/figures/nullspace.png",
            "artifacts/figures/steering.png",
        ],
    }
    out = ARTIFACTS / "results" / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
