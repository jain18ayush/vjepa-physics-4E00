# Dataset reference

The take-home includes separate datasets for direction, speed, and
acceleration:

```text
data/<variable>/
├── manifest.jsonl
└── videos/
    ├── scene_0000/
    │   ├── video.mp4
    │   └── metadata.json
    └── ...
```

## Video format

Every clip is an MP4 containing a single blue disk moving on a dark background:

- resolution: 256 × 256 pixels;
- duration: 16 frames;
- frame rate: 24 frames per second.

## Manifests

Each `manifest.jsonl` is newline-delimited JSON. Every row contains an integer
`id` and paths to a video and metadata file:

```json
{"id": 0, "video": "videos/scene_0000/video.mp4", "metadata": "videos/scene_0000/metadata.json"}
```

Resolve `video` and `metadata` relative to the directory containing that
manifest. For example, paths in `data/speed/manifest.jsonl` are relative to
`data/speed/`.

## Labels

| Dataset | Primary target | Clips | Target values | Useful metrics |
|---|---|---:|---|---|
| `direction` | `theta_degrees` | 1,500 | 64 equally spaced directions in [0°, 360°) | circular MAE; R² on sine/cosine targets |
| `speed` | `magnitude` / `speed_mps` | 1,536 | 64 speeds from 0.25 to 4.0 m/s | MAE; R² |
| `acceleration` | `magnitude` / `acceleration_mps2` | 1,536 | 64 accelerations from 0.25 to 10.0 m/s² | MAE; R² |

Each `metadata.json` also records:

- `primary_label`;
- `theta_degrees`;
- `motion`;
- `speed_mps`;
- `acceleration_mps2`;
- `start_position_xy_m`;
- `fps`;
- `frames`.

Direction is circular. A practical probing target is
`(sin(theta), cos(theta))`; convert the predicted pair back to an angle before
computing circular error.

## Experimental use

Choose and document a train/validation/test protocol that provides a fair test
of generalization. Probe fitting, layer selection, nullspace construction, and
spline construction must not use the final held-out evaluation examples.

Keep all derived activations, trained probes, steering artifacts, and figures
outside `data/`. The supplied videos and metadata should remain unchanged.
