# Part 2 results: activation splines vs multi-probe steering

Minimum Goodfire-style manifold steering on the Part 1 V-JEPA 2
activations (`L* = 5`), compared with the Part 1 multi-probe edit.
Method lock: [`PART2_PLAN.md`](PART2_PLAN.md). Interactive method sketch:
[`manifold_explainer.html`](manifold_explainer.html).

Frozen encoder activations reused from Part 1
(`artifacts/activations/*.npz`). No new forward passes. Splines and
PCA fit on **train only** (direction: train / velocity). Evaluation probe
fit on **unsteered test only**, then frozen.

---

## Setup

| | Speed | Acceleration | Direction |
|---|---|---|---|
| Spline | Natural cubic | Natural cubic | Periodic cubic |
| Fit set | train (1,088) | train (1,088) | train / velocity (558) |
| Centroids | 64 magnitudes | 64 magnitudes | 64 angles |
| PCA @ L5 | 64 comps, 98.1% var | 64 comps, 98.0% var | 64 comps, 98.5% var |
| Test residual to curve (median) | 4.83 | 4.82 | 4.82 |
| Eval probe (unsteered test) | R² 0.80, MAE 0.41 | R² 0.33, MAE 1.99 | R² 0.80, circ. MAE 14.5° |

Paths use `K = 21` waypoints. At each step the top-64 PCA block is replaced
by a point on the path; the test clip’s off-subspace leftover is kept
(App. A.6 of Wurgaft et al.).

Endpoint pairs:

| Path | From → to |
|---|---|
| `speed_0.5_to_3.5` | 0.49 → 3.52 m/s |
| `accel_1_to_8` | 1.02 → 7.99 m/s² |
| `dir_0_to_180` | 0° → 180° |
| `dir_20_to_340` | 22.5° → 337.5° (short arc across 0°) |

---

## Main comparison

### Direction: the 1D periodic spline is the right *path*, multi-probe wins the *endpoint*

Half-turn path (`0° → 180°`), held-out test:

| Method | Mid-path MAE → claimed (`t = 0.5`) | Final MAE → target (`t = 1`) | Final MAE → truth |
|---|---:|---:|---:|
| Periodic spline | **5.5°** | 18.6° | 90.0° |
| Chord (same PCA subspace) | 170.9° | 18.6° | 90.0° |
| Multi-probe `n = 1` | — | 70.3° | 29.5° |
| Multi-probe `n = 5` | — | 14.4° | 80.4° |
| Multi-probe `n = 20` | — | **8.5°** | 86.8° |

![Decode curves for 0→180](artifacts/figures/part2_decode_dir_0_to_180.png)

The spline tracks the claimed angle through the intermediate waypoints. The
chord cuts the interior of the circle and the probe readout teleports — peak
mid-path error ~171°, the Goodfire signature. Endpoints of spline and chord
coincide by construction (same centroids), so final error matches at 18.6°.

Multi-probe steering toward the *same* endpoint (180°) with 20 orthogonal
nullspace probes reaches **8.5°**, better than the spline’s final 18.6°.
Part 1’s quote toward 90° was similar: 6.3° at `n = 20`. So:

- A **1D periodic curve is enough to walk intermediate directions**.
- Hitting a fixed target as hard as Part 1 still wants the **redundant
  multi-probe subspace** (Part 1 dim ≈ 82). The curve is a spine through a
  tube; multi-probe moves many copies of the same feature together.

Short-arc path across 0° (`22.5° → 337.5°`):

| Method | Mid-path MAE → claimed | Final MAE → target |
|---|---:|---:|
| Periodic spline | 8.7° | **1.3°** |
| Chord | 2.1° | **1.3°** |
| Multi-probe `n = 20` | — | 8.4° |

Here the arc is short, so the chord is nearly on-manifold and both path
methods land on the target. Multi-probe is *worse* on this nearby target:
it overwrites a large subspace toward one angle and pays a constant ~8°
tax, while walking to a nearby centroid stays on the fitted curve.

![PCA overlay for direction](artifacts/figures/part2_pca_dir_0_to_180.png)

Accelerating-motion transfer on `0° → 180°` (spline fit only on velocity):
final MAE → target is 18.4°, essentially the same as the mixed test set.
The velocity-trained circle transfers.

### Speed: flat geometry — spline ≈ chord

| Method | Mid-path MAE → claimed (`t = 0.5`) | Final MAE → target |
|---|---:|---:|
| Natural spline | 0.37 | 0.67 |
| Chord | **0.10** | 0.67 |

![Speed decode curves](artifacts/figures/part2_decode_speed_0.5_to_3.5.png)

Both methods move the decode from ~0.95 toward ~2.86 (target 3.52). The
gap between them is small. The chord is even a bit closer to the claimed
ramp mid-path. That matches Part 1: speed lived in about **5** dimensions
and was linearly readable from layer 0. Curvature is not buying control
here; Euclidean interpolation in the PCA subspace is already the right
geometry.

Neither path reaches the exact endpoint label under the global probe
(final decode 2.86 vs 3.52). The train centroid for 3.52 does not sit on
the probe’s isosurface for that magnitude — the plan’s known failure mode.

### Acceleration: same story, weaker probe

| Method | Mid-path MAE → claimed (`t = 0.5`) | Final MAE → target |
|---|---:|---:|
| Natural spline | 0.58 | 2.57 |
| Chord | **0.04** | 2.57 |

Eval probe on unsteered test is weak (R² 0.33, MAE 1.99), so absolute
numbers are soft. Relative to Part 1’s nullspace result (dim ≈ 2), the
geometry is again nearly linear: spline and chord agree at the ends and
differ only mildly in the middle. Flat Euclidean steering is enough; the
1D spline is not an upgrade.

![Comparison bar](artifacts/figures/part2_comparison_bar.png)

---

## Verdict vs multi-probe

| Question | Answer on this data |
|---|---|
| Does direction live on a 1D circle you can walk? | **Yes.** Periodic spline mid-path MAE → claimed is 5.5° on a half turn; the chord is 171°. |
| Does that circle replace multi-probe for targeting? | **No.** Multi-probe `n = 20` still halves the final error on `0° → 180°` (8.5° vs 18.6°). |
| When is the spline *better* than multi-probe? | Nearby moves on the circle (`22.5° → 337.5°`: 1.3° vs 8.4°). |
| Speed / acceleration: use a spline? | **Optional.** Chord matches the spline; Part 1’s low dimensionality already said the geometry is flat. |
| Residual to the curve | ~4.8 for all three (thick tube in PCA units). Direction still has a usable spine; the thickness is why multi-probe helps at the endpoint. |

Bottom line for the presentation: **direction is curved and causal along
the curve; speed and acceleration are flat.** Multi-probe remains the
stronger hammer for forcing a single held-out angle. The periodic spline
is the better account of *how* the model moves between angles, and it is
enough when the target is a nearby centroid on that circle.

---

## Artifacts

```text
artifacts/results/part2_splines.json
artifacts/figures/part2_pca_{speed_0.5_to_3.5,accel_1_to_8,dir_0_to_180,dir_20_to_340}.png
artifacts/figures/part2_decode_{speed_0.5_to_3.5,accel_1_to_8,dir_0_to_180,dir_20_to_340}.png
artifacts/figures/part2_comparison_bar.png
scripts/part2_splines.py
```

Reproduce on the GPU box (activations already under `artifacts/`):

```bash
cd /workspace/vjepa-physics/scripts
python part2_splines.py --device cuda
```

Follow-ups (pullback + tube-not-curve): see
[`PART2_PULLBACK_TUBE_WRITEUP.md`](PART2_PULLBACK_TUBE_WRITEUP.md).

Why the direction tube exists (fiber / nuisance / copies / layers): see
[`PART2_TUBE_WHY_WRITEUP.md`](PART2_TUBE_WHY_WRITEUP.md).
