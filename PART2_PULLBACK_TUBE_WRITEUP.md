# Pullback + tube: direction is a spine in a tube

Two follow-ups to Part 2’s periodic direction spline. Plans:
[`PART2_TUBE_PLAN.md`](PART2_TUBE_PLAN.md). Scripts:
`scripts/part2_pullback.py`, `scripts/part2_tube.py`.

Shared setup: `L* = 5`, train/velocity spline, frozen eval probe on
unsteered test (seed 0), held-out leftovers kept off the 64-D PCA block.

---

## 1. Pullback — optimizing for the probe leaves the spline

Paper move (§3.3 / mountain-car C.3): initialize on the chord, L-BFGS a path
in the first 32 PCs so the probe tracks the claimed angle ramp. Ask whether
the optimized path falls onto `Mh`.

### `0° → 180°`

| Path | Mean dist → Mh (PCA) | Mid-path MAE → claimed | Final MAE → claimed |
|---|---:|---:|---:|
| Geometric (`Mh`) | 0.04 | 12.5° | 8.8° |
| Chord init | 1.60 | 174.6° | 8.8° |
| Pullback | **3.31** | **1.05°** | **1.17°** |

Recovery fraction vs chord→geometric gap: **−1.09** (pullback is *farther*
from `Mh` than the chord).

![Pullback 0→180](artifacts/figures/part2_pullback_dir_0_to_180.png)

### Read

Opposite of the mountain-car story on this encoder. Matching the probe
trajectory **improves** mid-path and endpoint readout, but the optimizer
**leaves** the activation spline. So the circle is a useful spine for
ordered walks, not the unique probe-optimal path. That is what you expect
if direction lives in a **tube**: many off-spine offsets still decode as
the right angle.

Short-arc pullback is milder (dist 1.02 vs chord 0.81) but still not a
recovery onto `Mh`. Speed pullback (earlier run) also left `Mh` while
nearly zeroing scalar error — flat variables do not need the curve either.

---

## 2. Tube — two normal directions close the endpoint gap

### Geometry of the residual

For each train/velocity point: nearest spine location, residual, strip the
local tangent, PCA the perpendicular residuals.

| Quantity | Value |
|---|---:|
| Median ‖z − s(u*)‖ | 4.62 |
| Mean tangential energy | 0.03 |
| Mean perpendicular energy | **26.45** |
| Cum. tube variance @ 5 PCs | 64% |
| Cum. tube variance @ 20 PCs | **91%** |

Almost all scatter around the circle is **normal** to the spine. The tube
is real and moderately low-rank.

![Tube variance](artifacts/figures/part2_tube_variance.png)

### Steering: spine + tube vs multi-probe (`0° → 180°`)

At waypoint `t`, set `z(t) = s(u(t)) + t · V_k c*`, with `c*` chosen so the
frozen probe reads `(sin θ*, cos θ*)` at the endpoint (same lstsq idea as
Part 1, but basis = tube, pinned to the spine).

| Method | Mid MAE → claimed | Final MAE → target |
|---|---:|---:|
| Spline only (`k=0`) | 12.5° | 19.8° |
| Chord | 174.6°† | 19.8°† |
| Spine + resid-PCA tube `k=1` | 11.7° | 19.7° |
| Spine + resid-PCA tube `k=2` | **4.2°** | **2.8°** |
| Spine + nullspace tube `k=1` | 4.9°‡ | **2.6°** |
| Spine + nullspace / resid `k≥2` | ~4.2° | **2.8°** |
| Multi-probe `n=5` | — | 8.0° |
| Multi-probe `n=20` | — | 3.5° |

†Chord mid from the pullback run’s probe; tube run’s chord mid is ~96°+ —
same qualitative teleport.  
‡From the `k=1` nullspace row in the tube log / JSON.

![Tube curves 0→180](artifacts/figures/part2_tube_dir_0_to_180.png)

### Short arc `22.5° → 337.5°`

| Method | Final MAE → target |
|---|---:|
| Spline only | 7.1° |
| Spine + tube `k=2` | **2.4°** |
| Multi-probe `n=20` | 3.9° |

Same pattern: two tube dimensions beat a large multi-probe edit, without
giving up an on-circle mid-path (mid stays ~7–13°, not a chord teleport).

---

## Joint verdict

| Claim | Evidence |
|---|---|
| Direction has a circular **spine** | Part 2: spline mid-path ~5–12° vs chord ~100–170° |
| The spine is not the whole story | Pullback tracks the probe better **off** `Mh` (recovery &lt; 0) |
| Extra Part 1 dims are a **tube** | Perp energy ≫ tang; 20 tube PCs → 91% residual variance |
| Tube coordinates are the missing control | `k=2` spine+tube ≤ multi-probe `n=20` on final MAE, mid-path stays ordered |

Practical recipe on this data: **walk the periodic spline for intermediate
angles; add ~2 normal coordinates (residual PCA or the first nullspace
directions, orthogonalized to the tangent) to nail the endpoint.** That
unifies Part 1’s multi-probe result with Part 2’s manifold walk.

---

## Artifacts

```text
PART2_TUBE_PLAN.md
scripts/part2_pullback.py
scripts/part2_tube.py
artifacts/results/part2_pullback.json
artifacts/results/part2_tube.json
artifacts/figures/part2_pullback_*.png
artifacts/figures/part2_tube_*.png
```

```bash
cd /workspace/vjepa-physics/scripts
python part2_pullback.py --device cuda --kinds direction
python part2_tube.py --device cuda
```
