# Tube-not-curve experiment plan

Direction at `L* = 5` is high-dimensional (Part 1: ~82 dims, sawtooth) but
Part 2’s periodic spline still walks intermediate angles cleanly. Multi-probe
beats the spline at a fixed endpoint (~8.5° vs ~18.6° on `0°→180°`). Hypothesis:
the representation is a **tube** around a 1D circular spine — the spline is the
spine; the extra Part 1 dimensions are the tube’s cross-section.

## Question

Does adding a small **normal bundle** (tube coordinates orthogonal to the
spline tangent) close the endpoint gap to multi-probe, while keeping mid-path
decodes on the claimed angle ramp?

## Construction

Reuse Part 2’s direction spline (train / velocity, periodic cubic, 64-D PCA).

1. For each train (velocity) activation `x`, project to PCA, find nearest point
   `s(u*)` on the spline, store residual `r = z - s(u*)`.
2. Also estimate the unit tangent `T(u*) ∝ s'(u*)` and remove the tangential
   component: `r_⊥ = r - (r·T) T`.
3. PCA on `{r_⊥}` → orthonormal tube basis `V_k` (columns). Record variance
   explained by the first `k` tube components vs tangential energy.
4. Optional second basis: Part 1 nullspace probe rows, QR’d, then projected
   orthogonal to `T` at the target `u` — links tube coords to the multi-probe
   construction.

## Steering conditions (held-out test, frozen eval probe)

Target path: `0° → 180°` (and short-arc `22.5° → 337.5°` as a secondary).

| Condition | PCA coords at waypoint `t` |
|---|---|
| Spline only | `s(u(t))` |
| Chord | `(1-t) s(u0) + t s(u1)` |
| Spline + tube @ endpoint | mid-path: `s(u(t))`; at `t=1`: `s(u1) + V_k c*` |
| Spline + tube along path | `s(u(t)) + t · V_k c*` (ramp tube in) |
| Multi-probe `n∈{1,5,20}` | Part 1 edit toward path endpoint |

`c*` is the least-squares solve so the frozen probe reads
`(sin θ*, cos θ*)`, same algebra as Part 1 multi-probe but with basis `V_k`
and the point pinned to the spine (plus tube). Leftover off-PCA coords stay
per-clip as in Part 2.

Sweep `k ∈ {0,1,2,5,10,20}` (`k=0` = spline only).

## Success criteria

- Mid-path (`t=0.5`) MAE → claimed stays near the spline-only number (~5°)
  for spline+tube-along-path (tube ramp should not recreate chord teleport).
- Final MAE → target falls from ~18° toward multi-probe’s ~8° as `k` grows.
- Tube PCA variance: a few components should explain most residual energy;
  if not, the “tube” is a ball and this framing fails.

## Out of scope

Pullback (separate script). Speed/acceleration (already flat). New encoder
passes.
