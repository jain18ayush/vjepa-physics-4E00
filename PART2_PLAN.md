# Part 2 plan: one spline each for speed, direction, and acceleration

Minimum extension of Part 1. Fit the Goodfire activation spline
([`manifold_explainer.html`](manifold_explainer.html) is the method sketch;
Wurgaft et al., arXiv:2605.05115, §2.2, §3.1–3.2, App. A.3 and A.6, and the
mountain-car variant in §5 and App. B) to the activations Part 1 already
saved, then steer on held-out clips and compare with the multi-probe edit.

Leave pullback optimization, thin-plate surfaces, Hellinger isometry sweeps,
and new encoder passes out of this pass.

## What “behavior” is here

V-JEPA does not emit a distribution over concept tokens. Use the mountain-car
substitution, not the language-model one.

- **Activation manifold.** A 1D cubic spline through centroids in a PCA
  subspace of layer `L* = 5` (the Part 1 layer).
- **Readout.** The same kind of linear probe as Part 1, fit without ever
  seeing a steered vector. Report the probe’s decoded scalar or angle along
  the path.
- **Optional belief.** Softmax over squared distance from the steered vector
  to the centroids, temperature 0.5, as in their equation (9). Use it only as
  a picture of whether intermediate mass sits on intermediate bins.

## Data

Reuse `artifacts/activations/{direction,speed,acceleration}.npz`, shape
`(N, 24, 1024)`, and the seed-0 splits locked in Part 1. Fit centroids, PCA,
and the spline on **train only**. Validation chooses nothing that needs a
search beyond a single recorded residual. Test is evaluation only.

| Variable | Labels | Train centroids | Spline | Intrinsic coordinate |
|---|---|---|---|---|
| Speed | 64 magnitudes, 0.25–4.0 m/s | Mean activation per exact magnitude | Natural cubic | The magnitude itself, scaled to [0, 1] |
| Acceleration | 64 magnitudes, 0.25–10 m/s² | Same | Natural cubic | The magnitude itself, scaled to [0, 1] |
| Direction | 64 angles in [0°, 360°) | Mean activation per exact angle | Periodic cubic | `theta_degrees / 360` |

Direction clips mix constant-velocity and accelerating motion. Fit the
direction spline on the **train / velocity** subset, matching the Part 1
steering basis. Score accelerating test clips as a transfer check, separately.

Sixty-four populated labels are the discrete concepts. Do not re-bin. With
~1,088 train clips that is on the order of 15–17 samples per centroid.

## Fit

At layer 5, on the train matrix:

1. PCA, fit on train. Keep 64 components, matching App. A.3. Record the
   fraction of variance kept. Speed and acceleration may concentrate in far
   fewer components; that number is worth stating next to the Part 1
   dimensionalities (speed 5, acceleration 2, direction 82).
2. Project, average to one centroid per label value.
3. `scipy.interpolate.CubicSpline` through the centroids in order, one spline
   per PCA coordinate.
   - Speed and acceleration: `bc_type="natural"`.
   - Direction: `bc_type="periodic"`.
4. The spline interpolates the centroids. No smoothing penalty. If a centroid
   is empty, drop that label rather than imputing it.

Also store, for every train centroid and every test vector, the Euclidean
distance in PCA space to the nearest point on the spline. A large test
residual means the representation is a tube around the curve. Part 1’s
direction dimensionality says this residual should be big for direction and
small for the scalars.

## Steer

Endpoints are two label values. The path has `K = 21` waypoints, `t = 0 … 1`.

- **Spline path.** `u(t)` moves monotonically in the intrinsic coordinate.
  For direction, unwrap so the walk takes the shorter arc (the geodesic on
  the circle). Decode with the spline, inverse-PCA back to 1024 dimensions,
  and add the test clip’s own leftover coordinates (the part outside the 64
  PCs). This is App. A.6: the base activation changes only inside the
  subspace the manifold lives in.
- **Chord path.** The same endpoints in PCA space, linearly interpolated,
  then the same inverse-PCA plus leftover. This is the linear-steering
  control, restricted to the same subspace so the comparison is the path
  shape.
- **Multi-probe path, direction only.** The Part 1 subspace edit toward a
  fixed target (90°), already computed. It is a third method, not a
  reimplementation of the spline. Quote it at `n = 1`, `n = 5`, and
  `n = 20` from `PART1_WRITEUP.md`.

Base points are **test** activations. The path does not depend on the test
clip’s label inside the PCA block; the clip only contributes the orthogonal
leftover. That matches the paper’s “replace the subspace, keep the rest.”

Suggested endpoint pairs, all inside the label grid:

| Variable | Pair | Why this pair |
|---|---|---|
| Speed | 0.5 → 3.5 m/s | Interior of the range, not an endpoint of the spline |
| Acceleration | 1 → 8 m/s² | Same |
| Direction | 0° → 180° | Half turn, where a chord crosses the interior |
| Direction | 20° → 340° | Short arc across 0°. A non-periodic fit would take the long way |

## Scores, on test

Fit one evaluation probe per variable on **unsteered test** activations at
layer 5, with the Part 1 target (`(sin θ, cos θ)` or raw magnitude). Apply
that frozen probe to every waypoint. Do not refit it on steered vectors.

For each method and each waypoint `t`:

- Decoded value, mean over test clips.
- Error of that decode to the label the path claims at `t`
  (circular MAE for direction, MAE for scalars).
- Error of that decode to the clip’s true label.

The Part 1 signature, translated:

- At `t = 1`, error to the target is low and error to the truth has risen.
- Between the ends, the spline decode moves through the intermediate labels.
  The chord decode jumps, stalls, or leaves the label range.
- On direction, the short-arc pair stays on the short arc only for the
  periodic spline.

Plots to save under `artifacts/figures/`:

1. Layer-5 PCA plane (first three components, or the top two) colored by
   label, with the spline and one chord drawn on it. One panel per variable.
2. Decoded value versus `t` for spline and chord, with the ideal ramp.
3. A small table: final error to target for chord, spline, and (direction)
   multi-probe at 20 probes.

## How to read the three variables

Write the comparison in those terms. The expected pattern, to be confirmed
rather than assumed:

- **Acceleration, then speed.** Low-dimensional and already linear in Part 1.
  Spline and chord should be close. State the gap in MAE. A small gap means
  the flat geometry really is the right steering geometry for that variable.
- **Direction.** Part 1 needed many orthogonal probes. If the periodic spline
  reaches the target about as well as 20 probes, the extra dimensions were
  redundant copies of one curve. If the spline stalls and the residual to the
  curve stays large while multi-probe still hits ~6°, direction is a
  subspace, and a 1D spline is the wrong geometry. Either outcome is the
  result.
- **Failure cases to report if they show up.** Short-arc pair walking the
  long way. Accelerating direction clips whose centroids were not in the
  fit. Test points whose leftover coordinates dominate the probe. Spline
  endpoints that match the centroid probe reading but miss the target
  because the centroid itself is off the global probe.

## Out of scope

- Refitting the spline at every layer. Layer 5 is the comparison point with
  Part 1 steering. A single extra panel at layer 0 for speed is optional,
  and only if the layer-5 scalar result is ambiguous.
- Pullback (L-BFGS so that probe outputs track a target path). The paper
  uses it as causal evidence that the two manifolds match. The forward
  comparison above is the minimum.
- Editing pixels or running the V-JEPA predictor. The steered object is the
  pooled residual, scored by a probe.
- New videos, new pooling, or changes under `data/`.

## Done when

`artifacts/figures/` has the three PCA overlays and the waypoint decode
curves, a short `PART2_WRITEUP.md` states the three comparisons with the
held-out numbers, and the direction paragraph says explicitly whether the
1D periodic spline matched or lost to multi-probe steering.
