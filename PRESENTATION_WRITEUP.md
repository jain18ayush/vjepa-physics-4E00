# Walking through the experiment

A 15-minute account of how frozen V-JEPA 2 represents direction, speed, and
acceleration on the take-home disk videos. Numbers below are the ones in
`artifacts/results/`. Method detail lives in the Part 1 and Part 2 plans;
this is the path through the results.

The talk has one argument:

**Speed and acceleration are low-dimensional and already linear. Direction
is a circle, and the activations sit in a tube around that circle.** Part 1
shows the circle needs many linear probes. Part 2 shows a one-dimensional
spline can still walk the angle. The follow-ups show why both are true:
the extra dimensions are the tube’s cross-section, partly a fiber of the
angle readout and partly speed.

---

## 1. What was asked

The take-home is a small reproduction of Joseph et al., *Interpreting
Physics in Video World Models*, plus an open extension using the manifold
steering geometry from Wurgaft et al. / Goodfire.

Three physical variables, each on its own set of synthetic clips:

| Dataset | Clips | Target | What varies |
|---|---:|---|---|
| Direction | 1,500 | `theta_degrees`, 64 angles in [0°, 360°) | Half constant velocity, half accelerating |
| Speed | 1,536 | magnitude, 0.25–4.0 m/s | Direction also varies; it is not the target |
| Acceleration | 1,536 | magnitude, 0.25–10.0 m/s² | Initial speed is 0 |

Every clip is one blue disk on a dark field: 256×256, 16 frames, 24 fps.

The encoder is frozen `facebook/vjepa2-vitl-fpc64-256`: ViT-L, 24 blocks,
width 1024. The predictor is never run. Each clip is 16 frames, not padded
to the pretraining length of 64. Each block returns 2,048 tokens. Those
tokens are mean-pooled to one 1,024-dimensional vector per layer. That is
the probe input for everything after.

Splits are seed 0, stratified by label bin, and locked before any fit.
Roughly: direction 1,116 / 128 / 256; speed and acceleration 1,088 / 128 /
320. Validation chooses the layer. Test is only for the final readout and
for steering.

Direction is never trained as a wrapped angle. The probe target is
`(sin θ, cos θ)` with mean-squared error. The reported angle error is
circular MAE. Chance on a uniform circle is 90°.

---

## 2. Part 1 — three experiments, one layer

### Where each variable becomes readable

A linear probe is trained at every block. The curve is 5-fold group
cross-validation on train+val, grouped by start position. The layer used
later is chosen on a separate train-fit / val-score table, not on test.

Speed and acceleration are already strong at layer 0: validation R² about
0.80 and 0.79. Constant-velocity direction is weak there, R² 0.23, then
jumps to 0.93 at layer 5 and stays high (0.97 at layer 23). The emergence
rule — first layer at the midpoint between layer-0 and peak direction R²,
provided both scalars are still above 0.5 — selects **layer 5**. Call it
`L*`. The paper’s Physics Emergence Zone on Kubric balls was layer 8 of
this same architecture. Same shape of result, earlier on these disks.
Their direction curve softened toward the output. Ours does not. The beige
band at layers 7–9 on the figure is their zone drawn in for reference.

### How many dimensions

At layer 5, fit a probe, score it, delete the subspace spanned by its
weight with a QR projection, and repeat. The probe that falls below the
stop rule is not kept.

Stop rules, on validation: direction dies at R² < 0.1 or circular MAE >
80°; scalars die at R² < 0.05 or at 90% of the error of predicting the
training mean.

| Run | Probes kept | Dimensions | Shape of the curve |
|---|---:|---:|---|
| Direction, constant velocity | 41 | 82 | Jagged, sawtooth flagged |
| Direction, accelerating | 31 | 62 | Sawtooth |
| Direction, mixed | 23 | 46 | Sawtooth |
| Speed | 5 | 5 | Smooth decay |
| Acceleration | 2 | 2 | Smooth decay |

The quoted direction number is the constant-velocity run, matching the
paper’s direction set. Their estimate near the emergence zone was about
40–50 features. Ours is 82, on 64 angle bins instead of 8. The comparison
that matters is the ordering: direction is a large redundant subspace;
speed and acceleration are compact. The sawtooth is the sine/cosine pair
structure showing up as the subspace is peeled.

Speed’s R² at the stop was still about 0.13. The mean-baseline MAE rule
fired first. That is the appendix rule, and it is worth saying if someone
asks why speed looks “dead” while a little linear signal remains.

### Whether the subspace is causal

Steering uses the 41 constant-velocity probes. They never saw test. A
fresh evaluation probe is fit on unsteered test activations only
(R² 0.71, circular MAE 19.2° — weaker than the paper’s 0.99 on 8 angles,
which is what 64 bins on a flat disk should do).

Every test clip is steered toward 90°. Inside the span of the first `n`
probe weights, the coordinates are replaced so each included probe outputs
`(sin 90°, cos 90°)`. The orthogonal complement of that span is left
alone. The frozen evaluation probe then reads the edited vector.

| Probes | MAE to 90° | MAE to the true angle |
|---:|---:|---:|
| 0 | 90.3° | 19.2° |
| 1 | 58.4° | 38.1° |
| 5 | 11.4° | 82.3° |
| 20 | 6.3° | 88.9° |
| 41 | 4.4° | 88.6° |

One probe moves the decoded angle and does not finish the job. By five
probes the readout is close to the target and far from the clip’s real
angle. Velocity-only and acceleration-only test clips follow the same
pattern (at 20 probes: 7.2° and 5.5° to the target). A sanity check at
full `n`: the steering probes themselves predict `(1, 0)` to numerical
precision. The evaluation probe is a different readout, so it is allowed
to miss by a few degrees.

That is the paper’s causal signature, reached sooner than their layer-8
Kubric run (they were near 12° at 20 probes; we are near 11° by 5 probes
and 6° by 20). Simpler videos, an earlier layer, and a large basis.

Part 1 verdict, in one line: the three qualitative claims reproduce.
Emergence is earlier, the late-layer drop is absent, and steering is
easier. Those offsets fit a simpler visual world.

---

## 3. Part 2 — is the 82-dimensional subspace a curve?

Goodfire’s question, translated: if behavior lies on a low-dimensional
manifold in activation space, steer along that manifold instead of along
an arbitrary linear subspace.

There is no token distribution here. The mountain-car substitution is the
right one. The “behavior” is a frozen linear probe. The manifold is a
cubic spline through one centroid per label, in a 64-component PCA of
layer 5, fit on train only. Direction uses the constant-velocity train
clips (558) and a periodic spline. Speed and acceleration use natural
cubics. The test clip only contributes the coordinates outside those 64
components. Inside the PCA block, the activation is replaced by a point
on the path.

Two paths share endpoints:

- **Spline.** Walk the intrinsic coordinate. For direction, take the short
  arc.
- **Chord.** Straight line in PCA space between the same centroids. This
  is linear steering inside the same subspace, so the only difference is
  the shape of the path.

A third method, for direction only, is the Part 1 multi-probe edit aimed
at the same endpoint.

### Direction is curved

Half turn, 0° → 180°, held-out test, spline-run probe (R² 0.80, circular
MAE 14.5° on unsteered test):

| Method | Error at the middle of the path | Error at the target |
|---|---:|---:|
| Periodic spline | 5.5° | 18.6° |
| Chord | 170.9° | 18.6° |
| Multi-probe, 1 direction | — | 70.3° |
| Multi-probe, 5 | — | 14.4° |
| Multi-probe, 20 | — | 8.5° |

The spline’s decoded angle tracks the ramp from 0° to 180°. The chord
enters the interior of the circle and the probe teleports; the mid-path
error peaks near 171°. The two paths share endpoints, so they finish
together, at 18.6°. Twenty orthogonal probes, aimed only at the endpoint,
finish at 8.5°.

So a 1D periodic curve is enough to *visit intermediate angles*. Forcing
a single held-out angle as hard as Part 1 still wants the redundant
subspace. The curve is a spine. The Part 1 dimensions are the thickness
around it.

On a short arc, 22.5° → 337.5°, the chord barely leaves the manifold.
Spline and chord both finish at 1.3°. Multi-probe at 20 directions
finishes at 8.4°, worse: it overwrites a large subspace to hit a nearby
angle. The spline is the better tool when the target is a neighbor on
the circle.

Accelerating test clips, which were not in the spline fit, finish at
18.4° on the half turn. The velocity-trained circle transfers.

### Speed and acceleration are flat

Speed, 0.49 → 3.52 m/s. Mid-path error to the claimed magnitude is 0.37
for the spline and 0.10 for the chord. Both finish at decoded 2.86
against a target of 3.52 (error 0.67). The chord is the closer ramp.
Curvature is not buying control. That matches Part 1: speed lived in
about five dimensions and was linearly readable at layer 0. The missed
endpoint is a real failure mode of this protocol: the train centroid for
3.52 m/s does not sit on the global probe’s isosurface for 3.52.

Acceleration tells the same geometry with a weaker ruler. The unsteered
test probe has R² 0.33 and MAE 1.99 m/s². Mid-path error is 0.58
(spline) versus 0.04 (chord); both finish 2.57 away from the claimed
endpoint. Relative to a 2-dimensional nullspace, Euclidean interpolation
is already the right steering geometry. Absolute errors should be quoted
softly.

Median distance from a test point to the fitted curve is about 4.8 PCA
units for all three variables. Direction still has a usable spine. The
shared residual says “tube,” which is the next experiment, not a reason
to throw out the spline.

---

## 4. The spine is real, and it is not the whole object

### Pullback leaves the spline

Wurgaft et al. optimize a path so the probe tracks a target schedule, and
ask whether that path falls back onto the activation manifold. Here the
path is initialized on the chord, in the first 32 principal components,
and L-BFGS’d so the frozen probe follows the claimed angle.

On 0° → 180°:

| Path | Mean distance to the spline | Mid-path error | Final error |
|---|---:|---:|---:|
| Spline | 0.04 | 12.5° | 19.8° |
| Chord | 1.60 | 174.6° | 19.8° |
| Pullback | 3.31 | 1.05° | 1.08° |

Recovery fraction versus the chord-to-spline gap: **−1.09**. The
optimized path is farther from the spline than the chord was, and the
probe reading is much better. Matching the readout and staying on the
centroid curve are different goals. That is what a tube predicts: many
off-spine offsets still decode as the right angle.

The short arc is milder (pullback distance 1.02 versus chord 0.81) and
still not a return to the spline.

A note if the pullback markdown is open during questions: that file’s
table lists the geometric final error as 8.8°. The result JSON has
19.8°, the same number as the chord endpoint. Endpoints coincide by
construction, so 19.8° is the figure to defend. Also, this pullback/tube
probe reads the same spline at 12.5° mid-path, where the earlier spline
script read 5.5°. Compare methods inside one run. The ordering is stable:
spline tracks, chord teleports, pullback wins the probe and leaves the
curve.

### Two normal directions close the endpoint gap

For each training activation, take the nearest point on the spline,
subtract it, and remove the component along the tangent. What remains is
the normal residual.

| Quantity | Value |
|---|---:|
| Median distance to the spine | 4.62 |
| Mean energy along the tangent | 0.03 |
| Mean energy perpendicular to the tangent | 26.45 |
| Variance in the first 5 normal components | 64% |
| Variance in the first 20 | 91% |

Almost all of the scatter is normal to the circle. The tube is real, and
it is moderately low-rank rather than a ball: 20 components cover 91%,
but the first two cover only 37%. Rank and “what you need for steering”
are not the same thing, which the next table shows.

Steering pins the activation to the spine and adds a ramped offset in a
`k`-dimensional normal basis, chosen so the probe reads the target angle
at the end of the path. On 0° → 180°, inside this run:

| Method | Mid-path error | Final error to target |
|---|---:|---:|
| Spline only | 12.5° | 19.8° |
| Chord | 174.6° | 19.8° |
| Spine + residual-PCA tube, k = 1 | 11.7° | 19.7° |
| Spine + residual-PCA tube, k = 2 | 4.2° | 2.8° |
| Spine + nullspace tube, k = 1 | 7.8° | 2.6° |
| Multi-probe, n = 20 | — | 3.5° |

One variance-maximizing normal direction does almost nothing for the
endpoint. Two of them, or a single nullspace direction orthogonalized to
the tangent, land on the target at least as well as 20 unrestricted
probes, and the middle of the path stays on the angle ramp. Past k = 2
the numbers do not move. The control that was missing is a small normal
correction, not the rest of the 82-dimensional edit.

On the short arc the same k = 2 correction finishes at 2.4°, against
7.1° for the spline alone and 3.9° for 20 probes.

Practical recipe on this data: walk the periodic spline when the angle
should pass through intermediates; add about two normal coordinates to
nail an endpoint.

---

## 5. Why the tube has thickness

Five checks on the same layer-5 direction spline. The point of the
campaign is to stop treating “82 dimensions” as 82 copies of angle.

**The cross-section is mostly a fiber.** From an anchor on the spine, a
unit step along the tangent moves the decoded angle by 8.8°. The same
step along tube PC1 moves it 2.2°. PC2 moves it 0.14°. The tube is much
closer to a level set of the angle map than the spine is. It is not a
perfect fiber: PC1 still leaks.

**The fiber is not empty.** A ridge regression on the spine coordinates
`(sin u, cos u)` predicts direction (R² 0.55) and does not predict speed
(R² about 0). The first 10 tube components do the opposite: direction
R² −0.54, speed R² 0.70, start-`y` R² 0.49, motion-type accuracy 0.78
versus 0.52 from the spine. Split of labor: the spine owns angle; the
tube owns speed, motion type, and vertical start position.

**Removing those labels does not collapse the tube.** Regressing start
position out of the activations removes about 3% of perpendicular energy.
Regressing speed out removes 28% (energy 26.5 → 19.1). About 70%
remains. The tube is partly a speed bundle and partly residual fiber.

**Part 1’s probes live in the tube, and they are not the top variance
axes.** The first five nullspace planes overlap the 20-dimensional tube
subspace at fractions 0.71–0.89. The mean squared overlap of the first 20
probes with the leading tube components is only 0.003, 0.007, and 0.05
at k = 1, 2, 5, then 0.29 at k = 10 and 0.45 at k = 20. Steering-relevant
directions and high-variance nuisance directions are different slices of
one neighborhood. That is why a nullspace-aligned normal worked at k = 1
while residual-PCA needed k = 2.

**Radius is a weak uncertainty signal.** Correlations of tube radius with
probe confidence, angular error, and belief entropy all sit around 0.3
or below. The tube is not primarily a confidence dial.

**Depth inflates the tube and softens the angle.** Same construction at
four layers, constant-velocity fit:

| Layer | Perpendicular energy | Tangent sensitivity | Tube-PC1 sensitivity |
|---:|---:|---:|---:|
| 0 | 3.4 | 28.6° | 15.1° |
| 5 | 26.5 | 8.8° | 2.2° |
| 12 | 49.5 | 4.3° | 0.37° |
| 23 | 221 | 0.78° | 0.17° |

After direction becomes linearly available, the representation does not
collapse onto a wire. Later layers keep more ways to realize the same
angle, and a single step moves that angle less. Part 1’s growing
direction dimensionality is this thickening, seen with probes.

---

## 6. The sentence the talk should land on

Direction in this encoder is an angle on a circle, entangled with speed
in the normal bundle, with a leftover fiber that linear probes can still
steer. Speed and acceleration do not need that story. They are flat, so
a chord is enough, and for acceleration the probe is the limiting
instrument.

What this changes about “physics in the latent”: a concept coordinate and
the support of the activations are different objects. The concept is
still a circle. The activations occupy a tubular neighborhood whose
coordinates are roughly `(θ, speed, and a fiber)`. Factored control
should treat direction and speed as a sheet, or residualize one before
fitting the other. Two independent 1D splines miss the coupling that the
nuisance audit measured.

---

## 7. What to concede in discussion

- Mean-pool only. No attentive pooling, no patch probes, no attention
  ablations, no IntPhys.
- The steered object is the pooled residual, scored by a linear probe.
  Pixels are never edited, and the predictor never runs. “Causal” here
  means causal for the probe.
- Acceleration’s Part 2 numbers are soft because the evaluation probe is
  weak (R² 0.33), not because the nullspace result is ambiguous.
- Absolute degrees move a few points when the evaluation probe is refit.
  Quote comparisons from a single result file.
- The world is one disk. A Kubric scene could move the emergence layer,
  the tube rank, and how much of the tube is speed. The qualitative split
  — scalars early and flat, direction later and curved — is what this
  dataset can support.
