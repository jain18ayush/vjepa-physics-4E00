# Why is direction a tube?

Five-step campaign on the Part 2 direction spline at V-JEPA 2 `L* = 5`
(plus a depth filmstrip). Script: `scripts/part2_tube_why.py`. Numbers:
`artifacts/results/tube_why_campaign.json`.

**Setup.** Periodic cubic spine fit on train / velocity. Tube = PCA of
residuals after removing the local tangent. Nuisance / belief stats use all
train clips scored in that frame. Fiber / conditional / layers use the
velocity fit set unless noted.

---

## Verdict in one paragraph

Direction is a **circular spine** whose **cross-section mixes two things**:
(1) a **probe fiber** — many activations map to the same angle — and
(2) a **physics nuisance bundle**, especially **speed** (and somewhat
start-`y` / motion type). It is not pure estimation noise, and it is not
only redundant copies of the circle. Depth makes the tube **thicker** while
angle becomes **less locally sensitive** — capacity grows around a
stabilizing concept coordinate. That is why Part 1 needed many probes, Part 2
could walk angles on a 1D spline, pullback could leave `Mh` and still improve
the probe, and spine+tube `k≈2` could match multi-probe at the endpoint.

---

## Step 1 — Fiber test + nuisance audit

### Fiber: does the tube keep θ fixed?

Displace a unit vector from an anchor near 90° and read the frozen probe.

| Direction | Sensitivity (° per unit ε) |
|---|---:|
| Spine tangent | **8.82** |
| Tube PC1 | 2.16 |
| Tube PC2 | 0.14 |
| Tube PC3 | 0.65 |

![Fiber test](artifacts/figures/tube_why_fiber.png)

The tangent moves angle ~4× harder than the leading tube PC, and PC2 is
almost flat. So the cross-section is **much closer to a fiber of the
direction map** than the spine is — hypothesis **A** holds in relative
terms. It is not a perfect fiber: tube PC1 still leaks ~2°/ε into θ.

### Nuisance: what else lives in the tube?

Cross-validated Ridge R² (spine = `(sin u, cos u)`; tube = first 10 residual
PCs):

| Target | Spine R² | Tube-10 R² | Both |
|---|---:|---:|---:|
| Direction `(sin θ, cos θ)` | **0.55** | −0.54 | 0.79 |
| Speed (m/s) | −0.01 | **0.70** | 0.70 |
| Acceleration (m/s²) | −0.01 | 0.24 | 0.23 |
| Start `(x, y)` | −0.01 | 0.24 | 0.24 |
| Start `y` alone | −0.00 | **0.49** | 0.49 |
| Start `x` alone | −0.01 | −0.00 | −0.01 |
| Motion type (accuracy) | 0.52 | **0.78** | 0.79 |

![Nuisance audit](artifacts/figures/tube_why_nuisance.png)

**Split of labor.** The spine owns angle. The tube does **not** own angle
(negative R² alone) but strongly owns **speed**, classifies motion better
than chance, and carries start-`y`. Hypothesis **B** is real for speed /
motion; not a generic “everything” soup.

---

## Step 2 — Copy alignment with Part 1 probes

Nullspace probe planes (Part 1) projected into PCA, compared to the tube
basis:

- Early probe planes sit mostly inside the 20-D tube subspace
  (overlap fractions ~0.71–0.89 for the first five probes).
- But **variance-PCA tube prefixes do not capture probe energy early**:
  mean squared overlap of the first 20 probes is only 0.003 / 0.007 / 0.05
  for tube `k = 1/2/5`, rising to 0.29 at `k=10` and 0.45 at `k=20`.

![Copy alignment](artifacts/figures/tube_why_copy.png)

**Read.** Hypothesis **C** is half-true: multi-probe directions live in the
tubular neighborhood of the circle, but they are **not** the same as the
top residual-variance axes. Steering-relevant tube coords ≠ leading
nuisance PCs. That explains why residual-PCA `k=2` and nullspace-tube `k=1`
both worked in the earlier tube experiment — different slices of the same
fat neighborhood.

---

## Step 3 — Radius and belief

Tube radius `ρ = ‖r_⊥‖` vs probe / belief:

| Correlation | Value |
|---|---:|
| ρ ↔ probe confidence `‖sincos‖` | −0.30 |
| ρ ↔ probe angular error | +0.29 |
| ρ ↔ belief entropy (softmax to centroids) | −0.18 |
| ρ ↔ belief mode error | +0.41 |

![Radius / belief](artifacts/figures/tube_why_radius.png)

**Read.** Larger radius ↔ slightly worse / less confident direction
readout, but the links are **weak**. Hypothesis **D** (radius = uncertainty)
is only a minor contributor. The tube is not primarily a confidence dial.

Along a tube-PC1 fiber at fixed spine `u`, belief entropy and probe θ move
only mildly compared with tangent walks in Step 1 — again fiber-like, with
leakage.

---

## Step 4 — Conditional spine (collapse nuisances)

Refit the circle after linearly removing covariates from activations; measure
perpendicular energy:

| Condition | Perp energy | Ratio vs baseline |
|---|---:|---:|
| Baseline | 26.45 | 1.00 |
| Regress out start `(x,y)` | 25.54 | 0.97 |
| Regress out **speed** | **19.06** | **0.72** |
| Regress out xy + speed | 18.38 | 0.70 |

**Read.** Position barely thins the tube. **Speed removes ~28%** of tube
energy and does not kill it. A thick residual remains (~70%). So the tube is
**partly a speed (and motion) bundle around the direction circle**, and
**partly a residual fiber / copy space** unexplained by those labels.
Hypotheses **A+B+C** together; not **B** alone.

---

## Step 5 — Layer filmstrip

Same spine+tube construction at L0, L5, L12, L23 (train / velocity):

| Layer | Perp energy | Tang energy | Fiber sens (tangent) | Fiber sens (tube PC1) | ρ mean |
|---|---:|---:|---:|---:|---:|
| 0 | 3.4 | 0.05 | 28.6 | 15.1 | 1.6 |
| **5 (`L*`)** | **26.4** | 0.03 | **8.8** | **2.2** | 4.9 |
| 12 | 49.5 | 0.04 | 4.3 | 0.37 | 6.8 |
| 23 | 220.7 | 0.27 | 0.78 | 0.17 | 14.4 |

![Layers](artifacts/figures/tube_why_layers.png)

**Read.** After direction emerges, the representation does not collapse onto
a wire. The tube **inflates with depth** while local angle sensitivity
**falls** — more ways to realize the same θ, softer single-step geometry.
That matches Part 1’s growing direction dimensionality / sawtooth toward
later layers, and Goodfire’s moral that geometry (here: a thickening
bundle) is the right object for control.

---

## What this means for manifolds and steering

1. **Concept manifold ≠ activation support.** The concept is still
   `S¹` (angle). Activations occupy a **tubular neighborhood** whose
   intrinsic coords are roughly `(θ, speed, …, fiber/copy)`.

2. **Why multi-probe worked.** Part 1’s orthogonal probes explore that
   neighborhood until the readout saturates — not because angle needs 82
   independent concepts.

3. **Why the spline walked mid-path.** Ordering lives on the spine
   (tangent).

4. **Why pullback left `Mh`.** Probe loss cares about θ (and whatever
   correlates with confidence); off-spine shortcuts in the fiber/nuisance
   directions can improve the probe without staying on the centroid curve.

5. **Why spine+tube `k≈2` matched `n=20`.** A few well-chosen normal
   coordinates (especially nullspace-aligned ones) beat raw variance PCs;
   you do not need the whole 82-D ambient edit once you pin the spine.

6. **Impact for “physics in the latent.”** Direction is not an isolated
   circle. On these disks it is **entangled with speed in the normal
   bundle**. Factored control (Goodfire ICLR-style) should treat
   `(θ, speed)` as a sheet/cylinder, not two independent 1D splines,
   unless you residualize.

---

## Hypothesis scorecard

| Hypothesis | Score | Evidence |
|---|---|---|
| **A. Probe fiber** | Strong (relative) | Tangent ≫ tube sensitivity; tube alone cannot predict θ |
| **B. Nuisance bundle** | Strong for speed / motion | Tube R²(speed)=0.70; motion acc 0.78; −28% energy after speed residualize |
| **C. Redundant copies** | Partial | Probes ⊂ tube20, but not = top variance PCs |
| **D. Uncertainty radius** | Weak | |r| with entropy/confidence ≲ 0.3 |
| **E. Estimation fluff** | Rejected | Tube steers; stable energy; depth structure; nuisance signal |

---

## Artifacts

```text
scripts/part2_tube_why.py
artifacts/results/tube_why_campaign.json
artifacts/figures/tube_why_fiber.png
artifacts/figures/tube_why_nuisance.png
artifacts/figures/tube_why_copy.png
artifacts/figures/tube_why_radius.png
artifacts/figures/tube_why_layers.png
artifacts/meta/direction_meta.json
```

```bash
cd /workspace/vjepa-physics/scripts
python part2_tube_why.py --device cuda
```
