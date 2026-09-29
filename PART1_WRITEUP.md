# Part 1 results: physics probes in V-JEPA 2

Reproduction of the main experimental progression in Joseph et al.,
[*Interpreting Physics in Video World Models*](https://arxiv.org/abs/2602.07050),
on the take-home synthetic disk videos. Qualitative agreement with the paper is
the target, not matching their exact R².

Frozen encoder: `facebook/vjepa2-vitl-fpc64-256` (ViT-L/16, 24 blocks, width
1024). Pooling: mean over all space–time tokens (`mean_spacetime`). One-clip
assert: 16 frames in → 2048 tokens × 1024 per block, 25 hidden states (embedding
+ 24 blocks). Artifacts live under `artifacts/`. Protocol details are in
[`PART1_PLAN.md`](PART1_PLAN.md).

---

## What we set out to reproduce

The paper’s Part 1–relevant claims, distilled:

1. **Layer-wise availability.** Scalar speed and acceleration magnitude are
   linearly readable from early layers. Motion direction becomes linearly
   available only around the Physics Emergence Zone (~⅓ depth for V-JEPA 2-L,
   near layer 8 of 24), peaks in the middle, and softens toward the output.
2. **High-dimensional redundancy.** Iterative nullspace probing needs many
   orthogonal directions before direction dies (~40–50 features; sawtooth
   pattern). Speed is lower-dimensional and decays smoothly.
3. **Multi-probe steering.** Editing one probe axis barely changes decoded
   angle. Coordinating many orthogonal probes does. On held-out data, their
   layer-8 experiment moved MAE to a fixed target from ~83° down to ~12° with
   ~20 probes.

Our data is simpler than their Kubric balls: flat blue disk, 256×256, 16 frames,
64 direction bins (vs their 8), mixed velocity/acceleration in the direction
set. Exact numbers should differ; the ordering and the multi-probe steering
signature should not.

---

## Methods

### Data and splits

| Dataset | Clips | Target | Train / val / test |
|---|---:|---|---|
| Direction | 1,500 | `theta_degrees` → `(sin θ, cos θ)` | 1,116 / 128 / 256 |
| Speed | 1,536 | `magnitude` (= `speed_mps`) | 1,088 / 128 / 320 |
| Acceleration | 1,536 | `magnitude` (= `acceleration_mps2`) | 1,088 / 128 / 320 |

Direction clips are half constant-velocity and half accelerating. Splits are
stratified by label bin (direction: `(θ, motion)`; scalars: magnitude), seed 0.
Cross-validation groups are start position quantized to 0.5 m. Test IDs are
locked before any probe, QR, or steering solve.

### Activation extraction

For each clip: decode exactly 16 RGB frames, run
`AutoVideoProcessor` + frozen `AutoModel` with `output_hidden_states=True` and
`skip_predictor=True`, mean-pool each block’s `(2048, 1024)` residual to one
vector. Saved as `artifacts/activations/<dataset>.npz` with shape
`(N, 24, 1024)`.

### Experiment A — layer-wise probes (Appendix B)

At every layer, train a linear probe `W h + b` with Adam and the paper’s 20-pair
sweep over learning rate and weight decay (100 epochs, MSE). Direction uses
sine/cosine targets; scalars use the raw magnitude.

- **Curves:** 5-fold `GroupKFold` on train+val; report mean ± std R² and error.
- **Series:** `direction_all`, `direction_velocity`, `direction_acceleration`,
  `speed`, `acceleration`.
- **Layer choice (val only):** on velocity-direction val R², let
  `L_emerg` be the first layer at or above `(r0 + rmax) / 2`. Use it as `L*` if
  speed and acceleration val R² there are both ≥ 0.5; else use `L_peak`.

### Experiment B — iterative nullspace (Appendix C.11)

At `L*`, repeatedly: fit a C.11 probe (Adam `1e-3`, WD `1e-4`; 100 epochs for
direction, 50 for scalars), score on val, QR the weight, project that subspace
out of the activations, repeat until stop.

Stop (val): direction `R² < 0.1` or circular MAE `> 80°`; scalars `R² < 0.05`
or MAE `> 0.9 ×` mean-baseline MAE. Dimensionality is `2K` for direction and
`K` for scalars.

### Experiment C — multi-probe steering (Appendix C.12)

Build an orthonormal basis from the **train / velocity** nullspace probes.
Fit a fresh evaluation probe on **unsteered test** only. For each prefix of
`n` probes, replace each test activation’s coordinates inside the subspace so
every included probe outputs `(sin 90°, cos 90°)`, leave the orthogonal
complement alone, and read the steered vector with the evaluation probe.
Report circular MAE to 90° and to the true angle.

---

## Results

Paper figure crops are taken from Joseph et al. (arXiv:2602.07050) for visual
comparison only. Full pages are under `artifacts/paper_figures/`.

### 1. Layer-wise probes

<table>
<tr>
<td width="50%" valign="top">

<strong>Paper — Figure 2c (polar)</strong><br>
V-JEPA 2-L on Kubric balls. Speed / accel magnitude high from the start;
direction jumps at <strong>layer 8</strong> (~⅓ depth).

<img src="artifacts/paper_figures/paper_fig2c_polar.png" alt="Paper Figure 2c polar layer curves"/>

</td>
<td width="50%" valign="top">

<strong>Ours — layer curves</strong><br>
Same encoder, take-home disks, mean space–time pool. Direction jumps at
<strong>layer 5</strong> (`L*`). Speed / accel high from L0.

<img src="artifacts/figures/layer_curves.png" alt="Our layer-wise probe curves"/>

</td>
</tr>
</table>

Paper full Figure 2 (dataset + Cartesian + polar) for context:

![Paper Figure 2](artifacts/paper_figures/paper_fig2_layers.png)

**What we see.** Speed and acceleration are already strong at layer 0 (val R²
≈ 0.80 and 0.79) and stay high. Direction is weak early (velocity subset
R² ≈ 0.23 at L0, circular MAE tens of degrees) and jumps sharply at **layer 5**
(R² ≈ 0.93). After that it stays high through the rest of the network. The
plan’s midpoint rule picked `L_emerg = 5 = L*`. Speed and acceleration at that
layer are both above 0.5 (0.95 and 0.92), so no peak fallback. Temporal-last
pooling ablation was **not** triggered (`r0 = 0.23 < 0.5`).

| Series (val R²) | L0 | L5 (`L*`) | L8 | L23 |
|---|---:|---:|---:|---:|
| Direction (velocity) | 0.23 | 0.93 | 0.96 | 0.97 |
| Speed | 0.80 | 0.95 | 0.96 | 0.96 |
| Acceleration | 0.79 | 0.92 | 0.94 | 0.94 |

**Side-by-side read.** Same qualitative story as their Figure 2c: scalars early,
direction later. Differences:

- Their emergence is near **layer 8 / 24** (~⅓). Ours is **layer 5 / 24**
  (~⅕). Still an early-to-mid jump, not a final-layer effect.
- Their direction R² softens toward the output. Ours keeps climbing slightly
  into the last layers (peak at L23 on the velocity val table). The CV curves
  are flatter after L5 than a dramatic late drop; this is a real difference
  from their Kubric plot and is expected to be softer on a flatter, more
  direction-dense dataset.
- The beige band at layers 7–9 on our plot is only a reference to their PEZ,
  not our `L*`.

### 2. Iterative nullspace at `L* = 5`

<table>
<tr>
<td width="50%" valign="top">

<strong>Paper — Figure 23 (Layer 8)</strong><br>
Direction accuracy vs orthogonal probe index shows a
<strong>sawtooth</strong>; speed R² falls smoothly.

<img src="artifacts/paper_figures/paper_fig23_sawtooth.png" alt="Paper Figure 23 sawtooth vs smooth"/>

</td>
<td width="50%" valign="top">

<strong>Ours — nullspace at L*=5</strong><br>
Direction R² stays jagged over many iterations; speed / accel
collapse in a few steps.

<img src="artifacts/figures/nullspace.png" alt="Our nullspace curves"/>

</td>
</tr>
</table>

Paper dimensionality across layers (Figure 22) — direction needs tens of
features at the PEZ and more near the output:

![Paper Figure 22 dimensionality](artifacts/paper_figures/paper_fig22_dimensionality.png)

**What we see.** Direction takes many iterations before the val probe dies.
Velocity-direction kept **41 probes → dimensionality 82**, with a jagged
R² curve the sawtooth detector flagged as present. Acceleration-direction: 31
probes (dim 62). Mixed direction: 23 probes (dim 46). Speed dies after **5**
kept probes; acceleration magnitude after **2**. Scalar MAE looks flat on the
shared right-hand axis because those errors are in m/s / m/s² and sit near 1–2,
while direction circular MAE climbs into the 60°+ range.

| Run | Kept probes | Dimensionality | Sawtooth? |
|---|---:|---:|---|
| Direction (velocity) — quoted | 41 | 82 | Yes |
| Direction (acceleration) | 31 | 62 | Yes |
| Direction (all) | 23 | 46 | Yes |
| Speed | 5 | 5 | No |
| Acceleration | 2 | 2 | No |

**Side-by-side read.** Same qualitative geometry as Figures 22–23: direction is
a large redundant subspace with paired structure; scalars are compact. Their
direction estimate at the PEZ was roughly **40–50 features**; ours is **82** at
L5 on 64 angle bins — higher, consistent with a denser angular label set and a
different layer. Their speed curve in Fig. 23 decays over ~25 probes under a
looser display threshold; ours hits the C.11 stop rule in 5. The important
comparison is **direction ≫ scalars**, not the exact count.

### 3. Multi-probe steering toward 90°

<table>
<tr>
<td width="50%" valign="top">

<strong>Paper — Figure 24 (Layer 8)</strong><br>
Held-out eval probe. MAE→90° falls from ~83° to
<strong>11.9°</strong> at 20 probes; MAE→truth rises.

<img src="artifacts/paper_figures/paper_fig24_steering.png" alt="Paper Figure 24 steering"/>

</td>
<td width="50%" valign="top">

<strong>Ours — steering at L*=5</strong><br>
Same protocol on take-home disks. MAE→90° falls from
~90° to <strong>6.3°</strong> at 20 probes (4.4° at full 41).

<img src="artifacts/figures/steering.png" alt="Our steering curves"/>

</td>
</tr>
</table>

Evaluation probe on unsteered test: R² = 0.71, circular MAE = 19.2° (weaker
than the paper’s 0.99 on 8 angles — expected with 64 bins). Steering-probe
sanity at full `n`: MSE to `(sin 90°, cos 90°)` ≈ 0 on the probes that built
the basis.

Held-out MAE on all test clips:

| Probes `n` | Ours MAE → 90° | Ours MAE → truth | Paper MAE → 90° (Fig. 24) |
|---:|---:|---:|---:|
| 0 | 90.3° | 19.2° | ~82.9° |
| 1 | 58.4° | 38.1° | ~78° |
| 5 | 11.4° | 82.3° | ~50° |
| 10 | 10.8° | 85.1° | ~25° |
| 20 | 6.3° | 88.9° | **11.9°** |
| 41 (full) | 4.4° | 88.6° | — |

Velocity-only and acceleration-only test subsets track the same pattern
(at `n = 20`: ~7.2° / ~5.5° to target).

**Side-by-side read.** Their causal signature is reproduced: as `n` grows,
error to the target falls and error to the true label rises. Single-probe
steering is weak; coordinated multi-probe steering works. Numerically we reach
the target *faster* than their layer-8 Kubric run (they needed ~20 probes to
get to ~12°; we are already at ~11° by `n = 5` and ~6° by `n = 20`). Likely
contributors: earlier `L*`, denser but linearly simpler disk videos, and a
large available basis (41 probes). The paper’s point still holds — you need a
**subspace**, not a single direction.

---

## Side-by-side verdict

| Paper claim | This run | Match? |
|---|---|---|
| Scalars readable early | Speed/accel R² ≥ 0.78 at L0 | Yes |
| Direction emerges mid-network | Jump at L5; weak at L0 | Yes (earlier than their L8) |
| Late-layer softening of physics | Mild / absent on val R² | Partial |
| Direction high-dim + sawtooth | Dim 82, sawtooth yes | Yes |
| Speed lower-dim, smooth | Dim 5, no sawtooth | Yes |
| Multi-probe steering needed; held-out eval | Target↓ / truth↑ with `n` | Yes |

Overall: the three Part 1 moves succeed on this dataset. The main quantitative
offsets from the paper are an earlier emergence layer, weaker late-layer drop,
and easier steering — all plausible under a simpler visual world and 64-way
direction labels.

---

## Limitations

- Mean-pool only. The paper also uses attentive-MLP and patch-level probes; we
  did not.
- No IntPhys / attention ablation (out of Part 1 scope).
- Nullspace stop rules follow Appendix C.11; speed’s R² at stop was still
  ~0.13 because the MAE baseline rule fired first.
- Steering evaluation probe is mediocre (R² 0.71). That makes the “MAE to
  truth” baseline higher than the paper’s, but the opposing target/truth curves
  remain the right causal check.
- Activations were extracted on an RTX 3090; probes are deterministic given the
  saved `.npz` and seed-0 splits.

---

## How to re-read the artifacts

```text
artifacts/figures/layer_curves.png
artifacts/figures/nullspace.png
artifacts/figures/steering.png
artifacts/paper_figures/paper_fig2c_polar.png      # Joseph et al. Fig 2c
artifacts/paper_figures/paper_fig2_layers.png      # Joseph et al. Fig 2
artifacts/paper_figures/paper_fig22_dimensionality.png
artifacts/paper_figures/paper_fig23_sawtooth.png    # Joseph et al. Fig 23
artifacts/paper_figures/paper_fig24_steering.png    # Joseph et al. Fig 24
artifacts/results/summary.json
```

Paper crops are for comparison in this writeup; the source PDF is
`Physics-Interp.pdf` (arXiv:2602.07050).

Reproduce from scratch (GPU): `bash scripts/run_all.sh` with data under
`data/` and deps from `requirements.txt`. Method lockfile:
[`PART1_PLAN.md`](PART1_PLAN.md). Interactive method sketch:
[`explainer.html`](explainer.html).
