# Part 1 handoff: reproduce the V-JEPA physics probes

Implement Part 1 of `README.md` only. The method is Appendix B, C.11, and C.12 of Joseph et al., *Interpreting Physics in Video World Models* (arXiv:2602.07050), applied to the clips already in `data/`. Qualitative agreement with the paper is the goal. Do not chase the paper’s exact R².

Do not implement IntPhys, attention ablations, patch-level probes, neuron tuning, or Part 2 spline steering.

Leave `data/` unchanged. Write every derived file under `artifacts/`.

## Definition of done

Produce these three figures, each annotated with the split, the layer index convention, and the pooling rule:

1. `artifacts/figures/layer_curves.png` — validation-chosen scores plus a 5-fold curve of R² versus layer for direction, speed, and acceleration. Direction also has a circular-MAE panel. Direction is shown three ways: all clips, `motion == velocity` only, and `motion == acceleration` only.
2. `artifacts/figures/nullspace.png` — at the chosen layer, validation score versus probe iteration for direction and for both scalars.
3. `artifacts/figures/steering.png` — on the locked test set, circular MAE to the target angle and circular MAE to the true angle, versus number of steering probes. Include the velocity-only and acceleration-only breakdown for direction.

Also write `artifacts/results/summary.json` with the chosen layer, the stopping iteration, the steering table, and every deviation from this plan.

## Model

Frozen encoder. Do not train it and do not run the predictor.

- Checkpoint: `facebook/vjepa2-vitl-fpc64-256`
- Class: `vjepa2_vit_large` in [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2). Loading through `transformers.AutoModel` is the intended path.
- 24 blocks, width 1024, 16 heads, patch 16, tubelet 2, crop 256.
- `frames_per_clip=64` is a pretraining setting. It does not change the encoder. Feed the real 16 frames. Do not repeat frames to make 64.

```python
from transformers import AutoModel, AutoVideoProcessor
import torch

repo = "facebook/vjepa2-vitl-fpc64-256"
processor = AutoVideoProcessor.from_pretrained(repo)
model = AutoModel.from_pretrained(repo, attn_implementation="sdpa").eval()

with torch.inference_mode():
    out = model(
        pixel_values_videos=pixels,  # (B, 16, 3, 256, 256), processor-normalized
        output_hidden_states=True,
        skip_predictor=True,
    )
```

`out.hidden_states[0]` is the patch embedding. `out.hidden_states[1 + ell]` is the residual stream after block `ell`, for `ell` in `0 .. 23`. The paper’s layer index is `ell`, not the tuple index. There are 25 tensors in `hidden_states`. Keep layers 0 through 23 only.

Token layout for a 16-frame 256×256 clip: 8 tubelets × 16 × 16 patches = 2048 tokens, each of size 1024. Mean-pool over the token axis. That pooled vector is the paper’s primary probe input (Appendix B: “linear probes on mean-pooled space–time patches”).

Before extracting the full set, run one clip and assert:

- processed video has 16 frames, not 64;
- each block output has shape `(B, 2048, 1024)`;
- `len(hidden_states) == 25`.

If the processor resamples time, bypass that and pass the 16 decoded frames in order. A duplicated-frame clip destroys speed and acceleration.

## Hardware

Extraction is the only heavy step. Probes are small linear models.

- On a rented GPU, use CUDA, fp16 or bf16 for the encoder forward, and float32 for the saved pooled vectors after casting. Batch 4–8 on a 12 GB card, 8–16 on a 24 GB card.
- On the M1 Pro, use `device="mps"`, batch size 1, float32. Expect a few hours.
- Do not store token-level activations. Pooled float32 for all clips and 24 layers is about 450 MB.

## Data contract

Resolve paths in each manifest relative to that manifest’s directory (`data/<name>/`).

| Dataset | Clips | Probe target | Notes |
|---|---:|---|---|
| `direction` | 1500 | `theta_degrees` | No `primary_label` and no `magnitude` field. 64 angles from 0° inclusive to 360° exclusive, step 5.625°. 750 clips have `motion == "velocity"`, 750 have `motion == "acceleration"`. |
| `speed` | 1536 | `magnitude`, which equals `speed_mps` | 64 speeds from 0.25 to 4.0 m/s. `motion` is always `velocity`. `theta_degrees` also varies; do not use it as the target. |
| `acceleration` | 1536 | `magnitude`, which equals `acceleration_mps2` | 64 accelerations from 0.25 to 10.0 m/s². `speed_mps` is always 0. `theta_degrees` varies; do not use it as the target. |

Every clip is 256×256, 16 frames, 24 fps. Decode exactly 16 frames in temporal order. Drop the audio. RGB.

## Labels and metrics

Direction target for training is the pair `(sin θ, cos θ)` with `θ` in radians. Loss is mean squared error. Do not train on the wrapped angle and do not use an angular loss. Appendix C.11 is explicit: circular regression, MSE.

Scores for direction:

- `R²` from `sklearn.metrics.r2_score(..., multioutput="uniform_average")` on the sine/cosine pair.
- Circular MAE in degrees. `pred = degrees(atan2(sin_hat, cos_hat)) mod 360`. Error is `(pred - true + 180) mod 360 - 180`. MAE is the mean of the absolute wrapped error. Chance on a uniform circle is 90°.

Scores for speed and acceleration:

- `R²` and MAE on the raw scalar in m/s or m/s².
- Random baseline MAE: predict the training-split mean of the scalar, measure MAE on the split being scored.

Do not z-score activations. Block outputs are already layer-normalized. Do not z-score sine/cosine. Do not z-score the scalars either; report MAE in physical units. Weight decay is what the appendix uses for regularization.

## Splits

One split per dataset. Seed 0. Write the clip ids to `artifacts/splits/<dataset>.json`.

Stratify so every label bin appears in train, val, and test:

- Direction: bin is `(theta_degrees, motion)`. About 12 clips per bin.
- Speed and acceleration: bin is `magnitude`. 24 clips per bin.

Within each bin, shuffle with `numpy.random.default_rng(0)` and assign:

- 20% test, at least 1 clip when the bin has at least 3;
- 10% val, at least 1 clip when the bin has at least 5;
- the rest train.

Rough sizes: direction 1050 / 150 / 300, speed and acceleration 1075 / 154 / 307. Exact counts depend on rounding inside bins. The test ids are locked before any probe, QR, or steering solve.

Group id for cross-validation, not for the locked split: quantize `start_position_xy_m` to 0.5 m,

```python
def start_group(xy):
    x, y = xy
    return f"{round(x * 2) / 2:.1f}_{round(y * 2) / 2:.1f}"
```

Direction has about 81 such groups. Speed and acceleration have about 25.

Layer curves use 5-fold `GroupKFold` on train+val only, grouped by `start_group`. Test never enters a fold. If `GroupKFold` cannot build 5 folds, fall back to `StratifiedKFold` on the label bin and record that fallback in `summary.json`.

## Probe module

Use one implementation for every linear readout.

```python
probe = torch.nn.Linear(1024, n_out)  # n_out is 2 for direction, 1 for scalars
# Adam, MSE loss
```

`probe.weight` has shape `(n_out, 1024)`. The bias is trained and is used at prediction time. The bias is not part of the subspace that gets projected out.

Two training recipes. Do not mix them.

**Appendix B, layer-wise curves only.** Sweep all 20 pairs:

- learning rate in `{1e-4, 3e-4, 1e-3, 3e-3, 5e-3}`
- weight decay in `{0.01, 0.1, 0.4, 0.8}`

Train 100 epochs, full batch (N is about 10³). Pick the pair with the best score on that fold’s held-out part: R² for every variable. Report mean ± std of the chosen score across the 5 folds.

Then, for layer selection only, refit the winning hyperparameters for each layer on the entire train split and score on val. Downstream steps use this val table, not the test table.

**Appendix C.11, nullspace and the steering basis.** Fixed hyperparameters, no sweep:

- Adam, learning rate `1e-3`, weight decay `1e-4`
- 100 epochs for direction, 50 epochs for speed and acceleration
- full batch, MSE
- fit on train only, score on val

## Step 1. Extract activations

Script: `scripts/extract_activations.py`.

For each dataset, save `artifacts/activations/<dataset>.npz` with:

- `ids`: int clip ids in manifest order
- `layers`: float32 array, shape `(N, 24, 1024)`, mean-pooled block outputs
- `targets`: float32. Direction stores `theta_degrees`. Speed and acceleration store `magnitude`.
- `motion`: string array
- `start_xy`: float32, shape `(N, 2)`

Pooling is the mean over all 2048 tokens. Record that string in the npz as `pooling = "mean_spacetime"`.

Run the one-clip assert above before the full loop. Save checkpoints every few hundred clips so a rented GPU can resume.

## Step 2. Layer-wise probes

Script: `scripts/probe_layers.py`.

For every dataset and every layer `0 .. 23`, run the Appendix B sweep inside each GroupKFold fold of train+val. Features are `layers[:, ell, :]`.

Direction folds are fit three times: all non-test clips, velocity clips only, acceleration clips only. Use the same clip split either way; just filter. Speed and acceleration are fit once each, on their own datasets, predicting `magnitude`.

Plots:

- R² versus layer, one line per series, with the 5-fold std as a band.
- A second panel for error: circular MAE for direction, MAE in physical units for the scalars.

Layer choice, using the train-refit / val scores, not the test scores:

- Let `r0` be direction R² at layer 0 on the **velocity** subset.
- Let `rmax` be the maximum direction R² on that subset, at layer `L_peak`.
- Let `L_emerg` be the smallest layer whose direction R² is at least `(r0 + rmax) / 2`.
- Choose `L_emerg` when val R² for speed and for acceleration at that layer are both at least 0.5. Otherwise choose `L_peak`.
- Call the chosen layer `L*`. Write `r0`, `rmax`, `L_peak`, `L_emerg`, and `L*` into `summary.json`.

The paper’s `L*` on V-JEPA 2-L was layer 8. If this dataset moves the rise, follow the rule above.

What the curve is supposed to look like: speed and acceleration are already readable in the early layers; direction on the velocity subset is weaker early and rises near the first third of the network; all three are a bit weaker at layer 23 than at their peak. The mixed direction set may blur the rise because half of those clips are accelerating. That is why the velocity-only curve is the one used to pick `L*`.

### Pooling ablation, only if the velocity curve has no rise

Run this only when velocity direction R² at layer 0 is above 0.5 and `L_emerg` is 0 or 1. Re-extract with `pooling = "temporal_last"`: reshape tokens to `(8, 16, 16, 1024)` in the model’s actual token order, mean-pool the spatial grid of the last tubelet, and repeat Step 2. An independent reproduction (Leesangoh/PEZ-Reproduction) only recovered a layer-8 onset with last-tubelet pooling. Keep both npz files. State which pooling the rest of the pipeline uses. Do not switch silently.

Confirm the token order with a one-clip hook before trusting the reshape. If the layout is not `(time, height, width)`, stop and record the real layout rather than guessing.

## Step 3. Iterative nullspace

Script: `scripts/nullspace.py`. Appendix C.11.

Run at layer `L*` only, in the same feature space as the chosen pooling. Three runs: direction (velocity clips only), speed, acceleration. Also run direction on all direction clips and on acceleration clips, and plot those, but the dimensionality quoted in the summary is the velocity run. That matches the paper’s direction dataset, which was constant-velocity.

Algorithm. `X_train` and `X_val` start as the layer-`L*` activations. `y` is sine/cosine or the scalar.

```text
probes = []
for k in 0, 1, 2, ...:
    fit probe k on (X_train, y_train) with the C.11 recipe
    score it on (X_val, y_val)
    record R² and MAE
    if stop(val scores):
        break
    Q, _ = qr(probe.weight.T, mode="reduced")   # Q is (1024, n_out)
    drop columns of Q whose corresponding |diag(R)| < 1e-6
    X_train = X_train - X_train @ Q @ Q.T
    X_val   = X_val   - X_val   @ Q @ Q.T
    append probe k to probes
```

Project val with the same Q. The recorded score for probe k is the score **before** that probe’s direction is removed. The next probe is both trained and scored in the residual.

The probe that crosses the stop threshold is not appended and is not used for steering.

Stop rules, evaluated on val, from C.11 (these differ from the looser cutoffs in the Figure 22 caption; use C.11):

- Direction: `R² < 0.1` or circular MAE `> 80°`.
- Speed and acceleration: `R² < 0.05` or MAE `> 0.9 ×` the val MAE of predicting the train-set mean.

Cap iterations at 80 for direction and 40 for scalars so a bug cannot loop. If the cap is hit, write that the representation was still linearly readable.

Dimensionality: `2 * len(probes)` for direction, `len(probes)` for a scalar. Each direction probe contributes a 2-row weight.

Save `artifacts/probes/<dataset>_L<L*>_nullspace.pt` with each kept probe’s `weight`, `bias`, and val metrics.

Plot val R² and val MAE against iteration. Direction’s paper curve is a sawtooth: a drop and a partial recovery in pairs, read as sine/cosine partners. Speed’s curve falls smoothly. Report whether that pair structure appears; its absence is a result, not a failed run.

## Step 4. Steering

Script: `scripts/steer.py`. Appendix C.12.

Steer direction only, at layer `L*`, using the kept velocity-subset probes from Step 3. Those probes were fit on train and never saw test.

Build a fresh evaluation probe on the **unsteered** test activations only, with the C.11 direction recipe (Adam `1e-3`, weight decay `1e-4`, 100 epochs, sine/cosine MSE). It does not see the steering probes, the train set, or any steered vector. Score it once on unsteered test before any intervention. The paper’s eval probe had R² 0.99 on 8 angles; 64 angles on a flat disk will be lower. Record the number and move on.

Target angle is `θ* = 90°` for every test clip, the same protocol as the paper.

For a chosen prefix of `n` kept probes, with weights `W_i` of shape `(2, 1024)` and biases `b_i`:

```python
def steer(x, probes_n, theta_star_rad):
    # V spans the row space of the n probe weights. Reduced QR.
    M = torch.cat([p.weight.T for p in probes_n], dim=1)  # (1024, 2n)
    V, _ = torch.linalg.qr(M, mode="reduced")
    x_perp = x - V @ (V.T @ x)
    t = torch.tensor([sin(theta_star_rad), cos(theta_star_rad)], device=x.device)
    rows, rhs = [], []
    for p in probes_n:
        rows.append(p.weight @ V)                         # (2, rank)
        rhs.append(t - p.bias - p.weight @ x_perp)        # (2,)
    A = torch.cat(rows, dim=0)
    b = torch.cat(rhs, dim=0)
    c_star = torch.linalg.lstsq(A, b).solution
    return V @ c_star + x_perp
```

This is the paper’s update: `c = Vᵀ x`, `x_perp = x - V c`, choose `c*` by least squares so every included probe outputs `(sin θ*, cos θ*)`, then `x* = V c* + x_perp`.

Evaluate `n` in `{1, 2, 5, 10, 15, 20}` and the full kept set, skipping values past the number of kept probes. For each `n`:

- steer every test activation;
- read the steered vector with the frozen evaluation probe;
- record circular MAE to 90° and circular MAE to the true angle;
- do this for all test clips, for `motion == velocity`, and for `motion == acceleration`.

The causal signature from the paper: MAE to the target falls as `n` grows, and MAE to the true angle rises. At their layer 8, 1–5 probes stayed above 50° from the target, and 20 probes reached about 12°, from a baseline near 83°. A single probe doing nothing is the expected small-`n` result, not a bug.

Sanity check on one test clip at the full `n`: after steering, each steering probe’s own `(sin, cos)` prediction should sit near `(1, 0)`. If it does not, the least-squares solve is wrong. The evaluation probe is a different readout and is allowed to miss the target.

## Step 5. Write the results

`artifacts/results/summary.json` must contain:

- commit-free config: repo id, pooling, `L*`, split counts, seed;
- per-layer val R² for all three variables and for the two direction subsets;
- nullspace length and final val metric for each run;
- steering table: `n`, MAE to target, MAE to truth, by subset;
- whether the sawtooth was visible;
- the one-clip shape asserts;
- any fallback (stratified CV, temporal-last pooling, iteration cap).

Figures must state that test clips were not used to pick `L*`, the hyperparameters, or the steering basis.

## Files to add

```text
scripts/extract_activations.py
scripts/probe_layers.py
scripts/nullspace.py
scripts/steer.py
scripts/metrics.py          # circular MAE, R², random-baseline MAE
requirements.txt            # torch, transformers, numpy, scikit-learn, matplotlib, tqdm
artifacts/splits/
artifacts/activations/
artifacts/probes/
artifacts/figures/
artifacts/results/summary.json
```

Add `artifacts/` to `.gitignore` if a gitignore is created. Do not commit the npz files.

## Checks before calling the run finished

1. One clip: 16 frames in, 2048 tokens, 24 block outputs.
2. Test ids are disjoint from every fit call. Grep the scripts for the test array and confirm it is only passed to the evaluation probe and to the steering apply step.
3. Direction loss is MSE on sine and cosine. A unit test: a perfect prediction has circular MAE 0; predictions rotated by 180° have circular MAE 180; layer-shuffled predictions land near 90°.
4. Nullspace unit test in 4-D: a probe with weight along axis 0, after projection, leaves that coordinate at 0 and leaves an orthogonal coordinate unchanged.
5. Steering unit test: two orthogonal probes in a toy 4-D space, after `steer`, both probes predict the target sine/cosine on the steered vector, and the component orthogonal to both weights is unchanged.
6. Layer curve, nullspace curve, and steering curve exist and are labeled with units.

## What not to copy from elsewhere

- Leesangoh/PEZ-Reproduction is evidence that pooling can hide the emergence curve. It is not the method. Do not regenerate Kubric, and do not adopt their `angle` regression target. This plan uses sine/cosine, as Appendix C.11 and `DATA.md` specify.
- `shauli-ravfogel/nullspace_projection` is the same geometry for classifiers. Do not port `debias.py`. The projector here is the reduced QR of the regression weight, given in C.11.
- There is no author-released code for the paper.
