# V-JEPA physics take-home

This take-home is about understanding how V-JEPA represents physical variables
in its latent space. The focus is on three properties of simple moving objects:
direction, speed, and acceleration.

The investigation combines a small-scale reproduction of results from Sonia
Joseph et al.'s *Interpreting Physics in Video World Models* with an open-ended
extension based on the spline-steering method introduced by Goodfire in
*Manifold Steering Reveals the Shared Geometry of Neural Network Representation
and Behavior*.

## Part 1: reproduce the physics-representation results

Use the supplied videos to reproduce the main experimental progression from
the physics paper:

1. **Layer-wise probing.** Train and evaluate probes at every V-JEPA transformer
   layer for direction, speed, and acceleration. Plot probe performance over
   layers and use the curves to identify where each variable becomes available.
2. **Iterative nullspace probing.** Select a suitable layer and repeatedly fit a
   probe, remove its readout subspace, and fit another probe. Use the resulting
   performance curve to investigate the dimensionality and redundancy with
   which each physical variable is represented.
3. **Multi-probe subspace steering.** Reproduce the paper's steering experiment
   by intervening in the subspace defined by multiple probes. Evaluate the
   intervention on held-out data that was not used to construct the steering
   subspace.

The supplied dataset is deliberately smaller and simpler than the datasets in
the paper. The aim is to reproduce the methodology and qualitative findings,
not the paper's exact numerical results.

## Part 2: extend the investigation with spline steering

Apply the manifold-steering ideas from the Goodfire paper to V-JEPA's
representations of physical variables. This part is intentionally open-ended.
At minimum, attempt to learn manifolds or splines for:

- speed;
- acceleration;
- direction.

Decide how the splines should be constructed, visualized, and evaluated. Think
carefully about the circular structure of direction and about what constitutes
a meaningful held-out steering evaluation. Compare spline steering with the
multi-probe subspace method from Part 1, including strengths, limitations, and
failure cases.

## Provided data

The repository contains three synthetic video datasets:

```text
data/
├── direction/
│   ├── manifest.jsonl
│   └── videos/scene_XXXX/{video.mp4,metadata.json}
├── speed/
│   ├── manifest.jsonl
│   └── videos/scene_XXXX/{video.mp4,metadata.json}
└── acceleration/
    ├── manifest.jsonl
    └── videos/scene_XXXX/{video.mp4,metadata.json}
```

Each manifest contains paths to a 16-frame video and its metadata. The metadata
contains the physical labels and rendering context. See [DATA.md](DATA.md) for
the precise fields, ranges, and loading convention.

Use the pretrained **V-JEPA 2 ViT-L/16, 256-resolution** encoder available as
[`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256).
The corresponding model implementation is `vjepa2_vit_large` in Meta's
[official V-JEPA 2 GitHub repository](https://github.com/facebookresearch/vjepa2).

Keep the encoder frozen, document how intermediate representations are
extracted and pooled, and clearly separate data used to fit probes or manifolds
from data used for evaluation. Do not modify the supplied data; store
activations, models, figures, and other derived artifacts elsewhere.

## Deliverable

The final output is a presentation of the findings, aimed at approximately 15
minutes. The presentation will serve as the basis for an open discussion, so
the duration is general guidance rather than a strict limit. Present the main
methods, results, interpretations, comparisons, and limitations clearly.

You may use any IDE and coding tools you prefer, including AI coding
assistants. You remain responsible for understanding the implementation,
experimental choices, and conclusions.

## Relevant papers

- Sonia Joseph et al., [*Interpreting Physics in Video World Models*](https://arxiv.org/abs/2602.07050).
- Goodfire / Wurgaft et al., [*Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior*](https://arxiv.org/abs/2605.05115).
