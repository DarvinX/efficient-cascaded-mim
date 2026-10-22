# Efficient Cascaded Masked Image Modeling

**MTP · Resource-efficient text-to-image generation**

![Project overview](assets/overview.svg)

This repository accompanies Tamal Hansda's MTP, **Efficient Text to Image Generation using Cascaded Masked Image Modeling**. It provides a reconstructed implementation of the thesis pipeline and a local-GPU extension that evaluates the parameter and memory trade-offs of adaptive low-rank training. The original training implementation was unavailable; reconstruction decisions and later measurements are identified explicitly.

The resource-efficient generator uses **33.31M learned parameters**, compared with **368.92M** for the dense control, and records **1.63 GiB peak generation allocation**, compared with **4.30 GiB**. Image quality remains a trade-off: dense achieves lower FID and KID. These are measurements from the same 2,048 validation prompts and training seed 17, with different training budgets.

[Training protocol](docs/TRAINING.md) · [Evaluation details](docs/EVALUATION.md) · [Reproduction](docs/REPRODUCIBILITY.md) · [Machine-readable results](results/comparison.json)

## Resource efficiency

| Quantity | Dense | Grown low-rank |
|---|---:|---:|
| Learned generator parameters, B + C | 368.92M | **33.31M** |
| Peak allocated memory during generation | 4.30 GiB | **1.63 GiB** |
| Peak reserved memory during generation | 4.86 GiB | **2.13 GiB** |
| FID ↓ | **66.15** | 95.10 |
| KID ↓ | **0.06986** | 0.10745 |
| CLIP cosine ↑ | **0.34476** | 0.34264 |

The grown generator has **90.97% fewer learned parameters** and **62.08% lower measured generation allocation**. Learned parameter counts include selected fine-tuned B and C, with their dense boundary layers; shared frozen VQ/text components are excluded. GPU allocation measurements cover the complete native generation process, including the frozen components. They are PyTorch allocator measurements rather than total device memory.

![Resources](figures/02_resources.png)

The grown run trained for more updates. Its whole-training peak memory was not retained after the requested phase stops and is reported as unavailable. These results establish a parameter/memory/quality trade-off, without a matched-budget training-speed claim.

## Quality versus parameter count

![Quality versus parameters](figures/01_metrics_vs_parameters.png)

The comparison uses one dense and one grown seed-17 cascade. All metrics use the same 2,048 held-out validation rows, RGB256 preprocessing, 50 DDIM steps, 12 masked refinement steps and guidance 1. The reserved test split is unscored. The thesis's reported FID uses an unspecified protocol and is not treated as numerically equivalent.

## Qualitative results

![Reference, dense and grown samples](figures/03_sample_comparison.png)

The grid shows the first eight validation prompts in their original order, with no favorable sample selection. Captions and references match between arms; native final sampling noise is not paired. These images are separate from the fixed-seed training progression below.

## Training progression

![Fine-tuning progression](animations/finetuning.gif)

The same eight prompts and noise seeds are retained across training. This compact preview spans fine-tuning steps 0–91,000; metrics use the selected checkpoint at 89,500. Original PNG masters and full recovery states remain preserved in the local research archive.

<details>
<summary>Joint B and feature-diffusion C progression</summary>

| Joint B: neutral-feature text probe | C: generated feature conditioning |
|---|---|
| ![Joint B progression](animations/stage-previews/b.gif) | ![C progression](animations/stage-previews/c.gif) |

Each preview shows 32 uniform native keyframes with the original eight prompts and feature/token seeds. Last displayed steps are 387,000 for B and 229,000 for C; optimizer shutdown steps are documented separately. Playback is illustrative, with no image interpolation or new model inference.

</details>

<details>
<summary>Learning curves and rank history</summary>

![Learning and rank history](figures/learning-and-rank.png)

Joint B grew naturally from rank 2 to 8; C grew from 2 to 10. Fine-tuning kept B's rank 8 fixed under the later experimental instruction. Each phase was ended explicitly before a terminal plateau, so the run is not described as fully converged.

</details>

## Method

A frozen VQ model maps images to discrete tokens. Joint Stage B learns image features and masked image tokens; Stage C learns text-conditioned diffusion over the image-feature grid. B is then fine-tuned against generated C features. Text-only inference follows **caption → feature diffusion C → masked-token B → VQ decoder → image**. See the [full training account](docs/TRAINING.md) for architecture, data splits, losses, optimization, rank growth, checkpointing and exact observed stage boundaries.

The low-rank extension draws on **GILoRaT**, related work coauthored by Tamal Hansda: [preprint](https://doi.org/10.13140/RG.2.2.19338.17608), [code](https://github.com/DarvinX/GILoRaT). Its validation-driven factorization is adapted here to the generative stages; this extension is evaluated separately from the original classification experiments.

## Additional evaluation

Beyond the thesis reporting, this release retains FID/KID and independent CLIP embeddings and adds PRDC precision/recall/density/coverage, caption retrieval at 1/5/10, alignment against shuffled captions, parameter accounting and scoped GPU-memory measurements. [Evaluation details](docs/EVALUATION.md) explain the protocols and limitations. CPU-only numerical replay uses the saved validation features; it does not require training or test evaluation.

## Repository structure

```text
code/          Measured training implementation, configurations and asset setup
analysis/      CPU-only figure generation and saved-evidence verification
assets/        Project overview
animations/    Compact fixed-seed training animations and provenance
figures/       Research figures, with PNG and vector SVG exports
results/       Comparison table, learning history and provenance
evidence/       native summaries, journals, vectors and samples
docs/           training, evaluation and reproduction account
```

Large datasets, pretrained weights and optimizer checkpoints are kept outside Git. Pinned public asset revisions and selected checkpoint SHA256 values are documented; the local source archive retains all original images and resumable states. Nothing in this repository automatically launches training or installs the recovery service.
