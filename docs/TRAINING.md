# Training protocol

## Scope and provenance

My MTP studies resource-efficient text-to-image generation through cascaded masked image modeling and GILoRaT-based low-rank training. Both the staged architecture and this training approach were part of the MTP. The protocol below describes the recorded local runs used for the additional dense-versus-grown evaluation in this repository. Source hashes identify the measured implementation, whose core modules are preserved byte for byte. Implementation provenance is documented in [reproduction notes](REPRODUCIBILITY.md).

## Dataset and frozen components

The pinned Nouns dataset contains 49,859 captioned pixel-art images. Original RGB image hashes define deterministic, image-grouped splits: **44,894 training, 2,467 validation and 2,498 test**. Duplicate image content stays in one split. The dataset revision and downloaded component hashes are recorded under `code/results/assets.json`.

Images are converted to RGB and bilinearly resized to 256×256. The frozen aMUSEd-256 VQ component supplies a 16×16 token map from an 8,192-entry, 64-dimensional codebook. The frozen training text encoder supplies 77×768 penultimate hidden states. An independent CLIP ViT-B/32 model is used for evaluation. The trainable B/C modules and EfficientNet image encoder start from scratch; the public aMUSEd transformer weights are not a pretrained backbone for these modules.

## Common optimization

Training uses one local RTX 5060 Ti with 16GB VRAM, BF16 forward autocast, fused AdamW, weight decay 0.01, microbatch 4 and accumulation 8. Learning rates are 1e-4 for joint B/C and 2e-5 for B fine-tuning, with global gradient clipping at 1.0 and no learning-rate schedule. Every 500 optimizer updates, a fixed 512-row validation prefix supplies checkpoint selection and the rank controller's signal. The effective full batch contains 32 images; the last fine-tuning update of each exact dataset pass contains 30.

The source configuration, private random streams, data order, frozen components and phase dependencies are retained. Configuration duration fields for the extended experiment are safety ceilings rather than evidence of convergence.

## Joint Stage B

A scratch EfficientNet-B0 feature extractor produces a 192×7×7 image-feature grid. It learns jointly with a masked-token transformer: width 768, 12 blocks and 12 attention heads. The image tokens are downsampled from 16×16 to 8×8; conditioning includes the 49 image-feature tokens, 77 text tokens and a mask-ratio token. The output head predicts VQ token logits at 16×16 positions.

Training uses a random cosine-derived masking ratio with at least 5% masked tokens. Cross-entropy supervises masked token targets. Validation fixes the mask ratio at 75% and uses an independent generator. A zero-feature ablation checks the dependence on learned image features.

The dense control completes 20,000 optimizer updates. The extended low-rank run resumes its earlier 20,000-update B checkpoint with the complete optimizer/RNG state. It grows naturally through ranks 2, 4, 6 and 8, then ends at 387,453 updates. The selected checkpoint is **385,500**, with fixed-mask CE **1.6631238330**, and all 48 adapted B layers at rank 8. This phase ended manually before a verified terminal plateau.

## Rebuild real feature targets

The selected B image encoder creates a new FP16 feature cache of shape 49,859×192×7×7. Channel statistics use training rows only. The control and reported grown run use raw features (`standardize=false`). Extending B changes C's targets, so the downstream C model starts again from its original initialization rather than reusing an optimizer trained against the old encoder.

## Stage C: feature diffusion

A text-conditioned diffusion transformer predicts noise over the 192×7×7 feature grid, using width 1,152, 16 blocks and 16 heads. The schedule has 1,000 cosine timesteps. Training samples a timestep and Gaussian noise; the objective is inverse-noise-variance-weighted squared error capped at 100. Reported validation MSE is unweighted and uses fixed noise. These quantities have different scales.

The dense control completes 10,000 C updates. Extended C grows through ranks 2, 4, 6, 8 and 10, then ends at 229,472 updates. Selected checkpoint **228,500** has fixed-noise MSE **0.5306405896** and all 64 adapted C layers at rank 10. The phase was stopped manually before a verified terminal plateau. Raw C MSE depends on its learned encoder's feature targets and does not independently establish image-quality superiority across methods.

## Generated features and B fine-tuning

Selected C generates a new feature tensor for every caption using 50 DDIM steps. It receives text and seeded noise, without access to a clean image feature at generation time. Its row cursor and generator state are durable.

Fine-tuning starts from selected extended B, keeping the grown layer shapes. EfficientNet, C, text encoders and VQ are frozen. B adapts to generated feature conditioning at learning rate 2e-5, with no feature mixture or conditioning dropout. A new Adam optimizer starts at this original phase boundary; interruptions within the phase restore that optimizer exactly.

Fine-tuning samples exact shuffled training passes without replacement. With 44,894 rows, each pass has 1,403 updates, including a final 30-row update weighted by its true sample count. The dense control performs ten complete passes (14,030 updates). The extended run keeps B's inherited rank 8 fixed while its weights remain trainable. It ends through the native graceful shutdown at **91,059** updates, selecting **89,500** with CE **2.1844024385**. The final cursor represents 2,913,760 exposures, 64 completed passes and 40,544 rows into the next pass. No terminal plateau is claimed.

## Adaptive factorization

Transformer-block linear layers use trainable scratch factors; dense boundary, conditioning and convolutional layers remain. The starting rank is 2 and growth adds 2. Four non-improving validation checks trigger the initial assessment; patience subsequently scales as twice the maximum adapted rank. The generative implementation uses the original gradient-importance gate, zero-extension and Adam-state migration. The configured orthogonality weight is zero. GILoRaT is credited briefly in the README; the recorded generative settings are recorded implementation settings.

## Selection, sampling and evaluation

Validation selects the lowest observed B/FT CE and lowest C fixed-noise MSE. Final quality uses selected fine-tuned B and C on the first 2,048 validation prompts, with 50 DDIM steps, 12 masked refinement steps and guidance 1. VQ decoding yields RGB256 images. The untouched test is not used for tuning or reporting. The animation's last displayed fine-tuning step, 91,000, is different from selected 89,500 and final durable 91,059.

## Checkpointing and recovery

Native atomic checkpoint writes flush and sync before publication. Distinct latest, previous and validation-selected best states retain model weights, Adam moments/update counters, global/private RNG streams, data permutation/epoch/cursor and rank trackers. Independent audits replayed the shuffled cursor and next batch and verified frozen-encoder/source lineage. The local campaign also retained a boot-resume service; that host-specific deployment remains in the local archive. This repository installs no service and starts no job automatically. A physical power-cut test was not performed.

## Observed versus unproven

Actual B/C rank expansion is verified. All three learned phases were subsequently ended explicitly before terminal plateau. The current result is one extended seed, with substantially more training updates than the dense control; it does not establish equal-budget training efficiency, a training speedup, complete convergence or replicated quality superiority. Parameter efficiency and the scoped generation-memory reduction are the measured resource findings.
