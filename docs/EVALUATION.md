# Evaluation protocol and additional diagnostics

## Primary comparison

I compare both arms using training seed 17 and the same 2,048 validation captions and reference images. RGB256 preprocessing, guidance 1, 50 DDIM steps and 12 masked refinement steps are identical. The grown arm receives longer training, and its final native sampling noise is not paired with dense. I treat the results as descriptive rather than a controlled estimate of the causal benefit of rank growth.

FID and KID use clean-fid 0.1.35 preprocessing and its pinned Inception detector. FID is computed using the equivalent PSD/SVD formulation retained in `code/evaluate.py`, handling singular 2,048-sample covariance matrices. Independent replay from the saved vectors matches the native FID. KID is the recorded native estimate; recomputing a new randomly subsampled KID estimate need not be bit-identical without its original subset draws.

CLIP cosine uses the independent pinned ViT-B/32 evaluator, not the cached training text hidden states. Saved normalized image/text embeddings support alignment and caption retrieval replay. FID/KID depend on feature representation and sample count; I do not directly overlay the unspecified FID protocol from my thesis.

## Additional metrics

I add PRDC as a post-hoc diagnosis of fidelity and diversity in saved Inception space. I use k=5 as the primary neighborhood size and retain k=3 and 10 for sensitivity. I checked the optimized calculation against the official implementation pinned at `e320c1d2811d33081361a08f595b43830b78641c`, including a self-manifold fixture. Recall must be read with precision, density and coverage rather than interpreted alone.

| Diagnostic, k=5 | Dense | Grown |
|---|---:|---:|
| Precision | 0.30322 | 0.14844 |
| Recall | 0.78271 | 0.67334 |
| Density | 0.11924 | 0.04453 |
| Coverage | 0.32422 | 0.13770 |
| Caption retrieval R@1 | 4.980% | 3.857% |
| Caption retrieval R@5 | 12.695% | 11.621% |
| Caption retrieval R@10 | 17.578% | 16.553% |

Caption retrieval ranks all 2,048 saved captions by CLIP similarity. Identical caption strings count as a match; semantically equivalent but different strings may not. The recorded alignment margin compares matched captions with a shuffled-caption baseline. These measures complement FID/KID and do not establish a universal measure of semantic correctness.

## Resource accounting

I count selected fine-tuned B plus C, including their dense boundary layers, as learned generator parameters. Shared frozen VQ/text parameters and the image encoder unused by text-only inference are excluded. Joint-B training counts include its trainable image encoder. Counts are scalar parameters, not checkpoint file size or nonzero-weight counts.

Generation memory is the maximum PyTorch allocated/reserved memory during complete native generation. CUDA context/driver overhead and total-device capacity are separate. Recorded dense training-process peaks are available; I report complete peaks spanning the grown training stages as unavailable (N/A). I display actual completed training updates separately to expose unequal compute budgets. Generation process walls include durable image writes and are not a controlled speed benchmark.

## Qualitative protocol

I show exactly the first eight validation rows in the comparison grid, without favorable selection. Reference bytes match across arms. These native final draws are separate from fixed-seed progression animations. Animation seed banks preserve the same eight captions and feature/token seeds across optimizer keyframes. I use compact, lossy GIFs for display and keep the native PNG masters unchanged in my local archive.

## Evidence

`evidence/dense-17` and `evidence/grown-17` retain native summaries, configurations, complete journals, both full 2,048-row embedding arrays and the displayed PNGs/references. `results/comparison.json` records every figure value. `results/provenance.json` binds copied source files and native source commit hashes. I checked source/sample checksums, stable checkpoint identities and FID/CLIP replay before reporting. My analysis scripts do not score test examples.
