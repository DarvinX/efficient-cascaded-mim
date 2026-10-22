# Evaluation protocol and additional diagnostics

## Primary comparison

Both reported arms use training seed 17 and the same 2,048 validation captions and reference images. RGB256 preprocessing, guidance 1, 50 DDIM steps and 12 masked refinement steps are identical. The grown arm receives longer training, and its final native sampling noise is not paired with dense. Results are descriptive rather than a controlled estimate of the causal benefit of rank growth.

FID and KID use clean-fid 0.1.35 preprocessing and its pinned Inception detector. FID is computed using the equivalent PSD/SVD formulation retained in `code/evaluate.py`, handling singular 2,048-sample covariance matrices. Independent replay from the saved vectors matches the native FID. KID is the recorded native estimate; recomputing a new randomly subsampled KID estimate need not be bit-identical without its original subset draws.

CLIP cosine uses the independent pinned ViT-B/32 evaluator, not the cached training text hidden states. Saved normalized image/text embeddings support alignment and caption retrieval replay. FID/KID depend on feature representation and sample count; the thesis's unspecified FID protocol is not directly overlaid.

## Additional metrics

PRDC is a post-hoc diagnosis of fidelity and diversity in saved Inception space. Primary neighborhood size is k=5; k=3 and 10 are retained for sensitivity. The optimized calculation was checked against the official implementation pinned at `e320c1d2811d33081361a08f595b43830b78641c`, including a self-manifold fixture. Recall must be read with precision, density and coverage rather than interpreted alone.

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

Learned generator parameters sum selected fine-tuned B plus C, including their dense boundary layers. Shared frozen VQ/text parameters and the image encoder unused by text-only inference are excluded. Joint-B training counts include its trainable image encoder. Counts are scalar parameters, not checkpoint file size or nonzero-weight counts.

Generation memory is the maximum PyTorch allocated/reserved memory during complete native generation. CUDA context/driver overhead and total-device capacity are separate. Recorded dense training-process peaks are available; complete peaks spanning the user-ended grown training stages are unavailable and remain N/A. Actual completed training updates are displayed separately to expose unequal compute budgets. Generation process walls include durable image writes and are not a controlled speed benchmark.

## Qualitative protocol

The comparison grid shows exactly the first eight validation rows, without favorable selection. Reference bytes match across arms. These native final draws are separate from fixed-seed progression animations. Animation seed banks preserve the same eight captions and feature/token seeds across optimizer keyframes. GIFs are compact, lossy display derivatives; native PNG masters remain unchanged in the local archive.

## Evidence

`evidence/dense-17` and `evidence/grown-17` retain native summaries, configurations, complete journals, both full 2,048-row embedding arrays and the displayed PNGs/references. `results/comparison.json` records every figure value. `results/provenance.json` binds copied source files and native source commit hashes. Source/sample checksums, stable checkpoint audits and FID/CLIP replay preceded reporting. No test examples are scored by analysis scripts.
