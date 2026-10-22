# Reproduction and artifact layout

## Read or regenerate the results without a GPU

The figures and native numerical evidence are included. Python 3.13.5 was used locally. For figure regeneration:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
python analysis/plot_figures.py
python analysis/verify_results.py
python analysis/plot_learning.py
```

The plotting command reads saved evidence only and writes a new `figures-rebuilt/` directory. It refuses to overwrite an existing output. Figure metric values are read from recorded results. The verification command independently replays FID, CLIP, PRDC and retrieval from the saved vectors and checks displayed PNG hashes, without training, model selection, downloads or test evaluation. KID remains the original native estimate because its subset draws were not serialized. The data underlying the three main figures are `results/comparison.json`.

## Reference training implementation

The cascaded model and GILoRaT-based training approach were developed in my MTP. The code in this repository was recreated from the retained thesis sources and diagrams because the original implementation and checkpoints were unavailable. The additional measurements describe the recorded local runs; they are not presented as the original thesis results.

The measured core implementation is in `code/`; its local paths intentionally resolve from that directory. Trainable modules and optimizer/RNG behavior are preserved byte for byte from the measured source. The packaging manifest lists their SHA256 digests. `code/scripts/download_assets.py` additionally pins the exact recorded evaluation CLIP revision instead of resolving a future default.

The recorded CUDA stack is in `environment.json`. Install its matching PyTorch/torchvision CUDA builds, then install the remaining dependencies from `requirements.txt`. The environment was inherited from the verified local CUDA installation; this release does not claim a clean-machine GPU retraining certification.

Asset download and cache preparation are explicit operations:

```bash
cd code
python scripts/download_assets.py
python prepare.py --limit 0 --batch 4
```

The downloaded dataset and model revisions are pinned and recorded in `code/results/assets.json`. Caches, pretrained models and native optimizer checkpoints are intentionally excluded from Git.

Dense stage commands use the original finite-budget configuration:

```bash
python train.py --config configs/dense.json --run dense-17 --seed 17 --stage b
python train.py --config configs/dense.json --run dense-17 --seed 17 --stage features
python train.py --config configs/dense.json --run dense-17 --seed 17 --stage c
python train.py --config configs/dense.json --run dense-17 --seed 17 --stage generated
python train.py --config configs/dense.json --run dense-17 --seed 17 --stage finetune
python train.py --config configs/dense.json --run dense-17 --seed 17 --stage generate --count 2048 --split val --guidance 1.0
python train.py --config configs/dense.json --run dense-17 --seed 17 --stage quality --count 2048 --split val --guidance 1.0
```

`configs/grown.json` preserves the extended run's safety ceilings. Those ceilings are not its observed stop points. Its B/C/FT phases were manually stopped at 387,453/229,472/91,059 and selected at 385,500/228,500/89,500. Rebuilding its cascade requires the coherent selected B → real targets → C → generated targets → fine-tuned B chain documented in `TRAINING.md`; the earlier downstream targets must not be reused. The exact measured checkpoints are identified below and remain in the local artifact vault rather than being recreated by figure regeneration.

For the recorded fixed-rank fine-tuning policy, `code/scripts/train_fixed_rank.py` applies the retained native interception using `code/configs/fixed_rank_policy.json`. This keeps inherited B ranks at 8 while weights train, validates source/configuration hashes and dispatches only fine-tuning for `grown-17`. A native checkpoint with matching grown dimensions and feature-cache lineage is required first.

## Selected native checkpoint identities

- `b-best.pt`: `dde4e9faf5dcc61831008ee5f6a6a493ad070bfee01d3fcf5aeaab8256678316`
- `c-best.pt`: `9cd798a226549e709c6e800ffbdbdb0281576e373f186630556f6f8ddebc67fc`
- `feature-stats.pt`: `ee94a488f5631fa3990d4de56ab5d45a0a4550112538a95cef346e86aa8120ab`
- `finetune-best.pt`: `9ec30256d5d1848ef542adadb61122400a0fdc3adab1391eaed549e375a88e78`

## Local research archive

The original campaign repository retains all training/checkpoint histories, native PNG masters, source guards, stopped boot-resume state and my full thesis archive. The final byte inventory covers 77,217 native data/model/run paths. That inventory is preservation evidence, not a remote backup. This curated repository contains source, selected numerical evidence, figures and compact animations; raw training caches and optimizer tensors remain outside Git. Resuming a native job requires its original local lineage and explicit execution, not merely opening this repository.

## Reproducibility limits

Reported numbers are native validation results with source/PNG/embedding audits, not a claim of exact reproduction of my original thesis results or hardware-independent bitwise retraining. The extended arm has one seed and unequal training duration. All phases ended before a verified terminal plateau. Test data remain unscored. These limits accompany the resource savings throughout the README and figures.
