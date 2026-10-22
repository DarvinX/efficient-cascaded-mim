"""Download only pinned, relevant weights; leave other user caches untouched."""
import hashlib
import json
import os
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download


def main():
    api = HfApi()
    specs = [
        ("m1guelpf/nouns", "dataset", "505bb434cc751d0b5158ae82f368a7c63e7a94c6", "data/nouns", ["*.parquet", "README.md", "dataset_infos.json"]),
        ("amused/amused-256", "model", "09b6259bf96dbe6d70a852b70812420fe02df55e", "models/amused-256", ["model_index.json", "README.md", "vqvae/config.json", "vqvae/diffusion_pytorch_model.safetensors", "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "tokenizer/*", "scheduler/*", "transformer/config.json", "transformer/diffusion_pytorch_model.fp16.safetensors"]),
        ("openai/clip-vit-base-patch32", "model", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "models/clip-eval", ["config.json", "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "pytorch_model.bin"]),
    ]
    manifest = []
    for repo, kind, revision, dest, patterns in specs:
        if revision is None:
            revision = api.repo_info(repo, repo_type=kind).sha
        print(f"Downloading {repo}@{revision}", flush=True)
        path = Path(snapshot_download(repo, repo_type=kind, revision=revision, local_dir=ROOT / dest, allow_patterns=patterns, max_workers=2))
        files = []
        for f in sorted(path.rglob("*")):
            if f.is_file() and ".cache" not in f.parts:
                h = hashlib.file_digest(f.open("rb"), "sha256").hexdigest()
                files.append({"path": str(f.relative_to(ROOT)), "bytes": f.stat().st_size, "sha256": h})
        manifest.append({"repo": repo, "type": kind, "revision": revision, "files": files})
        (ROOT / "results/assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    detector = ROOT / "models/inception/inception-2015-12-05.pt"
    detector.parent.mkdir(parents=True, exist_ok=True)
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    if not detector.exists():
        partial = detector.with_suffix(".partial")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        partial.replace(detector)
    (ROOT / "results/inception-asset.json").write_text(json.dumps({"url": url, "path": str(detector.relative_to(ROOT)), "bytes": detector.stat().st_size, "sha256": hashlib.file_digest(detector.open("rb"), "sha256").hexdigest()}, indent=2) + "\n")
    print("Assets ready", flush=True)


if __name__ == "__main__":
    main()
