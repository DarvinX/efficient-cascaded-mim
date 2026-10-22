"""Cache frozen components on GPU, with a stable image-group split and resume."""
import argparse
import hashlib
import fcntl
import io
import json
import os
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from diffusers import VQModel
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from torchvision.utils import save_image
from persistence import atomic_json, flush_arrays, durable_mkdir
import shutdown

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=4096, help="0 = complete dataset")
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()
    shutdown.install()
    lock = (ROOT / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(6)
    assert torch.cuda.is_available(), "GPU required; run outside sandbox"
    target = ROOT / "data" / (f"cached-{args.limit}" if args.limit else "cached-full")
    durable_mkdir(target)
    if (target / "READY.json").exists():
        print("Already ready:", target, flush=True)
        return
    parquet = pq.ParquetFile(next((ROOT / "data/nouns/data").glob("*.parquet")))
    n = min(args.limit or parquet.metadata.num_rows, parquet.metadata.num_rows)
    shapes = {"images": ((n, 3, 256, 256), np.uint8), "tokens": ((n, 16, 16), np.int16), "text": ((n, 77, 768), np.float16)}
    arrays = {}
    for name, (shape, dtype) in shapes.items():
        path = target / f"{name}.npy"
        arrays[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=shape)
    vq = VQModel.from_pretrained(ROOT / "models/amused-256", subfolder="vqvae", local_files_only=True).eval().requires_grad_(False).cuda()
    text_model = CLIPTextModelWithProjection.from_pretrained(ROOT / "models/amused-256", subfolder="text_encoder", variant="fp16", local_files_only=True, dtype=torch.float16).eval().requires_grad_(False).cuda()
    tokenizer = CLIPTokenizer.from_pretrained(ROOT / "models/amused-256/tokenizer", local_files_only=True)
    progress = target / "progress.json"
    start = json.loads(progress.read_text())["completed"] if progress.exists() else 0
    rows = json.loads((target / "rows.json").read_text())[:start] if start else []
    assert len(rows) == start, "Cache metadata is behind its committed progress"
    previous = ROOT / "data/cached-4096"
    if not args.limit and start == 0 and (previous / "READY.json").exists():
        old = json.loads((previous / "READY.json").read_text())
        assert old["vq_revision"] == "09b6259bf96dbe6d70a852b70812420fe02df55e"
        assert old["preprocessing"] == "RGB; bilinear resize 256; VQ input [0,1]"
        start = old["n"]
        assert start % args.batch == 0
        for name, out in arrays.items():
            src = np.load(previous / f"{name}.npy", mmap_mode="r")
            for chunk in range(0, start, 256):
                out[chunk:chunk + 256] = src[chunk:chunk + 256]
            flush_arrays([out], [target / f"{name}.npy"])
        rows = json.loads((previous / "rows.json").read_text())
        atomic_json(target / "rows.json", rows)
        atomic_json(progress, {"completed": start})
        print(f"Reused {start} verified frozen-component cache entries", flush=True)
    begin = time.monotonic()
    print(f"Resuming frozen cache at {start}/{n}", flush=True)
    offset = 0
    with torch.inference_mode():
        for batch in parquet.iter_batches(batch_size=args.batch):
            raw = batch.to_pylist()
            stop = min(offset + len(raw), n)
            if offset >= n:
                break
            if stop <= start:
                offset = stop
                continue
            assert offset >= start, "Resume requires unchanged batch size"
            imgs, captions = [], []
            for row_id, row in enumerate(raw[:stop - offset], offset):
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                image_hash = hashlib.sha256(np.asarray(img).tobytes()).hexdigest()
                bucket = int(image_hash[:8], 16) % 100
                split = "train" if bucket < 90 else ("val" if bucket < 95 else "test")
                resized = np.asarray(img.resize((256, 256), Image.Resampling.BILINEAR)).copy().transpose(2, 0, 1)
                imgs.append(resized)
                captions.append(row["text"])
                rows.append({"row_id": row_id, "image_sha256": image_hash, "split": split, "caption": row["text"]})
            pixels = torch.from_numpy(np.stack(imgs)).cuda().float() / 255
            latent = vq.encode(pixels).latents
            tokens = vq.quantize(latent)[2][2].reshape(-1, 16, 16)
            inputs = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.cuda()
            embeddings = text_model(inputs, output_hidden_states=True).hidden_states[-2]
            assert tokens.min() >= 0 and tokens.max() < 8192
            assert torch.isfinite(embeddings).all()
            arrays["images"][offset:stop] = np.stack(imgs)
            arrays["tokens"][offset:stop] = tokens.cpu().numpy().astype(np.int16)
            arrays["text"][offset:stop] = embeddings.cpu().numpy()
            if offset == 0:
                decoded = vq.decode(tokens.reshape(-1), force_not_quantize=True, shape=(len(imgs), 16, 16, 64)).sample.clamp(0, 1)
                save_image(torch.cat([pixels, decoded]), ROOT / "results/vq-reconstruction.png", nrow=len(imgs))
                (ROOT / "results/vq-check.json").write_text(json.dumps({"latent_shape": list(latent.shape), "token_shape": list(tokens.shape), "reconstruction_mse": (decoded - pixels).square().mean().item(), "finite": bool(torch.isfinite(decoded).all()), "input_range": "[0,1]", "parameters": sum(p.numel() for p in vq.parameters())}, indent=2))
            offset = stop
            if offset % 64 == 0 or offset == n or shutdown.requested:
                flush_arrays(arrays.values(), [target / f"{name}.npy" for name in arrays])
                atomic_json(target / "rows.json", rows)
                atomic_json(progress, {"completed": offset})
                print(f"cached {offset}/{n}; {(offset-start) / max(time.monotonic()-begin, 1):.1f} fresh images/s", flush=True)
                if shutdown.requested:
                    print("Cache checkpoint flushed; stopping safely", flush=True)
                    raise SystemExit(75)
    assert len(rows) == n
    counts = {s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]}
    digest = hashlib.sha256((target / "rows.json").read_bytes()).hexdigest()
    atomic_json(target / "READY.json", {"n": n, "counts": counts, "rows_sha256": digest, "preprocessing": "RGB; bilinear resize 256; VQ input [0,1]", "vq_revision": "09b6259bf96dbe6d70a852b70812420fe02df55e"})
    print("Cache ready", counts, flush=True)


if __name__ == "__main__":
    main()
