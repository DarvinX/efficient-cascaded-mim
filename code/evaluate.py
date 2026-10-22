"""Clean-FID/KID and independent CLIP similarity, with exact manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from cleanfid import fid
from cleanfid.inception_torchscript import InceptionV3W
from transformers import CLIPModel, CLIPProcessor
from sample_manifest import validate_sample_manifest
from persistence import atomic_json, atomic_write

ROOT = Path(__file__).resolve().parent


def frechet_from_features(x, y):
    """Standard sample FID via nuclear norm; handles singular small-sample covariances."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    trace_x = np.square(xc).sum() / (len(x) - 1)
    trace_y = np.square(yc).sum() / (len(y) - 1)
    if max(len(x), len(y)) <= x.shape[1]:
        # Nonzero sqrt-eigenvalues of Cx Cy are singular values of X Y^T.
        cross = np.linalg.svd(xc @ yc.T, compute_uv=False).sum() / np.sqrt((len(x) - 1) * (len(y) - 1))
    else:
        cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        eig, vec = np.linalg.eigh(cx)
        root = (vec * np.sqrt(np.clip(eig, 0, None))) @ vec.T
        middle = root @ cy @ root
        cross = np.sqrt(np.clip(np.linalg.eigvalsh((middle + middle.T) / 2), 0, None)).sum()
    return float(max(0., np.square(mx - my).sum() + trace_x + trace_y - 2 * cross))


def evaluate(run, split="val", guidance=1.):
    torch.set_num_threads(6)
    generated = run / f"samples-{split}-g{guidance}"
    refs = run / f"references-{split}"
    rows = json.loads((generated / "prompts.json").read_text())
    assert len(rows) >= 32
    files = validate_sample_manifest(rows, generated, refs, fid.EXTENSIONS)
    detector = InceptionV3W(str(ROOT / "models/inception"), download=False, resize_inside=False).cuda().eval()
    real_features = fid.get_folder_features(str(refs), detector, mode="clean", device=torch.device("cuda"), batch_size=16, num_workers=0, verbose=False)
    gen_features = fid.get_folder_features(str(generated), detector, mode="clean", device=torch.device("cuda"), batch_size=16, num_workers=0, verbose=False)
    atomic_write(run / f"inception-features-{split}-g{guidance}.npz", lambda stream: np.savez(stream, real=real_features, generated=gen_features, files=np.array(files)))
    clean_fid = frechet_from_features(real_features, gen_features)
    kid = float(fid.kernel_distance(real_features, gen_features))
    del detector
    torch.cuda.empty_cache()
    model = CLIPModel.from_pretrained(ROOT / "models/clip-eval", local_files_only=True).eval().cuda()
    processor = CLIPProcessor.from_pretrained(ROOT / "models/clip-eval", local_files_only=True)
    similarities, image_embeddings, text_embeddings, real_similarities = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            sub = rows[start:start + 16]
            images = [Image.open(generated / f"{r['row_id']:06d}.png").convert("RGB") for r in sub]
            inputs = processor(text=[r["caption"] for r in sub], images=images, padding=True, truncation=True, return_tensors="pt").to("cuda")
            output = model(**inputs)
            image_unit = F.normalize(output.image_embeds, dim=-1)
            text_unit = F.normalize(output.text_embeds, dim=-1)
            similarities.extend((image_unit * text_unit).sum(-1).cpu().tolist())
            image_embeddings.append(image_unit.cpu())
            text_embeddings.append(text_unit.cpu())
            real_images = [Image.open(refs / f"{r['row_id']:06d}.png").convert("RGB") for r in sub]
            real_inputs = processor(images=real_images, return_tensors="pt").to("cuda")
            real_features_clip = model.get_image_features(**real_inputs)
            if not isinstance(real_features_clip, torch.Tensor):
                real_features_clip = real_features_clip.pooler_output
            real_similarities.extend((F.normalize(real_features_clip, dim=-1) * text_unit).sum(-1).cpu().tolist())
    image_embeddings = torch.cat(image_embeddings)
    text_embeddings = torch.cat(text_embeddings)
    negative_text = text_embeddings.roll(1, 0)
    negative_valid = torch.tensor([r["caption"] != rows[(i-1) % len(rows)]["caption"] for i, r in enumerate(rows)])
    negative_scores = (image_embeddings * negative_text).sum(-1)[negative_valid]
    shuffled_mean = negative_scores.mean().item()
    matched_valid = (image_embeddings * text_embeddings).sum(-1)[negative_valid].mean().item()
    retrieval = (image_embeddings @ text_embeddings.T).argmax(-1).tolist()
    recall_at_1 = float(np.mean([rows[j]["caption"] == rows[i]["caption"] for i, j in enumerate(retrieval)]))
    atomic_write(run / f"clip-embeddings-{split}-g{guidance}.npz", lambda stream: np.savez(stream, images=image_embeddings.numpy(), texts=text_embeddings.numpy(), row_ids=np.array([r["row_id"] for r in rows])))
    manifest = [{"file": p.name, "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()} for p in sorted(generated.glob("*.png"))]
    reference_manifest = [{"file": p.name, "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()} for p in sorted(refs.glob("*.png"))]
    result = {"clean_fid": clean_fid, "kid": kid, "clip_cosine_mean": float(np.mean(similarities)), "clip_cosine_std": float(np.std(similarities)), "n": len(rows), "split": split, "guidance": guidance, "reference": "fixed held-out images, resized RGB256; clean-fid 0.1.35 preprocessing/detector; mathematically equivalent PSD/SVD FID calculation for SciPy1.18 compatibility", "manifest": manifest, "warning": "FID is sample-count sensitive; thesis FID191.8 has unknown protocol and is not directly comparable"}
    result.update(clip_cosine_shuffled=shuffled_mean, clip_alignment_margin=matched_valid-shuffled_mean, clip_real_reference_cosine_mean=float(np.mean(real_similarities)), clip_caption_retrieval_at_1=recall_at_1)
    result["reference_manifest"] = reference_manifest
    out = run / f"quality-{split}-g{guidance}.json"
    atomic_json(out, result)
    brief = {k: v for k, v in result.items() if k not in ("manifest", "reference_manifest")}
    return brief
