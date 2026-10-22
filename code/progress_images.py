"""Retain fixed-prompt, fixed-noise PNG keyframes for later training animations."""
import json
from pathlib import Path
import time
import secrets
import torch
from torchvision.utils import save_image
from diffusers import VQModel, AmusedScheduler
from models import build_masked_model
from low_rank import prepare_for_state
from persistence import atomic_json, save_png, durable_mkdir

ROOT = Path(__file__).resolve().parent


def write_timeline(history, frame):
    path = history / "timeline.json"
    timeline = json.loads(path.read_text()) if path.exists() else {"format_version": 1, "frames": []}
    key = (frame["phase"], frame["step"])
    timeline["frames"] = [f for f in timeline["frames"] if (f["phase"], f["step"]) != key] + [frame]
    order = {"pilot": -1, "b": 0, "c": 1, "finetune": 2}
    timeline["frames"].sort(key=lambda f: (order.get(f["phase"], 99), f["step"]))
    atomic_json(path, timeline)


def seed_bank(history, rows):
    path = history / "seed_bank.json"
    expected = [(r["row_id"], r["caption"]) for r in rows]
    if path.exists():
        bank = json.loads(path.read_text())
        assert [(r["row_id"], r["caption"]) for r in bank["samples"]] == expected, "Animation prompts changed within a run"
        return bank["samples"]
    # Repeated full-data runs should vary training initialization, while their
    # visual probes keep the original prompts/noise. Existing histories always
    # win above: adopting a shared bank can never replace a saved run's seeds.
    shared_path = ROOT / "results/visual_history/full-17-seeds.json"
    if shared_path.exists():
        shared = json.loads(shared_path.read_text())["samples"]
        if [(r["row_id"], r["caption"]) for r in shared] == expected:
            atomic_json(path, {"selection": "Inherited original OS-random seeds for matching prompts; never reselected on resume", "shared_source": str(shared_path.relative_to(ROOT)), "samples": shared})
            return shared
    samples = []
    for row in rows:
        seed = secrets.randbits(63)
        samples.append({"row_id": row["row_id"], "caption": row["caption"], "seed": seed, "feature_seed": seed, "token_seed": (seed + 1) % 2**63})
    atomic_json(path, {"selection": "OS randomness once; never reselected on resume", "samples": samples})
    return samples


def load_weights(path, name, model):
    # Do not put the saved optimizer on GPU merely to render a keyframe.
    state = torch.load(path, map_location="cpu", weights_only=False)["modules"][name]
    prepare_for_state(model, state)
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def save_progress_images(data, run, cfg, phase, step, *, b_model=None, c_model=None, stats=None, build_c=None, sample_features=None):
    count = cfg.get("progress_samples", 0)
    if not count:
        return 0.
    history = run / "visual_history"
    dest = history / phase / f"step_{step:07d}"
    if (dest / "metadata.json").exists():
        saved = json.loads((dest / "metadata.json").read_text())
        write_timeline(history, {"phase": phase, "step": step, "directory": str(dest.relative_to(history)), "grid": str((dest / "grid.png").relative_to(history)), "prior": saved["prior"], "sample_count": saved["sample_count"]})
        return 0.
    begin = time.monotonic()
    durable_mkdir(dest)
    original_modes = [(m, m.training) for m in [b_model, c_model] if m is not None]
    # Sampling, temporary model initialization and decoding cannot advance training
    # RNGs. Feature noise and token randomness are separate fixed streams as well.
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            ids = data.ids["val"][:count]
            _, text = data.batch(ids, images=False)
            seeds = seed_bank(history, [data.rows[row] for row in ids])
            if b_model is None:
                b_model = load_weights(run / "b-best.pt", "b", build_masked_model(cfg).cuda()).requires_grad_(False)
            else:
                b_model.eval()
            if phase == "b":
                features = torch.zeros(len(ids), 192, 7, 7, device="cuda")
                prior_kind = "neutral zero feature prior; text-conditioned Stage B probe before C exists"
            else:
                if c_model is None:
                    c_model = load_weights(run / "c-best.pt", "c", build_c(cfg)).requires_grad_(False)
                else:
                    c_model.eval()
                if stats is None:
                    stats = torch.load(run / "feature-stats.pt", map_location="cuda", weights_only=True)
                    if not cfg["standardize"]:
                        stats = {"mean": torch.zeros_like(stats["mean"]), "std": torch.ones_like(stats["std"])}
                features = torch.cat([sample_features(c_model, text[i:i+1], stats, cfg, torch.Generator(device="cuda").manual_seed(item["feature_seed"]), cfg.get("progress_guidance", 1.)) for i, item in enumerate(seeds)])
                prior_kind = "text-only generated C features; fixed initial feature noise"
            token_batches = []
            for i, item in enumerate(seeds):
                token_gen = torch.Generator(device="cuda").manual_seed(item["token_seed"])
                tokens = torch.full((1, 16, 16), 8192, device="cuda", dtype=torch.long)
                scheduler = AmusedScheduler(mask_token_id=8192, masking_schedule="cosine")
                scheduler.set_timesteps(cfg["sampling_steps"], temperature=(2., 0.), device="cuda")
                for t in scheduler.timesteps:
                    ratio = tokens.eq(8192).float().mean((1, 2))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = b_model(tokens, text[i:i+1], features[i:i+1], ratio).float()
                    tokens = scheduler.step(logits, t, tokens, generator=token_gen).prev_sample
                token_batches.append(tokens)
            tokens = torch.cat(token_batches)
            assert tokens.min() >= 0 and tokens.max() < 8192
            vq = VQModel.from_pretrained(ROOT / "models/amused-256", subfolder="vqvae", local_files_only=True).cuda().eval().requires_grad_(False)
            images = []
            for start in range(0, len(ids), 2):
                sub = tokens[start:start + 2]
                decoded = vq.decode(sub.reshape(-1), force_not_quantize=True, shape=(len(sub), 16, 16, 64)).sample.clamp(0, 1)
                assert torch.isfinite(decoded).all()
                images.extend(decoded.cpu())
            del vq
            references = history / "references"
            durable_mkdir(references)
            records = []
            for i, (row_id, img) in enumerate(zip(ids, images)):
                name = f"sample_{i:02d}_row_{row_id:06d}.png"
                save_png(img, dest / name)
                reference = references / f"row_{row_id:06d}.png"
                if not reference.exists():
                    real = torch.from_numpy(data.arrays["images"][row_id].copy()).float() / 255
                    save_png(real, reference)
                records.append({**data.rows[row_id], **seeds[i], "file": name})
            from torchvision.utils import make_grid
            save_png(make_grid(torch.stack(images), nrow=4), dest / "grid.png")
            metadata = {"phase": phase, "step": step, "prior": prior_kind, "seed_bank": "../../seed_bank.json", "sample_count": len(ids), "guidance": cfg.get("progress_guidance", 1.), "sampling_steps": cfg["sampling_steps"], "ddim_steps": cfg["ddim_steps"], "source": "current training weights, not a later checkpoint", "prompts": records, "created_unix": time.time()}
            atomic_json(dest / "metadata.json", metadata)
            write_timeline(history, {"phase": phase, "step": step, "directory": str(dest.relative_to(history)), "grid": str((dest / "grid.png").relative_to(history)), "prior": prior_kind, "sample_count": len(ids)})
    finally:
        for model, mode in original_modes:
            model.train(mode)
    torch.cuda.synchronize()
    return time.monotonic() - begin
