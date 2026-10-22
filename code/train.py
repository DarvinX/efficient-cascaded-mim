"""One guarded runner for training, sampling and machine-readable measurements."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from models import ImageFeatures, MaskedModel, build_masked_model, FeatureDiffusion, FeatureResNet, noise_schedule, mask_tokens
from low_rank import factorize_transformer, prepare_for_state, grow_low_score_layers, grow_with_seed, maximum_rank, orthogonal_penalty
from random_streams import training_streams, stream_states, restore_streams
from epoch_sampler import ShuffledEpochs
from progress_images import save_progress_images
from persistence import atomic_json, atomic_link, atomic_write, save_torch, load_torch, checkpoint_exists, flush_arrays, durable_mkdir, save_png
import shutdown

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Cache:
    def __init__(self, path):
        self.path = path
        self.info = json.loads((path / "READY.json").read_text())
        self.rows = json.loads((path / "rows.json").read_text())
        assert hashlib.sha256((path / "rows.json").read_bytes()).hexdigest() == self.info["rows_sha256"]
        self.arrays = {n: np.load(path / f"{n}.npy", mmap_mode="r") for n in ["images", "tokens", "text"]}
        self.ids = {s: np.array([i for i, r in enumerate(self.rows) if r["split"] == s]) for s in ["train", "val", "test"]}
        groups = {s: {self.rows[i]["image_sha256"] for i in ids} for s, ids in self.ids.items()}
        assert not groups["train"] & groups["val"] and not groups["train"] & groups["test"] and not groups["val"] & groups["test"]

    def batch(self, ids, images=True):
        tokens = torch.from_numpy(self.arrays["tokens"][ids].copy()).to(DEVICE, dtype=torch.long)
        text = torch.from_numpy(self.arrays["text"][ids].copy()).to(DEVICE, dtype=torch.bfloat16)
        if images:
            pixels = torch.from_numpy(self.arrays["images"][ids].copy()).to(DEVICE, dtype=torch.float32) / 255
            return pixels, tokens, text
        return tokens, text


def autocast():
    return torch.autocast(DEVICE, dtype=torch.bfloat16)


def save_checkpoint(path, modules, optimizer, step, cfg, rng, extra=None, streams=None):
    value = {"modules": {k: m.state_dict() for k, m in modules.items()}, "optimizer": optimizer.state_dict() if optimizer else None, "step": step, "config": cfg, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "numpy_rng": rng.bit_generator.state, "streams": stream_states(streams or {}), "extra": extra or {}}
    save_torch(path, value, keep_previous=True)


def load_checkpoint(path, modules, optimizer=None, rng=None, restore_rng=False, streams=None):
    state = load_torch(path, map_location=DEVICE, weights_only=False)
    changed = False
    for k, m in modules.items():
        changed = prepare_for_state(m, state["modules"][k]) or changed
        m.load_state_dict(state["modules"][k])
    if optimizer is not None:
        if changed:
            assert len(optimizer.param_groups) == 1
            optimizer.param_groups[0]["params"] = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        rng.bit_generator.state = state["numpy_rng"]
        if streams:
            restore_streams(streams, state["streams"])
    return state


@torch.inference_mode()
def validation_b(model, eff, data, cfg, generated=None, zero_features=False):
    model.eval()
    eff.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(91231)
    sums, count = 0., 0
    ids = data.ids["val"][:cfg["val_samples"]]
    for start in range(0, len(ids), cfg["batch"]):
        sub = ids[start:start + cfg["batch"]]
        pixels, tokens, text = data.batch(sub)
        masked, labels, ratio = mask_tokens(tokens, gen, fixed_ratio=.75)
        with autocast():
            features = eff(pixels) if generated is None else torch.from_numpy(np.asarray(generated[sub]).copy()).to(DEVICE)
            if zero_features:
                features = torch.zeros_like(features)
            logits = model(masked, text, features, ratio)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        sums += loss.item()
        count += labels.ne(-100).sum().item()
    return sums / count


def json_event(run, event):
    with (run / "events.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(event), flush=True)


def train_b(data, run, cfg, stage):
    model, eff = build_masked_model(cfg).cuda(), ImageFeatures().cuda()
    modules = {"b": model, "eff": eff}
    generated = None
    if stage == "finetune":
        load_checkpoint(run / "b-best.pt", modules)
        eff.requires_grad_(False).eval()
        generated = np.load(run / "generated-features.npy", mmap_mode="r")
    params = list(model.parameters()) + (list(eff.parameters()) if stage == "b" else [])
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"] if stage == "b" else cfg["finetune_lr"], weight_decay=cfg["weight_decay"], fused=True)
    rng = np.random.default_rng(cfg["seed"] + (1000 if stage == "finetune" else 0))
    streams = training_streams(cfg, stage)
    total = cfg["b_steps"] if stage == "b" else math.ceil(len(data.ids["train"]) / (cfg["batch"] * cfg["accumulate"])) * cfg["finetune_epochs"]
    latest, best = run / f"{stage}-latest.pt", run / f"{stage}-best.pt"
    step, best_val, original_initial, stagnant = 0, float("inf"), None, 0
    settings=cfg.get("b_gilorat")
    saved_sampler=None
    if checkpoint_exists(latest):
        state = load_checkpoint(latest, modules, optimizer, rng, True, streams)
        assert state["config"] == cfg, "Refuse resume with changed configuration"
        step, best_val = state["step"], state["extra"]["best_val"]
        original_initial = state["extra"].get("initial_val_ce")
        stagnant=state["extra"].get("stagnant",0)
        saved_sampler=state["extra"].get("ft_sampler")
        params=list(model.parameters())+(list(eff.parameters()) if stage=="b" else [])
        del state  # Optimizer owns its moments; release duplicate loaded model tensors.
    epoch_sampler=None
    if stage=="finetune" and cfg.get("rng_protocol",1)>=2 and (step==0 or saved_sampler is not None):
        epoch_sampler=ShuffledEpochs(data.ids["train"],cfg["batch"],cfg["accumulate"],cfg["finetune_epochs"],rng)
        if saved_sampler is not None:
            epoch_sampler.load_state_dict(saved_sampler)
            assert epoch_sampler.updates_completed==step,"Model/data cursor checkpoint mismatch"
    resumed_val = validation_b(model, eff, data, cfg, generated)
    initial = resumed_val if original_initial is None else original_initial
    sampling="shuffled-epochs-v1" if epoch_sampler else ("legacy-random-epoch-equivalent" if stage=="finetune" else "random-step-budget")
    json_event(run, {"stage": stage, "step": step, "initial_val_ce": initial, "current_val_ce": resumed_val, "resumed_from_step": step,"data_sampling":sampling, "trainable_parameters": sum(p.numel() for p in params)})
    snapshot_seconds = save_progress_images(data, run, cfg, stage, step, b_model=model, build_c=build_c, sample_features=sample_features)
    initial_step = step
    def checkpoint_extra():
        return {"best_val":best_val,"initial_val_ce":initial,"stagnant":stagnant,"ft_sampler":epoch_sampler.state_dict() if epoch_sampler else None}
    if not checkpoint_exists(latest):
        save_checkpoint(latest, modules, optimizer, step, cfg, rng, checkpoint_extra(), streams)
    begin = time.monotonic()
    last_save = time.monotonic()
    while step < total:
        model.train()
        if stage == "b":
            eff.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        objective_sum=0.
        epoch_ids=epoch_sampler.next_batch() if epoch_sampler else None
        micro_count=math.ceil(len(epoch_ids)/cfg["batch"]) if epoch_ids is not None else cfg["accumulate"]
        for micro in range(micro_count):
            ids=epoch_ids[micro*cfg["batch"]:(micro+1)*cfg["batch"]] if epoch_ids is not None else rng.choice(data.ids["train"],cfg["batch"],replace=False)
            pixels, tokens, text = data.batch(ids)
            masked, labels, ratio = mask_tokens(tokens, streams.get("mask"))
            with autocast():
                if generated is None:
                    features = eff(pixels)
                else:
                    features = torch.from_numpy(np.asarray(generated[ids]).copy()).to(DEVICE)
                    if cfg["feature_mix"]:
                        with torch.no_grad():
                            real = eff(pixels)
                        use_real = torch.rand(len(ids), 1, 1, 1, device=DEVICE, generator=streams.get("mixture")) < cfg["feature_mix"]
                        features = torch.where(use_real, real, features)
                if cfg["cond_dropout"]:
                    text = text * (torch.rand(len(ids), 1, 1, device=DEVICE, generator=streams.get("dropout")) >= cfg["cond_dropout"])
                logits = model(masked, text, features, ratio)
                mean_ce=F.cross_entropy(logits.float(),labels)
                masked_ce=mean_ce*(len(ids)/len(epoch_ids)) if epoch_ids is not None else mean_ce/cfg["accumulate"]
                loss=masked_ce
                if settings and settings.get("orthogonal_weight",0):
                    scale=len(ids)/len(epoch_ids) if epoch_ids is not None else 1/cfg["accumulate"]
                    loss=loss+settings["orthogonal_weight"]*orthogonal_penalty(model)*scale
            assert torch.isfinite(loss), "Nonfinite B loss"
            loss.backward()
            loss_sum += masked_ce.item()
            if settings and settings.get("orthogonal_weight",0):
                objective_sum+=loss.item()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(norm), "Nonfinite B gradient"
        optimizer.step()
        step += 1
        if step % 25 == 0:
            event={"stage":stage,"step":step,"train_ce":loss_sum,"seconds":time.monotonic()-begin}
            if settings and settings.get("orthogonal_weight",0):
                event["train_objective"]=objective_sum
            json_event(run,event)
        evaluate_now = step % cfg["eval_every"] == 0 or step == total
        is_best = False
        if evaluate_now:
            val = validation_b(model, eff, data, cfg, generated)
            is_best = val < best_val
            best_val = min(best_val, val)
            stagnant=0 if is_best else stagnant+1
            if settings and stagnant>=max(1,maximum_rank(model)*settings.get("patience_factor",2)):
                basis_seed=cfg["seed"]+52129+step*1009+(1000000 if stage=="finetune" else 0)
                growth=grow_with_seed(model,settings,optimizer,basis_seed)
                stagnant=0
                params=list(model.parameters())+(list(eff.parameters()) if stage=="b" else [])
                json_event(run,{"stage":stage,"step":step,"rank_growth":growth,"trainable_parameters":sum(p.numel() for p in params)})
        save_now = evaluate_now or step % cfg.get("checkpoint_every", 100) == 0 or time.monotonic() - last_save >= cfg.get("checkpoint_seconds", 60) or shutdown.requested
        if save_now:
            save_checkpoint(latest, modules, optimizer, step, cfg, rng, checkpoint_extra(), streams)
            last_save = time.monotonic()
            if is_best:
                atomic_link(latest, best)
            json_event(run, {"stage": stage, "step": step, "durable_checkpoint": str(latest)})
        if shutdown.requested:
            print("Graceful stop: checkpoint flushed to disk", flush=True)
            raise SystemExit(75)
        if evaluate_now:
            json_event(run, {"stage": stage, "step": step, "val_ce": val, "best_val_ce": best_val})
            elapsed = save_progress_images(data, run, cfg, stage, step, b_model=model, build_c=build_c, sample_features=sample_features)
            snapshot_seconds += elapsed
            if elapsed:
                json_event(run, {"stage": stage, "step": step, "progress_images_seconds": elapsed})
    training_parameters=sum(p.numel() for p in params)
    training_rank=maximum_rank(model)
    if epoch_sampler:
        assert epoch_sampler.completed_epochs==cfg["finetune_epochs"]
        assert epoch_sampler.examples_seen==len(data.ids["train"])*cfg["finetune_epochs"]
    load_checkpoint(best, modules)
    ablated = validation_b(model, eff, data, cfg, generated, zero_features=True)
    return {"initial_val_ce": initial, "best_val_ce": best_val, "zero_feature_val_ce": ablated, "steps": total, "steps_this_process": total - initial_step, "resumed_from_step": initial_step,"data_sampling":sampling,"epochs_completed":epoch_sampler.completed_epochs if epoch_sampler else None,"training_examples_seen":epoch_sampler.examples_seen if epoch_sampler else total*cfg["batch"]*cfg["accumulate"], "timing_scope": "this process; do not use interrupted segments for an uninterrupted speed verdict", "progress_images_seconds": snapshot_seconds, "trainable_parameters":training_parameters,"selected_trainable_parameters":sum(p.numel() for p in model.parameters())+(sum(p.numel() for p in eff.parameters()) if stage=="b" else 0),"maximum_rank_at_training_end":training_rank,"selected_maximum_rank":maximum_rank(model)}


@torch.inference_mode()
def cache_features(data, run, cfg):
    eff = ImageFeatures().cuda().eval()
    load_checkpoint(run / "b-best.pt", {"eff": eff})
    dest = run / "features.npy"
    marker = run / "features-progress.json"
    completed = json.loads(marker.read_text())["completed"] if marker.exists() else 0
    out = np.lib.format.open_memmap(dest, mode="r+" if dest.exists() and completed else "w+", dtype=np.float16, shape=(len(data.rows), 192, 7, 7))
    for start in range(completed, len(data.rows), cfg["batch"]):
        ids = np.arange(start, min(start + cfg["batch"], len(data.rows)))
        pixels, _, _ = data.batch(ids)
        with autocast():
            features = eff(pixels)
        out[ids] = features.float().cpu().numpy()
        end = start + len(ids)
        if end % 128 == 0 or end == len(data.rows) or shutdown.requested:
            flush_arrays([out], [dest])
            atomic_json(marker, {"completed": end})
            if shutdown.requested:
                raise SystemExit(75)
    # Fit on train only; never fit a statistic to validation or test.
    train = torch.from_numpy(np.asarray(out[data.ids["train"]]).astype(np.float32))
    mean = train.mean((0, 2, 3), keepdim=True)
    std = train.std((0, 2, 3), keepdim=True).clamp(min=.01)
    save_torch(run / "feature-stats.pt", {"mean": mean, "std": std})
    return {"samples": len(data.rows), "feature_shape": [192, 7, 7], "mean_channel_std": std.mean().item()}


def get_feature_data(run, cfg):
    raw = np.load(run / "features.npy", mmap_mode="r")
    stats = torch.load(run / "feature-stats.pt", map_location=DEVICE, weights_only=True)
    if not cfg["standardize"]:
        stats = {"mean": torch.zeros_like(stats["mean"]), "std": torch.ones_like(stats["std"])}
    return raw, stats


def build_c(cfg):
    arch = cfg.get("c_arch", "transformer")
    assert arch in ["transformer", "resnet_attention"]
    model = (FeatureDiffusion if arch == "transformer" else FeatureResNet)(**cfg["c"])
    if cfg.get("gilorat"):
        assert arch == "transformer", "GILoRaT ablation is defined for transformer blocks"
        model = factorize_transformer(model, cfg["gilorat"])
    return model.cuda()


def feature_batch(raw, ids, stats):
    x = torch.from_numpy(np.asarray(raw[ids]).copy()).to(DEVICE, dtype=torch.float32)
    return (x - stats["mean"]) / stats["std"]


@torch.inference_mode()
def validation_c(model, data, raw, stats, cfg, alpha):
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(91231)
    errors = []
    ids = data.ids["val"][:cfg["val_samples"]]
    for start in range(0, len(ids), cfg["batch"]):
        sub = ids[start:start + cfg["batch"]]
        _, text = data.batch(sub, images=False)
        x = feature_batch(raw, sub, stats)
        noise = torch.randn(x.shape, device=DEVICE, generator=gen)
        t = torch.randint(0, 1000, (len(sub),), device=DEVICE, generator=gen)
        a = alpha[t, None, None, None]
        with autocast():
            pred = model(a.sqrt() * x + (1 - a).sqrt() * noise, t, text)
        errors.extend((pred.float() - noise).square().mean((1, 2, 3)).cpu().tolist())
    return float(np.mean(errors))


def train_c(data, run, cfg):
    model = build_c(cfg)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=True)
    rng = np.random.default_rng(cfg["seed"])
    streams = training_streams(cfg, "c")
    raw, stats = get_feature_data(run, cfg)
    alpha = noise_schedule(DEVICE)
    latest, best = run / "c-latest.pt", run / "c-best.pt"
    step, best_val, stagnant, original_initial = 0, float("inf"), 0, None
    if checkpoint_exists(latest):
        state = load_checkpoint(latest, {"c": model}, optimizer, rng, True, streams)
        assert state["config"] == cfg
        step, best_val = state["step"], state["extra"]["best_val"]
        stagnant = state["extra"].get("stagnant", 0)
        original_initial = state["extra"].get("initial_val_mse")
        params = list(model.parameters())
        del state
    resumed_val = validation_c(model, data, raw, stats, cfg, alpha)
    initial = resumed_val if original_initial is None else original_initial
    json_event(run, {"stage": "c", "step": step, "initial_val_mse": initial, "current_val_mse": resumed_val, "resumed_from_step": step, "trainable_parameters": sum(p.numel() for p in params)})
    snapshot_seconds = save_progress_images(data, run, cfg, "c", step, c_model=model, stats=stats, build_c=build_c, sample_features=sample_features)
    initial_step = step
    if not checkpoint_exists(latest):
        save_checkpoint(latest, {"c": model}, optimizer, step, cfg, rng, {"best_val": best_val, "initial_val_mse": initial, "stagnant": stagnant}, streams)
    begin = time.monotonic()
    last_save = time.monotonic()
    while step < cfg["c_steps"]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, mses = [], []
        for _ in range(cfg["accumulate"]):
            ids = rng.choice(data.ids["train"], cfg["batch"], replace=False)
            _, text = data.batch(ids, images=False)
            x = feature_batch(raw, ids, stats)
            noise = torch.randn(x.shape, device=DEVICE, generator=streams.get("noise"))
            t = torch.randint(0, 1000, (len(ids),), device=DEVICE, generator=streams.get("time"))
            a = alpha[t, None, None, None]
            if cfg["cond_dropout"]:
                text = text * (torch.rand(len(ids), 1, 1, device=DEVICE, generator=streams.get("dropout")) >= cfg["cond_dropout"])
            with autocast():
                pred = model(a.sqrt() * x + (1 - a).sqrt() * noise, t, text)
                mse = (pred.float() - noise).square().mean((1, 2, 3))
                weight = (1 / (1 - alpha[t])).clamp(max=100) if cfg["diffusion_weight"] == "inverse_variance" else torch.ones_like(t, dtype=torch.float32)
                loss = (mse * weight).mean() / cfg["accumulate"]
                if cfg.get("gilorat", {}).get("orthogonal_weight", 0):
                    loss = loss + cfg["gilorat"]["orthogonal_weight"] * orthogonal_penalty(model) / cfg["accumulate"]
            assert torch.isfinite(loss), "Nonfinite C loss"
            loss.backward()
            losses.append(loss.item())
            mses.append(mse.mean().item())
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        assert torch.isfinite(norm), "Nonfinite C gradient"
        optimizer.step()
        step += 1
        if step % 25 == 0:
            json_event(run, {"stage": "c", "step": step, "train_objective": sum(losses), "train_mse": float(np.mean(mses)), "seconds": time.monotonic() - begin})
        evaluate_now = step % cfg["eval_every"] == 0 or step == cfg["c_steps"]
        is_best = False
        if evaluate_now:
            val = validation_c(model, data, raw, stats, cfg, alpha)
            is_best = val < best_val
            best_val = min(best_val, val)
            stagnant = 0 if is_best else stagnant + 1
            settings = cfg.get("gilorat")
            if settings and stagnant >= max(1, maximum_rank(model) * settings.get("patience_factor", 2)):
                growth = grow_low_score_layers(model, settings, optimizer)
                stagnant = 0
                params = list(model.parameters())
                json_event(run, {"stage": "c", "step": step, "rank_growth": growth, "trainable_parameters": sum(p.numel() for p in params)})
        save_now = evaluate_now or step % cfg.get("checkpoint_every", 100) == 0 or time.monotonic() - last_save >= cfg.get("checkpoint_seconds", 60) or shutdown.requested
        if save_now:
            save_checkpoint(latest, {"c": model}, optimizer, step, cfg, rng, {"best_val": best_val, "initial_val_mse": initial, "stagnant": stagnant}, streams)
            last_save = time.monotonic()
            if is_best:
                atomic_link(latest, best)
            json_event(run, {"stage": "c", "step": step, "durable_checkpoint": str(latest)})
        if shutdown.requested:
            print("Graceful stop: checkpoint flushed to disk", flush=True)
            raise SystemExit(75)
        if evaluate_now:
            json_event(run, {"stage": "c", "step": step, "val_mse": val, "best_val_mse": best_val})
            elapsed = save_progress_images(data, run, cfg, "c", step, c_model=model, stats=stats, build_c=build_c, sample_features=sample_features)
            snapshot_seconds += elapsed
            if elapsed:
                json_event(run, {"stage": "c", "step": step, "progress_images_seconds": elapsed})
    return {"initial_val_mse": initial, "best_val_mse": best_val, "steps": cfg["c_steps"], "steps_this_process": cfg["c_steps"] - initial_step, "resumed_from_step": initial_step, "timing_scope": "this process; do not use interrupted segments for an uninterrupted speed verdict", "progress_images_seconds": snapshot_seconds, "trainable_parameters": sum(p.numel() for p in params)}


@torch.inference_mode()
def sample_features(model, text, stats, cfg, generator, guidance=1.):
    from diffusers import DDIMScheduler
    model.eval()
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=False)
    sched.set_timesteps(cfg["ddim_steps"], device=DEVICE)
    x = torch.randn(len(text), 192, 7, 7, device=DEVICE, generator=generator)
    for t in sched.timesteps:
        ts = t.expand(len(text))
        with autocast():
            noise = model(x, ts, text).float()
            if guidance != 1.:
                uncond = model(x, ts, torch.zeros_like(text)).float()
                noise = uncond + guidance * (noise - uncond)
        x = sched.step(noise, t, x, eta=0.).prev_sample
    result = x * stats["std"] + stats["mean"]
    assert torch.isfinite(result).all()
    return result


@torch.inference_mode()
def generate_feature_cache(data, run, cfg):
    model = build_c(cfg).eval()
    load_checkpoint(run / "c-best.pt", {"c": model})
    _, stats = get_feature_data(run, cfg)
    path = run / "generated-features.npy"
    marker = run / "generated-progress.pt"
    saved = load_torch(marker, map_location="cpu", weights_only=False) if marker.exists() else None
    completed = saved["completed"] if saved else 0
    out = np.lib.format.open_memmap(path, mode="r+" if path.exists() and completed else "w+", dtype=np.float16, shape=(len(data.rows), 192, 7, 7))
    gen = torch.Generator(device=DEVICE).manual_seed(cfg["seed"] + 5000)
    if saved:
        gen.set_state(saved["generator_state"])
    generation_batch = cfg.get("feature_generation_batch", 32)
    for start in range(completed, len(data.rows), generation_batch):
        ids = np.arange(start, min(start + generation_batch, len(data.rows)))
        _, text = data.batch(ids, images=False)
        out[ids] = sample_features(model, text, stats, cfg, gen).cpu().numpy()
        end = start + len(ids)
        if end % 256 == 0 or end == len(data.rows) or shutdown.requested:
            flush_arrays([out], [path])
            save_torch(marker, {"completed": end, "generator_state": gen.get_state()})
            print(f"generated features {end}/{len(data.rows)}; saved resume state", flush=True)
            if shutdown.requested:
                raise SystemExit(75)
    return {"samples": len(data.rows), "max_abs_feature": float(np.abs(out).max())}


@torch.inference_mode()
def generate(data, run, cfg, count, split, guidance):
    import diffusers
    from diffusers import VQModel, AmusedScheduler
    from generation_recovery import GenerationRecovery
    from sample_manifest import validate_sample_manifest
    from torchvision.utils import make_grid
    from PIL import Image
    bpath = run / "finetune-best.pt"
    if not bpath.exists():
        bpath = run / "b-best.pt"
    ids = data.ids[split][:count]
    if not len(ids):
        raise ValueError("Generation needs at least one prompt")
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    dest, references = run / f"samples-{split}-g{guidance}", run / f"references-{split}"
    rows = [data.rows[int(i)] for i in ids]
    recovery = GenerationRecovery(dest, references, rows, {"config": cfg, "requested_count": count, "split": split, "guidance": guidance, "b_checkpoint": bpath.name, "b_sha256": digest(bpath), "c_sha256": digest(run / "c-best.pt"), "stats_sha256": digest(run / "feature-stats.pt"), "assets_sha256": digest(ROOT / "results/assets.json"), "cache_rows_sha256": data.info["rows_sha256"], "model_code_sha256": digest(ROOT / "models.py"), "sampler_code_sha256": digest(ROOT / "train.py"), "torch_version": torch.__version__, "diffusers_version": diffusers.__version__})
    gen = torch.Generator(device=DEVICE).manual_seed(cfg["seed"] + 9999)
    completed, previous_seconds = recovery.restore(gen)
    if not checkpoint_exists(recovery.path):
        recovery.commit(0, gen, 0.)
    b, c = build_masked_model(cfg).cuda().eval(), build_c(cfg).eval()
    load_checkpoint(bpath, {"b": b})
    load_checkpoint(run / "c-best.pt", {"c": c})
    _, stats = get_feature_data(run, cfg)
    vq = VQModel.from_pretrained(ROOT / "models/amused-256", subfolder="vqvae", local_files_only=True).cuda().eval()
    torch.cuda.synchronize()
    begin = time.monotonic()
    for start in range(completed, len(ids), cfg["batch"]):
        sub = ids[start:start + cfg["batch"]]
        _, text = data.batch(sub, images=False)
        features = sample_features(c, text, stats, cfg, gen, guidance)
        tokens = torch.full((len(sub), 16, 16), 8192, device=DEVICE, dtype=torch.long)
        sched = AmusedScheduler(mask_token_id=8192, masking_schedule="cosine")
        sched.set_timesteps(cfg["sampling_steps"], temperature=(2., 0.), device=DEVICE)
        for t in sched.timesteps:
            ratio = (tokens == 8192).float().mean((1, 2))
            with autocast():
                logits = b(tokens, text, features, ratio).float()
            tokens = sched.step(logits, t, tokens, generator=gen).prev_sample
        assert tokens.max() < 8192
        images = vq.decode(tokens.reshape(-1), force_not_quantize=True, shape=(len(sub), 16, 16, 64)).sample.clamp(0, 1)
        assert torch.isfinite(images).all()
        for local, row in enumerate(sub):
            save_png(images[local], dest / f"{row:06d}.png")
            pixels = torch.from_numpy(data.arrays["images"][row].copy()).float() / 255
            save_png(pixels, references / f"{row:06d}.png")
        recovery.commit(start + len(sub), gen, previous_seconds + time.monotonic() - begin)
        if shutdown.requested:
            raise SystemExit(75)
    torch.cuda.synchronize()
    wall = time.monotonic() - begin
    validate_sample_manifest(rows, dest, references)
    grid = []
    for row in ids[:16]:
        with Image.open(dest / f"{row:06d}.png") as image:
            grid.append(torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 255)
    save_png(make_grid(torch.stack(grid), nrow=4), run / f"grid-{split}-g{guidance}.png")
    atomic_json(dest / "prompts.json", rows)
    return {"samples": len(ids), "seconds_per_image": wall / len(ids) if completed == 0 else None, "images_this_process": len(ids)-completed, "resumed_from_image": completed, "generation_seconds_completed_segments": previous_seconds+wall, "timing_scope": "durable image generation including PNG/fsync/progress writes; model loading and final grid excluded; resumed segments are not an uninterrupted latency measurement", "sample_dir": str(dest), "reference_dir": str(references), "b_checkpoint": bpath.name, "guidance": guidance}


def record(stage, cfg, metrics, wall, run, rc=0):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metrics.update({"peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "wall_seconds": wall, "wall_seconds_excluding_progress_images": wall - metrics.get("progress_images_seconds", 0.), "stage": stage, "code_revision": revision, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()})
    atomic_json(run / f"{stage}-{'summary' if rc == 0 else 'failure'}.json", metrics)
    path = ROOT / "results/measurements.csv"
    exists = path.exists()
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["time", "name", "seed", "stage", "config", "metrics", "rc", "wall", "state"])
    if not exists:
        writer.writeheader()
    writer.writerow({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "name": cfg["name"], "seed": cfg["seed"], "stage": stage, "config": json.dumps(cfg, sort_keys=True), "metrics": json.dumps(metrics, sort_keys=True), "rc": rc, "wall": wall, "state": "fresh process; warm frozen-component cache"})
    payload = (path.read_bytes() if exists else b"") + output.getvalue().encode()
    atomic_write(path, lambda stream: stream.write(payload))
    log = ROOT / "RESEARCH_LOG.md"
    with log.open("a") as f:
        f.write(f"\n## Measurement {cfg['name']}, seed {cfg['seed']}, {stage}\nCommand: `{' '.join(sys.argv)}`\n\nConfig and metrics: `{json.dumps(metrics, sort_keys=True)}`. Single-run evidence; no replicated improvement verdict.\n")
    text = log.read_text()
    text = re.sub(r"^- Updated:.*$", "- Updated: 2026-10-19, measurements appended below; consult latest measurement.", text, flags=re.M)
    text = re.sub(r"^- (?:Current numbers|Latest measurement):.*$", lambda _: f"- Latest measurement: {cfg['name']}, seed {cfg['seed']}, {stage}; {json.dumps(metrics, sort_keys=True)}", text, flags=re.M)
    next_stage = {"b": "features", "features": "c", "c": "generated", "generated": "finetune", "finetune": "generate", "generate": "quality", "quality": "replicated experiments and full reconstruction"}[stage]
    if rc:
        next_stage = f"diagnose failed {stage}; failure is not an algorithmic refutation"
    text = re.sub(r"^- Next action:.*$", lambda _: f"- Next action: {next_stage}, with configuration in runs/{run.name}/config.json.", text, flags=re.M)
    atomic_write(log, lambda stream: stream.write(text.encode()))
    subprocess.run(["git", "add", "RESEARCH_LOG.md", "results/measurements.csv"], cwd=ROOT, check=True)
    subprocess.run(["git", "-c", "user.name=Codex", "-c", "user.email=codex@local", "commit", "-m", f"{stage}: measured {cfg['name']} seed {cfg['seed']}"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/development.json")
    p.add_argument("--stage", required=True, choices=["b", "features", "c", "generated", "finetune", "generate", "quality"])
    p.add_argument("--run", default="development-17")
    p.add_argument("--seed", type=int)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--guidance", type=float, default=1.)
    args = p.parse_args()
    shutdown.install()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    lock = (ROOT / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert psutil.virtual_memory().available > 8 * 2**30, "Host memory floor"
    assert shutil.disk_usage(ROOT).free > 20 * 2**30, "Disk floor"
    assert torch.cuda.is_available(), "CUDA is required"
    free, total = torch.cuda.mem_get_info()
    assert free > 8 * 2**30, "VRAM floor: another heavy job may be active"
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    for line in processes.strip().splitlines():
        pid, memory = line.split(",")
        if int(pid) != os.getpid() and memory.strip().isdigit():
            assert int(memory) < 512, f"Another GPU workload is active: PID {pid} uses {memory} MiB"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(cfg["seed"])
    data = Cache(ROOT / cfg["cache"])
    run = ROOT / "runs" / args.run
    durable_mkdir(run)
    events = run / "events.jsonl"
    if events.exists():
        content = events.read_bytes()
        if content and not content.endswith(b"\n"):
            prefix = content[:content.rfind(b"\n") + 1]
            atomic_write(events, lambda stream: stream.write(prefix))
    old_config = run / "config.json"
    if old_config.exists():
        assert json.loads(old_config.read_text()) == cfg, "Use a new run directory for a changed config"
    atomic_json(old_config, cfg)
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    try:
        if args.stage in ["b", "finetune"]:
            result = train_b(data, run, cfg, args.stage)
        elif args.stage == "features":
            result = cache_features(data, run, cfg)
        elif args.stage == "c":
            result = train_c(data, run, cfg)
        elif args.stage == "generated":
            result = generate_feature_cache(data, run, cfg)
        elif args.stage == "generate":
            result = generate(data, run, cfg, args.count, args.split, args.guidance)
        else:
            from evaluate import evaluate
            result = evaluate(run, args.split, args.guidance)
        torch.cuda.synchronize()
    except Exception as error:
        record(args.stage, cfg, {"error": repr(error), "verdict": "failed run, not a refutation"}, time.monotonic() - begin, run, rc=1)
        raise
    record(args.stage, cfg, result, time.monotonic() - begin, run)


if __name__ == "__main__":
    main()
