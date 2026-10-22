"""Publication-style dense versus grown comparisons from audited saved evidence.

Both arms use training seed17 and the same2048 validation prompts. Training
budgets differ. No rank2 arm, new inference, test use or timing measurement.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
import tempfile

def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = ROOT / "figures-rebuilt"
COLORS = {"Dense": "#28618c", "Grown": "#b66138"}
INK, MUTED = "#202124", "#575b60"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finish(fig, name, title, caption, bottom=.20, top=.75):
    fig.suptitle(title, x=.075, y=.975, ha="left", fontsize=17, weight="bold")
    fig.text(.075, .87, "Training seed 17; identical 2,048-image validation set. Grown: B rank 8, C rank 10, fine-tuned B fixed at rank 8.", fontsize=9, color=MUTED)
    fig.text(.075, .045, caption, fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=.075, right=.975, top=top, bottom=bottom, wspace=.36, hspace=.60)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=240, metadata={"Date": None} if extension == "svg" else None)
    plt.close(fig)


def metric_plot(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, metric, label in zip(axes, ("clean_fid", "kid", "clip_cosine_mean"),
                                ("(a) FID ↓", "(b) KID ↓", "(c) CLIP cosine ↑")):
        ax.set_title(label, loc="left", fontsize=12, pad=12)
        for method, marker in (("Dense", "o"), ("Grown", "s")):
            row = rows[method]
            x, y = row["generator_weights"] / 1e6, row[metric]
            ax.scatter(x, y, s=65, marker=marker, color=COLORS[method], edgecolor="white", linewidth=.6,
                       label=f"{method} ({x:.2f}M)", zorder=3)
            ax.annotate(f"{y:.2f}" if metric == "clean_fid" else f"{y:.4f}", (x, y),
                        xytext=(8 if method == "Grown" else -8, 12), textcoords="offset points",
                        ha="left" if method == "Grown" else "right", fontsize=11, color=COLORS[method])
        ax.set_xscale("log")
        ax.set_xlim(24, 520)
        ax.xaxis.set_major_locator(FixedLocator([30, 100, 400]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlabel("Learned generator parameters (M; log scale)", fontsize=9, labelpad=10)
        ax.grid(alpha=.25)
        if metric == "clean_fid":
            ax.set_ylim(45, 120)
        elif metric == "kid":
            ax.set_ylim(.05, .135)
        else:
            ax.set_ylim(.339, .349)
    axes[0].legend(loc="lower left", fontsize=9)
    finish(fig, "01_metrics_vs_parameters", "Validation metrics versus learned generator parameters",
           "Parameters count selected fine-tuned B + C, including dense boundary layers; shared frozen VQ/text weights are excluded. Unequal training budgets; one seed per arm.")


def resource_plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    phases, labels, x, width = ("b", "c", "finetune"), ("Joint B", "Diffusion C", "B fine-tuning"), np.arange(3), .34
    for method, offset in (("Dense", -.18), ("Grown", .18)):
        r = rows[method]
        bars = axes[0].bar(x + offset, [r["phase_parameters"][p] / 1e6 for p in phases], width,
                           color=COLORS[method], label=method, zorder=3)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        gen = [r["generation_peak_allocated_gib"], r["generation_peak_reserved_gib"]]
        bars = axes[1].bar(np.arange(2) + offset, gen, width, color=COLORS[method], label=method, zorder=3)
        axes[1].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    axes[0].set_title("(a) Trainable parameters by training phase", loc="left", pad=15)
    axes[0].set_ylabel("Parameters (M)")
    axes[0].set_ylim(0, 310)
    axes[0].set_xticks(x, labels)
    axes[0].legend(fontsize=9)
    axes[1].set_title("(b) Complete-generation peak GPU memory", loc="left", pad=15)
    axes[1].set_ylabel("PyTorch memory (GiB)")
    axes[1].set_xticks([0, 1], ["Allocated", "Reserved"])
    axes[1].set_ylim(0, 5.9)
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.25, zorder=0)
    finish(fig, "02_resources", "Model resources: dense versus grown",
           "Shared frozen assets excluded in (a). Memory in (b) covers complete generation, including frozen components. Unequal training budgets; one seed per arm.",
           bottom=.20, top=.73)


def samples_plot(dense, grown, sources):
    prompts = read(dense / "samples-val-g1.0/prompts.json")
    assert prompts == read(grown / "samples-val-g1.0/prompts.json")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    font = lambda size: ImageFont.truetype(font_path, size)
    width, top, rowheight, left, tile, gap = 1120, 170, 218, 390, 208, 20
    canvas = Image.new("RGB", (width, top + 8 * rowheight + 110), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 25), "Qualitative comparison on validation prompts", font=font(29), fill=INK)
    draw.text((38, 76), "First eight validation rows, without sample selection; training seed 17; unequal training budgets.", font=font(15), fill=MUTED)
    for index, label in enumerate(("Reference", "Dense", "Grown (B8 / C10)")):
        draw.text((left + index * (tile + gap), 125), label, font=font(18), fill=INK)
    identities = []
    for index, row in enumerate(prompts[:8]):
        y = top + index * rowheight
        filename = f"{row['row_id']:06d}.png"
        paths = [dense / "references-val" / filename, dense / "samples-val-g1.0" / filename, grown / "samples-val-g1.0" / filename]
        assert sha(paths[0]) == sha(grown / "references-val" / filename)
        draw.text((38, y + 8), f"Validation row {row['row_id']}", font=font(16), fill=INK)
        lines = textwrap.wrap(row["caption"], 35)
        assert len(lines) * 23 < tile - 30
        draw.multiline_text((38, y + 41), "\n".join(lines), font=font(14), fill=MUTED, spacing=5)
        for column, path in enumerate(paths):
            sources[str(path.relative_to(ROOT))] = sha(path)
            with Image.open(path) as image:
                assert image.mode == "RGB" and image.size == (256, 256)
                canvas.paste(image.resize((tile, tile), Image.Resampling.NEAREST), (left + column * (tile + gap), y))
        identities.append({"row_id": row["row_id"], "caption": row["caption"], "files": [str(p.relative_to(ROOT)) for p in paths]})
    draw.text((38, canvas.height - 68), "Native final evaluation draws; common captions and references, without paired common sampling noise.", font=font(14), fill=MUTED)
    draw.text((38, canvas.height - 40), "References are used for evaluation only. Display thumbnails are derived from retained 256×256 PNG masters.", font=font(14), fill=MUTED)
    canvas.save(OUT / "03_sample_comparison.png")
    return identities


if __name__ == "__main__":
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#a6a6a6", "text.color": INK, "svg.fonttype": "none"})
    old_path = ROOT / "analysis/inputs/dense.json"
    new_path = ROOT / "analysis/inputs/grown.json"
    old = next(r for r in read(old_path)["runs"] if r["method"] == "Dense" and r["seed"] == 17)
    new = read(new_path)
    dense, grown = ROOT / "evidence/dense-17", ROOT / "evidence/grown-17"
    old_generation = read(dense / "generate-summary.json")
    new_generation = read(grown / "generate-summary.json")
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (old_path, new_path, Path(__file__),
        dense / "generate-summary.json", grown / "generate-summary.json", dense / "samples-val-g1.0/prompts.json", grown / "samples-val-g1.0/prompts.json")}
    rows = {
        "Dense": {**{k: old[k] for k in ("clean_fid", "kid", "clip_cosine_mean")},
                  "phase_parameters": old["phase_trainable_parameters"],
                  "phase_training_peak_allocated_gib": {p: old["phase_peak_allocated_cuda_gib"][p] for p in ("b", "c", "finetune")},
                  "selected_updates": {}},
        "Grown": {**new["primary_metrics"],
                  "phase_parameters": {p: r["trainable_parameters"] for p, r in new["selected_phases"].items()},
                  "phase_training_peak_allocated_gib": {p: None for p in ("b", "c", "finetune")},
                  "selected_updates": {p: r["step"] for p, r in new["selected_phases"].items()}}}
    for phase in ("b", "c", "finetune"):
        summary = dense / f"{phase}-summary.json"
        sources[str(summary.relative_to(ROOT))] = sha(summary)
        # Native summaries expose selected steps via their saved validation journal.
        events = [json.loads(x) for x in (dense / "events.jsonl").read_text().splitlines()]
        key = "val_mse" if phase == "c" else "val_ce"
        candidates = [r for r in events if r.get("stage") == phase and key in r]
        selected = min(candidates, key=lambda r: r[key])
        rows["Dense"]["selected_updates"][phase] = selected["step"]
    sources[str((dense / "events.jsonl").relative_to(ROOT))] = sha(dense / "events.jsonl")
    for method, base in (("Dense", dense), ("Grown", grown)):
        path = base / "events.jsonl"
        journal = [json.loads(line) for line in path.read_text().splitlines()]
        rows[method]["completed_updates"] = {p: max(r.get("step", 0) for r in journal if r.get("stage") == p) for p in ("b", "c", "finetune")}
        sources[str(path.relative_to(ROOT))] = sha(path)
    for method, gen in (("Dense", old_generation), ("Grown", new_generation)):
        r = rows[method]
        r["generator_weights"] = r["phase_parameters"]["finetune"] + r["phase_parameters"]["c"]
        r["generation_peak_allocated_gib"] = gen["peak_cuda_gib"]
        r["generation_peak_reserved_gib"] = gen["peak_reserved_gib"]
    metric_plot(rows)
    resource_plot(rows)
    samples = samples_plot(dense, grown, sources)
    atomic_json(OUT / "data.json", {"created_unix": time.time(), "seed": 17, "n": 2048, "split": "val",
        "rows": rows, "qualitative_samples": samples,
        "scope": "Dense versus actual grown B8/C10 with fixed-rank8 FT only. Both training seed17; same validation rows/references. Different training budgets and independently sampled native final images. Shared frozen VQ/text excluded from learned weights; EfficientNet included only during joint B. Grown full-training peak memory unavailable; no new timing, speed, causal growth or replicated superiority claim. All grown training phases explicitly user-ended before plateau."})
    atomic_json(OUT / "manifest.json", {"status": "rendered_pending_visual_QA", "source_sha256": sources,
        "artifacts_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}})
    print("Rendered dense-versus-grown metrics, resources and sample comparison; visual QA pending.", flush=True)
