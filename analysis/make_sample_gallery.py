"""Lay out the first eight native grown validation samples without changing pixels."""
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def build():
    comparison = json.loads((ROOT / "results/comparison.json").read_text())
    rows = comparison["qualitative_samples"]
    assert len(rows) == 8
    tile, gap, margin = 256, 12, 16
    canvas = Image.new("RGB", (4 * tile + 3 * gap + 2 * margin,
                               2 * tile + gap + 2 * margin), "#101b2c")
    sources = []
    for index, row in enumerate(rows):
        path = ROOT / row["files"][2].replace(
            "analysis_inputs/converged-gilorat-17/", "evidence/grown-17/")
        with Image.open(path) as image:
            assert image.size == (tile, tile)
            original = image.convert("RGB")
            left = margin + (index % 4) * (tile + gap)
            top = margin + (index // 4) * (tile + gap)
            canvas.paste(original, (left, top))
            assert canvas.crop((left, top, left + tile, top + tile)).tobytes() == original.tobytes()
        sources.append({"source": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "sample": row})
    target = ROOT / "assets/generated-samples.png"
    canvas.save(target, optimize=True)
    record = {"selection": "First eight validation rows, same order as the comparison; no favorable selection.",
              "method": "Grown seed17; selected B8/C10 cascade; native RGB256 outputs.",
              "layout": "Two rows, four columns; each 256x256 RGB tile copied without resizing, cropping or pixel changes.",
              "sources": sources,
              "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "tile_pixel_identity": "pass"}
    (ROOT / "assets/generated-samples.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {target.name}: {target.stat().st_size:,} bytes; all eight tiles preserve native pixels.")


if __name__ == "__main__":
    build()
