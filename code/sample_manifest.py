"""Bind image counts to prompts before a folder-based quality metric reads them."""
from glob import glob
from pathlib import Path

SAMPLE_EXTENSIONS = frozenset({"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp", "npy", "JPEG", "JPG", "PNG"})

def validate_sample_manifest(rows, generated, references, extensions=SAMPLE_EXTENSIONS, *, complete=True):
    names = [f"{row['row_id']:06d}.png" for row in rows]
    expected = set(names)
    if len(expected) != len(names):
        raise ValueError("Duplicate image IDs in prompts.json")
    for label, folder in [("generated", Path(generated)), ("reference", Path(references))]:
        # Match clean-fid's recursive glob and exact supported extensions,
        # including JPG/PNG/npy; an extra non-PNG can also contaminate its score.
        actual = {str(Path(file).relative_to(folder)) for ext in extensions for file in glob(str(folder / f"**/*.{ext}"), recursive=True)}
        extra = actual - expected
        missing = expected - actual if complete else set()
        if extra or missing:
            raise ValueError(f"{label} images do not match prompts.json: {len(extra)} extra, {len(missing)} missing; use a fresh generation directory for different sample counts")
    return sorted(expected)
