"""Durable completed-image boundaries and sampler RNG for final generation."""
import hashlib
import json
from pathlib import Path

from persistence import atomic_json, checkpoint_exists, durable_mkdir, load_torch, save_torch
from sample_manifest import validate_sample_manifest


class GenerationRecovery:
    def __init__(self, generated, references, rows, settings):
        self.generated, self.references = Path(generated), Path(references)
        self.rows = rows
        for folder in (self.generated, self.references):
            durable_mkdir(folder)
        validate_sample_manifest(rows, self.generated, self.references, complete=False)
        self.contract = {"settings": settings, "rows": rows}
        self.digest = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()
        contract_path = self.generated / ".generation-contract.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != self.contract:
                raise ValueError("Generation inputs/count/checkpoints changed; use a fresh run directory")
        else:
            if any(self.generated.glob("*.png")):
                raise ValueError("Existing generated images have no bound recovery contract; preserve them and use a fresh run directory")
            atomic_json(contract_path, self.contract)
        self.path = self.generated / ".generation-latest.pt"
        self.completed, self.elapsed_seconds = 0, 0.

    def _check_files(self, start, end):
        for row in self.rows[start:end]:
            name = f"{row['row_id']:06d}.png"
            if not (self.generated / name).is_file() or not (self.references / name).is_file():
                raise ValueError(f"Generation checkpoint would reference a missing image: {name}")

    def restore(self, generator):
        if checkpoint_exists(self.path):
            saved = load_torch(self.path, map_location="cpu", weights_only=False)
            if saved["contract_sha256"] != self.digest:
                raise ValueError("Generation checkpoint contract mismatch")
            self.completed = int(saved["completed"])
            if not 0 <= self.completed <= len(self.rows):
                raise ValueError("Invalid generation checkpoint boundary")
            self._check_files(0, self.completed)
            generator.set_state(saved["generator_state"].cpu())
            self.elapsed_seconds = float(saved["elapsed_seconds"])
        return self.completed, self.elapsed_seconds

    def commit(self, completed, generator, elapsed_seconds):
        if not self.completed <= completed <= len(self.rows):
            raise ValueError("Invalid generation progress transition")
        self._check_files(self.completed, completed)
        save_torch(self.path, {"contract_sha256": self.digest, "completed": completed, "generator_state": generator.get_state(), "elapsed_seconds": elapsed_seconds}, keep_previous=True)
        self.completed, self.elapsed_seconds = completed, elapsed_seconds
