"""Keep data corruption and conditioning randomness independent of architecture."""
import torch


def training_streams(cfg, phase, device="cuda"):
    if cfg.get("rng_protocol", 1) < 2:
        return {}
    phase_offset = {"b": 10000, "c": 20000, "finetune": 30000}[phase]
    offsets = {"mask": 101, "noise": 202, "time": 303, "dropout": 404, "mixture": 505}
    return {name: torch.Generator(device=device).manual_seed(cfg["seed"] + phase_offset + offset) for name, offset in offsets.items()}


def stream_states(streams):
    return {name: stream.get_state() for name, stream in streams.items()}


def restore_streams(streams, states):
    assert set(streams) == set(states), "Checkpoint random-stream protocol differs from configuration"
    for name, stream in streams.items():
        stream.set_state(states[name].cpu())
