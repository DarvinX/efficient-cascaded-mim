"""Durable local writes: flush payload, atomically publish, then sync directory."""
import json
import os
import pickle
from pathlib import Path
import tempfile
import torch


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path):
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        sync_directory(directory.parent)


def atomic_write(path, writer, before_publish=None):
    path = Path(path)
    durable_mkdir(path.parent)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if before_publish is not None:
            before_publish()
        os.replace(temp, path)
        sync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, lambda stream: stream.write((json.dumps(value, indent=2) + "\n").encode()))


def atomic_link(source, destination):
    source, destination = Path(source), Path(destination)
    temp = destination.with_name(f".{destination.name}.link-{os.getpid()}")
    temp.unlink(missing_ok=True)
    try:
        os.link(source, temp)
        os.replace(temp, destination)
        sync_directory(destination.parent)
    finally:
        temp.unlink(missing_ok=True)


def previous_checkpoint(path):
    return Path(str(path).replace("-latest.pt", "-previous.pt"))


def save_torch(path, value, keep_previous=False):
    path = Path(path)
    # Do not rotate the older backup until the new payload has finished syncing.
    # A failed write must leave two distinct old checkpoint generations intact.
    def rotate():
        if keep_previous and path.exists():
            atomic_link(path, previous_checkpoint(path))
    atomic_write(path, lambda stream: torch.save(value, stream), before_publish=rotate)


def checkpoint_exists(path):
    path = Path(path)
    return path.exists() or ("-latest.pt" in path.name and previous_checkpoint(path).exists())


def load_torch(path, **kwargs):
    path = Path(path)
    paths = [path]
    if "-latest.pt" in path.name:
        paths.append(previous_checkpoint(path))
    errors = []
    for candidate in paths:
        if not candidate.exists():
            continue
        try:
            value = torch.load(candidate, **kwargs)
            if candidate != path:
                print(f"Recovered intact previous checkpoint: {candidate}", flush=True)
                if path.exists():
                    os.replace(path, path.with_suffix(".corrupt"))
                    sync_directory(path.parent)
            return value
        except torch.cuda.OutOfMemoryError:
            raise
        except (EOFError, RuntimeError, OSError, ValueError, pickle.UnpicklingError) as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("No readable checkpoint: " + "; ".join(errors))


def flush_arrays(arrays, paths):
    for array in arrays:
        array.flush()
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_png(tensor, path):
    from torchvision.utils import save_image
    atomic_write(path, lambda stream: save_image(tensor, stream, format="PNG"))
