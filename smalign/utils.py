"""Small shared helpers: seeding, JSON/JSONL IO, configs, metric logging."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic, so a disconnect never leaves a half-written file


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _coerce(value):
    """YAML 1.1 reads ``1e-3`` (no dot) as a string; turn numeric-looking strings into numbers."""

    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() and "e" not in value.lower() and "." not in value else number
    return value


def load_config(path, overrides: dict | None = None) -> dict:
    """Load a YAML config and apply non-None CLI overrides."""

    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    return {key: _coerce(value) for key, value in config.items()}


def parse_overrides(pairs: list[str] | None) -> dict:
    """Parse ``key=value`` CLI overrides, interpreting values as YAML scalars."""

    import yaml

    overrides = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


class MetricsLogger:
    """Appends one JSON line per event to ``metrics.jsonl`` so partial runs survive."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, split: str, step: int, **values) -> dict:
        record = {"split": split, "step": step, "elapsed_sec": round(time.time() - self.start, 1)}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            record[key] = value
        append_jsonl(self.path, record)
        return record


def gpu_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    parts = []
    for index in range(torch.cuda.device_count()):
        allocated = torch.cuda.max_memory_allocated(index) / 2**30
        parts.append(f"cuda:{index} peak {allocated:.2f} GiB")
    return ", ".join(parts)


def resolve_device(name: str | None, fallback: str = "cuda:0") -> torch.device:
    """Map a requested device to one that exists (cuda:1 -> cuda:0 on a single GPU, cpu without GPUs)."""

    name = name or fallback
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        index = int(name.split(":")[1]) if ":" in name else 0
        return torch.device(f"cuda:{min(index, torch.cuda.device_count() - 1)}")
    return torch.device(name)
