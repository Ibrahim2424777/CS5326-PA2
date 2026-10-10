"""Shared pieces of the hand-written training loops (no HF Trainer / TRL)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from smalign.utils import read_jsonl, write_json


def load_split(path, limit=None) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"no rows in {path}; run scripts/prepare_data.py first")
    return rows[:limit] if limit else rows


def batches(rows: list[dict], batch_size: int, shuffle: bool, seed: int):
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start : start + batch_size]]


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def decay_param_groups(model, weight_decay: float) -> list[dict]:
    """AdamW groups: no weight decay on biases and normalisation weights."""

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or "norm" in name.lower() else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_lambda(schedule: str, warmup_steps: int, total_steps: int):
    """Multiplier on the peak LR: linear warmup, then constant or cosine decay to 0."""

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def save_checkpoint(model, tokenizer, out_dir, state: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    write_json(out_dir / "checkpoint_state.json", state)


def load_causal_lm(path, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM

    # torch_dtype is accepted by every transformers<5 release (newer ones also take dtype=)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    model.config.use_cache = False
    return model.to(device)


def load_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def dump_config(config: dict, out_dir) -> None:
    write_json(Path(out_dir) / "run_config.json", json.loads(json.dumps(config, default=str)))
