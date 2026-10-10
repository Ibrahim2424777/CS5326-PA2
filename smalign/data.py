"""Dataset download, filtering and formatting.

All prepared splits are written as JSONL rows with plain-text fields:

* SFT:  ``{"id", "prompt", "response"}``
* DPO:  ``{"id", "prompt", "chosen", "rejected"}``

``prompt`` here is the *raw user instruction*; the chat template from
``smalign.prompting`` is applied by the training scripts.

Filtering (identical for every track, all documented in the report):
* single-turn only (one user turn followed by one assistant turn);
* non-empty instruction / responses, and ``chosen != rejected``;
* formatted prompt <= ``max_prompt_tokens`` tokens (256), so that at least
  256 of the 512-token context is left for the response;
* UltraFeedback: drop pairs whose chosen and rejected scores are tied.
Sampling is a seeded shuffle (seed 42) followed by taking the first N rows
that pass the filters.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable

from smalign.prompting import format_prompt

HH_HUMAN = "\n\nHuman: "
HH_ASSISTANT = "\n\nAssistant: "


# ----------------------------------------------------------------------------- parsers


def parse_ultrachat(row: dict) -> dict | None:
    messages = row.get("messages") or []
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    prompt, response = messages[0]["content"].strip(), messages[1]["content"].strip()
    if not prompt or not response:
        return None
    return {"id": row.get("prompt_id"), "prompt": prompt, "response": response}


def _split_hh(text: str) -> tuple[str, str] | None:
    """Split a single-turn HH transcript into (human turn, assistant turn)."""

    if text.count("Human:") != 1 or text.count("Assistant:") != 1:
        return None
    if not text.startswith(HH_HUMAN) or HH_ASSISTANT not in text:
        return None
    human, assistant = text[len(HH_HUMAN) :].split(HH_ASSISTANT, 1)
    return human.strip(), assistant.strip()


def parse_hh_rlhf(row: dict) -> dict | None:
    chosen, rejected = _split_hh(row["chosen"]), _split_hh(row["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    prompt = chosen[0]
    if not prompt or not chosen[1] or not rejected[1] or chosen[1] == rejected[1]:
        return None
    return {"prompt": prompt, "chosen": chosen[1], "rejected": rejected[1]}


def parse_ultrafeedback(row: dict) -> dict | None:
    chosen, rejected = row.get("chosen") or [], row.get("rejected") or []
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if chosen[0]["role"] != "user" or chosen[1]["role"] != "assistant" or rejected[1]["role"] != "assistant":
        return None
    if row.get("score_chosen") is not None and row.get("score_chosen") == row.get("score_rejected"):
        return None
    prompt = (row.get("prompt") or chosen[0]["content"]).strip()
    chosen_text, rejected_text = chosen[1]["content"].strip(), rejected[1]["content"].strip()
    if not prompt or not chosen_text or not rejected_text or chosen_text == rejected_text:
        return None
    return {"id": row.get("prompt_id"), "prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}


# ----------------------------------------------------------------------------- sampling


def sample_rows(
    dataset,
    parser: Callable[[dict], dict | None],
    n: int,
    seed: int,
    tokenizer=None,
    max_prompt_tokens: int | None = None,
    id_prefix: str = "",
    exclude_prompts: Iterable[str] = (),
) -> tuple[list[dict], dict]:
    """Seeded shuffle, then keep the first ``n`` rows that parse and pass the filters."""

    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    excluded = set(exclude_prompts)
    kept, stats = [], {"scanned": 0, "parse_rejected": 0, "too_long": 0, "duplicate": 0}
    seen_prompts: set[str] = set()
    chunk = 512
    for start in range(0, len(order), chunk):
        indices = order[start : start + chunk]
        parsed = []
        for index in indices:
            stats["scanned"] += 1
            row = parser(dataset[index])
            if row is None:
                stats["parse_rejected"] += 1
                continue
            if row["prompt"] in seen_prompts or row["prompt"] in excluded:
                stats["duplicate"] += 1
                continue
            row["source_index"] = index
            parsed.append(row)
        if tokenizer is not None and max_prompt_tokens is not None and parsed:
            lengths = [len(ids) for ids in tokenizer([format_prompt(r["prompt"]) for r in parsed], add_special_tokens=False)["input_ids"]]
        else:
            lengths = [0] * len(parsed)
        for row, length in zip(parsed, lengths):
            if max_prompt_tokens is not None and length > max_prompt_tokens:
                stats["too_long"] += 1
                continue
            if row["prompt"] in seen_prompts:
                stats["duplicate"] += 1
                continue
            seen_prompts.add(row["prompt"])
            row["id"] = f"{id_prefix}{len(kept):05d}"
            kept.append(row)
            if len(kept) >= n:
                stats["kept"] = len(kept)
                return kept, stats
    stats["kept"] = len(kept)
    return kept, stats


# ----------------------------------------------------------------------------- AlpacaEval


def alpaca_eval_subset(n: int = 300, seed: int = 42) -> list[dict]:
    """Fixed AlpacaEval subset: ``random.Random(seed).sample`` of the 805 prompts, kept in original order."""

    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return subset_from_alpaca_rows(rows, n, seed)


def subset_from_alpaca_rows(rows: list[dict], n: int = 300, seed: int = 42) -> list[dict]:
    indices = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [
        {
            "id": f"alpaca-eval-{seed}-{position:04d}",
            "instruction": rows[index]["instruction"],
            "dataset": "alpaca_eval",
            "category": rows[index]["dataset"],
            "source_index": index,
        }
        for position, index in enumerate(indices)
    ]
