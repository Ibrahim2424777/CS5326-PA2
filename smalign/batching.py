"""Tokenisation, truncation, padding and response-only label masking.

Every sequence is built as::

    input_ids = prompt_ids + response_ids + [eos]      (then cut to max_length)
    labels    = [-100] * len(prompt_ids) + response_ids + [eos]   (same cut)

* The prompt and the response are tokenised separately, so the prompt/response
  boundary is exact and prompt tokens can be masked by position.
* EOS is appended to the response *before* truncation (manual 4.1), so a
  response that hits ``max_length`` loses its EOS.
* Padding is on the right. Padding positions get ``attention_mask = 0`` and
  ``labels = -100``. Masks are derived from lengths, never from the pad id,
  because Qwen's pad token equals its EOS token.
* Row order is always preserved.
"""

from __future__ import annotations

from typing import Sequence

import torch

IGNORE_INDEX = -100


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(token_id) for token_id in ids]


def _eos_id(tokenizer):
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return None if eos is None else int(eos)


def pad_token_id(tokenizer) -> int:
    """Pad id, falling back to EOS and then 0 for tokenizers without a pad token."""

    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = _eos_id(tokenizer)
    return 0 if pad is None else int(pad)


def tokenize_pair(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int | None,
    max_prompt_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` for one prompt/response pair (unpadded).

    ``max_prompt_length`` (optional, used by DPO training) keeps only the last
    ``max_prompt_length`` prompt tokens so that the assistant header survives.
    """

    prompt_ids = _encode(tokenizer, prompt)
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length :]

    response_ids = _encode(tokenizer, response)
    eos = _eos_id(tokenizer)
    if eos is not None:
        response_ids = response_ids + [eos]

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return input_ids, labels


def pad_sequences(
    rows: Sequence[tuple[list[int], list[int]]], pad_id: int, width: int | None = None
) -> dict[str, torch.Tensor]:
    """Right-pad ``(input_ids, labels)`` rows into an LM batch."""

    width = max([width or 0] + [len(ids) for ids, _ in rows] + [1])
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in rows:
        n_pad = width - len(ids)
        input_ids.append(ids + [pad_id] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels.append(labs + [IGNORE_INDEX] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int | None,
    max_prompt_length: int | None = None,
    width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build ``{input_ids, attention_mask, labels}`` with response-only labels."""

    if len(prompts) != len(responses):
        raise ValueError(f"got {len(prompts)} prompts but {len(responses)} responses")
    rows = [
        tokenize_pair(tokenizer, prompt, response, max_length, max_prompt_length)
        for prompt, response in zip(prompts, responses)
    ]
    return pad_sequences(rows, pad_token_id(tokenizer), width=width)


def build_sft_batch(tokenizer, prompts, responses, max_length, max_prompt_length=None):
    """SFT batch: the prompt is masked with -100, the response (+EOS) is supervised."""

    return build_lm_batch(tokenizer, list(prompts), list(responses), max_length, max_prompt_length)


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
    max_prompt_length=None,
):
    """DPO batch: ``{"chosen": lm_batch, "rejected": lm_batch}`` sharing the same prompts.

    Both sides are padded to a common width so they can be concatenated into a
    single forward pass during training.
    """

    prompts = list(prompts)
    chosen_rows = [
        tokenize_pair(tokenizer, p, r, max_length, max_prompt_length)
        for p, r in zip(prompts, chosen_responses, strict=True)
    ]
    rejected_rows = [
        tokenize_pair(tokenizer, p, r, max_length, max_prompt_length)
        for p, r in zip(prompts, rejected_responses, strict=True)
    ]
    width = max(len(ids) for ids, _ in chosen_rows + rejected_rows) if prompts else 1
    pad_id = pad_token_id(tokenizer)
    return {
        "chosen": pad_sequences(chosen_rows, pad_id, width=width),
        "rejected": pad_sequences(rejected_rows, pad_id, width=width),
    }


def concat_lm_batches(first: dict, second: dict) -> dict[str, torch.Tensor]:
    """Stack two equal-width LM batches along the batch dimension."""

    return {key: torch.cat([first[key], second[key]], dim=0) for key in ("input_ids", "attention_mask", "labels")}
