"""Prompt template shared by SFT, DPO and evaluation.

We use a ChatML-style single-turn template written out explicitly (instead of
``tokenizer.apply_chat_template``) so that it is identical everywhere and does
not inject Qwen's default system prompt:

    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<EOS>

``<|im_start|>`` / ``<|im_end|>`` are existing special tokens in the Qwen2.5
vocabulary. The response is terminated with ``tokenizer.eos_token``
(``<|endoftext|>`` for Qwen2.5-0.5B base), which the batching code appends.
At generation time both ``<|endoftext|>`` and ``<|im_end|>`` are stop tokens.
"""

from __future__ import annotations

TEMPLATE_NAME = "chatml_no_system"
USER_PREFIX = "<|im_start|>user\n"
USER_SUFFIX = "<|im_end|>\n"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


def format_prompt(instruction: str) -> str:
    """Return the model-facing prompt for one user instruction."""

    return f"{USER_PREFIX}{instruction.strip()}{USER_SUFFIX}{ASSISTANT_PREFIX}"


def stop_token_ids(tokenizer) -> list[int]:
    """Token ids that should end generation for models trained with this template."""

    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for text in STOP_STRINGS:
        token_id = tokenizer.convert_tokens_to_ids(text)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id and token_id not in ids:
            ids.append(token_id)
    return ids
