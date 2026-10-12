"""smalign: small-model alignment (SFT + from-scratch DPO) for CS5326 PA2.

Module map:
    prompting   -- the single prompt template used for SFT, DPO and evaluation
    batching    -- tokenisation, truncation, padding and response-only label masks
    losses      -- response log-probabilities, SFT loss, DPO loss and DPO metrics
    data        -- dataset download / filtering / formatting
    judge       -- judge prompts, verdict parsing and win/loss summaries
    utils       -- seeding, JSON(L) IO, metric logging, device helpers

The torch-based names below are resolved lazily, so ``smalign.judge`` / ``smalign.utils`` /
``smalign.notebook`` can be imported where PyTorch is not installed (Kaggle Benchmarks notebooks).
"""

from importlib import import_module

_LAZY = {
    "IGNORE_INDEX": "batching",
    "build_dpo_batch": "batching",
    "build_lm_batch": "batching",
    "build_sft_batch": "batching",
    "compute_dpo_loss": "losses",
    "compute_response_logprobs": "losses",
    "compute_sft_loss": "losses",
    "dpo_loss_from_logps": "losses",
    "dpo_metrics": "losses",
    "response_logprobs_from_batch": "losses",
    "sequence_logprobs": "losses",
    "format_prompt": "prompting",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(f"smalign.{_LAZY[name]}"), name)
    raise AttributeError(f"module 'smalign' has no attribute {name!r}")
