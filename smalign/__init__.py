"""smalign: small-model alignment (SFT + from-scratch DPO) for CS5326 PA2.

Module map:
    prompting   -- the single prompt template used for SFT, DPO and evaluation
    batching    -- tokenisation, truncation, padding and response-only label masks
    losses      -- response log-probabilities, SFT loss, DPO loss and DPO metrics
    data        -- dataset download / filtering / formatting
    utils       -- seeding, JSON(L) IO, metric logging, device helpers
"""

from smalign.batching import IGNORE_INDEX, build_dpo_batch, build_lm_batch, build_sft_batch
from smalign.losses import (
    compute_dpo_loss,
    compute_response_logprobs,
    compute_sft_loss,
    dpo_loss_from_logps,
    dpo_metrics,
    response_logprobs_from_batch,
    sequence_logprobs,
)
from smalign.prompting import format_prompt

__all__ = [
    "IGNORE_INDEX",
    "build_dpo_batch",
    "build_lm_batch",
    "build_sft_batch",
    "compute_dpo_loss",
    "compute_response_logprobs",
    "compute_sft_loss",
    "dpo_loss_from_logps",
    "dpo_metrics",
    "format_prompt",
    "response_logprobs_from_batch",
    "sequence_logprobs",
]
