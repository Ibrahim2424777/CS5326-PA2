"""Response log-probabilities, SFT loss and the from-scratch DPO objective.

Shift convention: the logit at position ``t`` predicts token ``t + 1``, so we
score ``logits[:, :-1]`` against ``labels[:, 1:]``. Positions whose label is
``-100`` (prompt and padding) contribute nothing.

SFT reduction (documented in the report): mean over all unmasked response
tokens in the batch, i.e. ``-sum(token logps) / num_response_tokens``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from smalign.batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch, concat_lm_batches


def _upcast(tensor) -> torch.Tensor:
    """fp16/bf16/int -> fp32; fp32/fp64 are kept so callers' dtypes are respected."""

    tensor = torch.as_tensor(tensor)
    if not tensor.is_floating_point() or tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    return tensor


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of response-token log-probs per row, plus the number of scored tokens.

    Computed in fp32 for numerical stability even when ``logits`` are fp16.
    """

    logits = _upcast(logits[:, :-1, :])
    labels = labels[:, 1:].to(logits.device)
    mask = labels.ne(IGNORE_INDEX)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.gather(logits, -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps - torch.logsumexp(logits, dim=-1)
    token_logps = token_logps * mask
    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def response_logprobs_from_batch(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` on an LM batch and return ``(summed response logps, token counts)``."""

    device = model_device(model)
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    return sequence_logprobs(outputs.logits, batch["labels"].to(device))


def sft_loss_from_batch(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-mean response NLL for one batch; also returns the token count."""

    logps, counts = response_logprobs_from_batch(model, batch)
    n_tokens = counts.sum()
    return -logps.sum() / n_tokens.clamp_min(1), n_tokens


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    """One summed response-token log-probability per input row (row order preserved)."""

    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps, _ = response_logprobs_from_batch(model, batch)
    return logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    loss, _ = sft_loss_from_batch(model, batch)
    return loss


# --------------------------------------------------------------------------- DPO


def dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Batch-mean of ``-log sigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))``."""

    policy_chosen_logps = _upcast(policy_chosen_logps)
    device = policy_chosen_logps.device
    policy_gap = policy_chosen_logps - _upcast(policy_rejected_logps).to(device)
    reference_gap = _upcast(reference_chosen_logps).to(device) - _upcast(reference_rejected_logps).to(device)
    return -F.logsigmoid(beta * (policy_gap - reference_gap)).mean()


@torch.no_grad()
def dpo_metrics(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Per-example implicit rewards (manual eqs. 7-10), not reduced."""

    policy_chosen_logps = _upcast(policy_chosen_logps)
    device = policy_chosen_logps.device
    reward_chosen = beta * (policy_chosen_logps - _upcast(reference_chosen_logps).to(device))
    reward_rejected = beta * (_upcast(policy_rejected_logps).to(device) - _upcast(reference_rejected_logps).to(device))
    margin = reward_chosen - reward_rejected
    return {
        "reward_chosen": reward_chosen,
        "reward_rejected": reward_rejected,
        "reward_margin": margin,
        "reward_accuracy": (reward_chosen > reward_rejected).float(),  # strict, so ties count as 0
        "loss": -F.logsigmoid(margin),
    }


def policy_dpo_logps(policy_model, dpo_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Chosen and rejected logps from one forward pass over the concatenated batch."""

    n = dpo_batch["chosen"]["input_ids"].shape[0]
    logps, _ = response_logprobs_from_batch(policy_model, concat_lm_batches(dpo_batch["chosen"], dpo_batch["rejected"]))
    return logps[:n], logps[n:]


@torch.no_grad()
def reference_dpo_logps(reference_model, dpo_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as :func:`policy_dpo_logps` but gradient-free, for the frozen reference."""

    return policy_dpo_logps(reference_model, dpo_batch)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
    max_prompt_length=None,
) -> torch.Tensor:
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, max_prompt_length)
    policy_chosen, policy_rejected = policy_dpo_logps(policy_model, batch)
    reference_chosen, reference_rejected = reference_dpo_logps(reference_model, batch)
    return dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
