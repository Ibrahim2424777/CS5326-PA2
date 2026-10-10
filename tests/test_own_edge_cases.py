"""Extra checks beyond the public smoke tests (manual 4.2): truncation boundaries,
unusual tokenizer configurations, mixed lengths, row ordering, exact numerics and
small optimisation steps."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None  # e.g. Llama-style tokenizers with no pad token configured


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


def _manual_logp(model, tokenizer, prompt, response, max_length):
    """Brute-force reference: score each response token one at a time."""

    ids = tokenizer.encode(prompt) + tokenizer.encode(response) + [tokenizer.eos_token_id]
    n_prompt = len(tokenizer.encode(prompt))
    ids = ids[:max_length]
    logits = model(torch.tensor([ids])).logits[0]
    total = 0.0
    for position in range(max(n_prompt, 1), len(ids)):
        total += torch.log_softmax(logits[position - 1].double(), -1)[ids[position]].item()
    return total


def test_eos_is_dropped_when_response_is_truncated():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab"], ["cdef"], max_length=4)
    assert batch["input_ids"].tolist() == [tok.encode("abcd")]
    assert batch["labels"].tolist() == [[-100, -100] + tok.encode("cd")]
    assert tok.eos_token_id not in batch["labels"][0].tolist()


def test_eos_kept_at_exact_boundary():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab"], ["cd"], max_length=5)
    assert batch["input_ids"].shape == (1, 5)
    assert batch["labels"][0, -1].item() == tok.eos_token_id


def test_prompt_longer_than_max_length_gives_no_response_tokens():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["abcdefgh", "a"], ["xy", "xy"], max_length=6)
    assert batch["labels"][0].eq(-100).all()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefgh", "a"], ["xy", "xy"], max_length=6)
    assert logps[0].item() == 0.0
    assert logps[1].item() < 0.0
    loss = adapter.compute_sft_loss(model, tok, ["abcdefgh", "a"], ["xy", "xy"], max_length=6)
    assert torch.isfinite(loss)


def test_tokenizer_without_pad_token_pads_with_eos_and_masks_by_length():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["a", "abc"], ["b", "b"], max_length=10)
    assert batch["input_ids"][0, -2:].tolist() == [tok.eos_token_id, tok.eos_token_id]
    # Row 0 = a b <eos> <pad=eos> <pad=eos>; only the real EOS is a label.
    assert batch["labels"][0].tolist() == [-100, tok.encode("b")[0], tok.eos_token_id, -100, -100]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]


def test_tokenizer_without_eos_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["a"], ["bc"], max_length=10)
    assert batch["input_ids"].tolist() == [tok.encode("abc")]
    assert batch["labels"].tolist() == [[-100] + tok.encode("bc")]


def test_logprobs_match_brute_force_with_mixed_lengths():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.3)
    prompts = ["Q: ", "A much longer prompt: ", ""]
    responses = ["hi", "x", "empty prompt"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=12)
    expected = [_manual_logp(model, tok, p, r, 12) for p, r in zip(prompts, responses)]
    torch.testing.assert_close(got.double(), torch.tensor(expected, dtype=torch.float64), rtol=1e-5, atol=1e-5)


def test_row_order_is_preserved():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts = ["a: ", "bb: ", "ccc: "]
    responses = ["xyz", "q", "hello"]
    forward = adapter.compute_response_logprobs(model, tok, prompts, responses, 16)
    backward = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], 16)
    torch.testing.assert_close(forward, backward.flip(0))
    dpo = adapter.build_dpo_batch(tok, prompts, responses, responses[::-1], 16)
    single = adapter.build_sft_batch(tok, prompts[1:2], responses[1:2], 16)
    width = single["input_ids"].shape[1]
    assert dpo["chosen"]["input_ids"][1, :width].tolist() == single["input_ids"][0].tolist()


def test_dpo_batch_shares_prompt_masking_on_both_sides():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["Task: "], ["good"], ["bad answer"], max_length=32)
    n_prompt = len(tok.encode("Task: "))
    for side, text in (("chosen", "good"), ("rejected", "bad answer")):
        labels = batch[side]["labels"][0]
        assert labels[:n_prompt].eq(-100).all()
        assert labels[n_prompt : n_prompt + len(text) + 1].tolist() == tok.encode(text) + [tok.eos_token_id]
        assert batch[side]["input_ids"][0, :n_prompt].tolist() == tok.encode("Task: ")


def test_dpo_loss_from_logps_matches_formula_and_keeps_float64():
    pc = torch.tensor([-1.0, -4.0, -2.5], dtype=torch.float64)
    pr = torch.tensor([-3.0, -1.0, -2.5], dtype=torch.float64)
    rc = torch.tensor([-2.0, -3.0, -1.0], dtype=torch.float64)
    rr = torch.tensor([-2.0, -2.0, -4.0], dtype=torch.float64)
    beta = 0.1
    expected = sum(
        -math.log(1 / (1 + math.exp(-beta * ((a - b) - (c - d)))))
        for a, b, c, d in zip(pc.tolist(), pr.tolist(), rc.tolist(), rr.tolist())
    ) / 3
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    assert loss.dtype == torch.float64
    assert loss.ndim == 0
    assert abs(loss.item() - expected) < 1e-12


def test_dpo_loss_is_stable_for_large_gaps():
    big = torch.tensor([1e4])
    zero = torch.tensor([0.0])
    assert torch.isfinite(adapter.compute_dpo_loss_from_logps(zero, big, zero, zero, 1.0))
    assert adapter.compute_dpo_loss_from_logps(big, zero, zero, zero, 1.0).item() == pytest.approx(0.0, abs=1e-6)


def test_dpo_loss_matches_logp_composition():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=-0.5)
    prompts, chosen, rejected = ["P: ", "Longer P: "], ["yes", "a"], ["no!", "bbbb"]
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.3, max_length=14)
    expected = adapter.compute_dpo_loss_from_logps(
        adapter.compute_response_logprobs(policy, tok, prompts, chosen, 14),
        adapter.compute_response_logprobs(policy, tok, prompts, rejected, 14),
        adapter.compute_response_logprobs(reference, tok, prompts, chosen, 14),
        adapter.compute_response_logprobs(reference, tok, prompts, rejected, 14),
        0.3,
    )
    torch.testing.assert_close(loss, expected)


def test_identical_policy_and_reference_give_log_two():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["x: "], ["abc"], ["d"], beta=0.1, max_length=10)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))


def test_small_dpo_steps_increase_preference_and_leave_reference_unchanged():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size)
    reference = ToyCausalLM(tok.vocab_size)
    reference_before = reference.logit_bias.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    args = (tok, ["Pick: "], ["aaa"], ["bbb"])
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = adapter.compute_dpo_loss(policy, reference, *args, beta=0.5, max_length=12)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] < math.log(2) + 1e-6
    assert torch.equal(reference.logit_bias.detach(), reference_before)
    assert reference.logit_bias.grad is None


def test_small_sft_steps_reduce_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = adapter.compute_sft_loss(model, tok, ["Q: ", "Say: "], ["aaaa", "ab"], max_length=16)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_sft_loss_is_token_mean_and_ignores_prompt_logits():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.7)
    prompts, responses = ["Do: ", "Say this: "], ["a", "longer"]
    batch = adapter.build_sft_batch(tok, prompts, responses, 16)
    logits = model(batch["input_ids"]).logits[:, :-1]
    labels = batch["labels"][:, 1:]
    expected = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(adapter.compute_sft_loss(model, tok, prompts, responses, 16), expected)
