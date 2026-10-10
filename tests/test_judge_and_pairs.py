"""Offline checks for the evaluation plumbing: blinding, verdict parsing and A/B -> model mapping."""

from __future__ import annotations

from smalign.judge import build_judge_prompt, outcome_for_candidate, parse_judge_response, summarize


def _pair(pair_id, candidate_slot="A", rubric="general"):
    other = "B" if candidate_slot == "A" else "A"
    return {
        "pair_id": pair_id,
        "id": pair_id,
        "instruction": "Explain DPO.",
        "category": "c1",
        "expected_behavior": "Answer normally.",
        "rubric": rubric,
        "response_a": "answer one",
        "response_b": "answer two",
        "hidden_meta": {candidate_slot: "dpo_uf", other: "sft", "candidate": "dpo_uf", "baseline": "sft"},
    }


def test_judge_prompt_never_reveals_model_identity():
    pairs = [_pair("p1"), _pair("p2", "B")]
    for rubric in ("general", "safety", "persona"):
        text = build_judge_prompt(pairs, rubric).lower()
        assert "dpo_uf" not in text and "sft" not in text and "hidden_meta" not in text
        assert "response a" in text and "response b" in text
    assert "expected behavior" in build_judge_prompt(pairs, "safety").lower()


def test_parse_handles_code_fences_case_and_missing_items():
    pairs = [_pair("p1"), _pair("p2"), _pair("p3")]
    text = 'Sure!\n```json\n[{"pair_id": "p1", "winner": "a", "reason": "x"}, {"pair_id": "p3", "winner": "Tie"}]\n```'
    parsed = parse_judge_response(text, pairs, "general")
    assert parsed["p1"]["winner"] == "A" and parsed["p3"]["winner"] == "tie" and "p2" not in parsed


def test_parse_rejects_invalid_verdicts_and_falls_back_to_objects():
    pairs = [_pair("p1"), _pair("p2")]
    text = 'noise {"pair_id": "p1", "winner": "C"} more {"pair_id": "p2", "winner": "Response B"} trailing ['
    parsed = parse_judge_response(text, pairs, "general")
    assert "p1" not in parsed and parsed["p2"]["winner"] == "B"


def test_persona_needs_both_criteria():
    pairs = [_pair("p1", rubric="persona"), _pair("p2", rubric="persona")]
    text = '[{"pair_id": "p1", "style_winner": "A", "quality_winner": "B"}, {"pair_id": "p2", "style_winner": "A"}]'
    parsed = parse_judge_response(text, pairs, "persona")
    assert parsed["p1"] == {"style_winner": "A", "quality_winner": "B", "reason": ""} and "p2" not in parsed


def test_outcome_mapping_uses_hidden_slot():
    assert outcome_for_candidate(_pair("p", "A"), "A") == "win"
    assert outcome_for_candidate(_pair("p", "B"), "A") == "loss"
    assert outcome_for_candidate(_pair("p", "B"), "tie") == "tie"
    assert outcome_for_candidate(_pair("p", "B"), None) == "unknown"


def test_summary_rates():
    rows = []
    for outcome, status in [("win", "ok"), ("win", "ok"), ("loss", "ok"), ("tie", "ok"), ("unknown", "unknown"), ("unknown", "failed")]:
        row = _pair(f"p{len(rows)}")
        row.update(status=status, outcome_winner=outcome, winner="A", judge_model="j", attempts=1)
        rows.append(row)
    stats = summarize(rows)["criteria"]["winner"]
    assert (stats["wins"], stats["losses"], stats["ties"], stats["unknown"], stats["failures"]) == (2, 1, 1, 1, 1)
    assert stats["win_rate"] == 0.5 and stats["win_rate_excl_ties"] == 2 / 3 and stats["win_rate_ties_half"] == 0.625
