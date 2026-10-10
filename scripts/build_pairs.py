"""Task 6 - build blinded pairwise comparison files (candidate vs baseline).

For every prompt present in both generation files we randomly (seeded) decide
whether the candidate is shown as Response A or Response B. The mapping is
stored in ``hidden_meta`` and is never sent to the judge.

Example:
    python scripts/build_pairs.py --candidate outputs/generations/dpo_uf_alpaca.json \
        --baseline outputs/generations/sft_alpaca.json --rubric general \
        --output_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smalign.utils import read_json, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, help="generation JSON of the model being evaluated (e.g. DPO)")
    parser.add_argument("--baseline", required=True, help="generation JSON of the reference model (e.g. SFT)")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--rubric", choices=["general", "safety", "persona"], required=True)
    parser.add_argument("--eval_set", default=None, help="optional eval set to copy expected_behavior from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidate = {row["id"]: row for row in read_json(args.candidate)}
    baseline = {row["id"]: row for row in read_json(args.baseline)}
    extra = {row["id"]: row for row in read_json(args.eval_set)} if args.eval_set else {}
    shared = [row_id for row_id in candidate if row_id in baseline]
    missing = len(candidate) + len(baseline) - 2 * len(shared)
    if missing:
        print(f"[pairs] warning: {missing} ids are not present in both files and were skipped", flush=True)

    rng = random.Random(args.seed)
    pairs = []
    for row_id in shared:
        cand, base = candidate[row_id], baseline[row_id]
        candidate_is_a = rng.random() < 0.5
        a, b = (cand, base) if candidate_is_a else (base, cand)
        pairs.append(
            {
                "pair_id": f"{row_id}",
                "id": row_id,
                "instruction": cand["instruction"],
                "category": cand.get("category"),
                "expected_behavior": extra.get(row_id, {}).get("expected_behavior"),
                "rubric": args.rubric,
                "response_a": a["output"],
                "response_b": b["output"],
                # Hidden from the judge: which model produced which slot.
                "hidden_meta": {
                    "A": a["generator"],
                    "B": b["generator"],
                    "candidate": cand["generator"],
                    "baseline": base["generator"],
                },
            }
        )
    write_jsonl(args.output_path, pairs)
    n_a = sum(pair["hidden_meta"]["A"] == pair["hidden_meta"]["candidate"] for pair in pairs)
    print(f"[pairs] wrote {len(pairs)} pairs to {args.output_path} (candidate shown as A in {n_a})", flush=True)


if __name__ == "__main__":
    main()
