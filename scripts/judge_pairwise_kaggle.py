"""Task 7 - resumable pairwise judging through Kaggle's model proxy (kaggle_benchmarks).

Reliability features (manual 13.2):
  * reads saved pairwise JSONL only (never regenerates outputs)
  * small batches (default 5 pairs per request)
  * checkpoints the judged file after every batch (atomic rewrite)
  * sleeps >= --sleep_seconds before every request attempt after the first
  * retries each batch up to --max_retries attempts (only the still-unresolved pairs)
  * unresolved pairs are logged to a separate failures JSONL
  * --reuse_judgments skips pairs already judged successfully
  * writes a summary JSON with win / loss / tie / unknown / failure rates

The judge prompt contains only "Response A" / "Response B"; the hidden
model mapping is applied after parsing.

Run this inside a Kaggle notebook created from https://www.kaggle.com/benchmarks/tasks/new
(that is what provides the model-proxy credentials). Example:

    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
        --output_path outputs/eval/dpo_uf_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

``--backend mock`` runs the whole loop offline with a fake judge (for testing).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smalign.judge import CRITERIA, build_judge_prompt, outcome_for_candidate, parse_judge_response, summarize  # noqa: E402
from smalign.utils import append_jsonl, read_jsonl, write_json, write_jsonl  # noqa: E402

DEFAULT_FALLBACKS = ("google/gemini-3.5-flash", "google/gemini-3-flash", "google/gemini-3-flash-preview", "google/gemini-2.5-flash")


class KaggleJudge:
    """Thin wrapper around kaggle_benchmarks: pick the model, send one prompt, return text."""

    def __init__(self, requested: str, fallbacks: list[str]):
        import kaggle_benchmarks as kbench

        self.kbench = kbench
        available = dict(getattr(kbench, "llms", {}) or {})
        print(f"[judge] models available through the Kaggle proxy: {sorted(available)}", flush=True)
        for name in [requested, *fallbacks]:
            if name in available:
                self.llm, self.name = available[name], name
                break
        else:
            judge_llm = getattr(kbench, "judge_llm", None)
            name = getattr(judge_llm, "name", None)
            if judge_llm is None or not name:
                raise RuntimeError(
                    "No judge model is available. Create the notebook from kaggle.com/benchmarks/tasks/new so that "
                    "the model proxy (MODEL_PROXY_URL / MODEL_PROXY_API_KEY) is configured."
                )
            self.llm, self.name = judge_llm, name
        if self.name != requested:
            print(f"[judge] WARNING: {requested} not available; using {self.name} instead (record this in the report)", flush=True)
        print(f"[judge] using judge model {self.name}", flush=True)

    def __call__(self, prompt: str, chat_name: str) -> str:
        chats = getattr(self.kbench, "chats", None)
        if chats is not None and hasattr(chats, "new"):
            with chats.new(chat_name):  # fresh chat per request, so no history leaks between batches
                response = self.llm.prompt(prompt)
        else:
            response = self.llm.prompt(prompt)
        return response if isinstance(response, str) else str(getattr(response, "content", response))


class MockJudge:
    """Offline stand-in that answers in the expected JSON format (with occasional junk)."""

    name = "mock-judge"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, prompt: str, chat_name: str) -> str:
        import json
        import re

        if self.rng.random() < 0.15:
            raise RuntimeError("simulated proxy error")
        ids = re.findall(r"### Pair (\S+)", prompt)
        persona = "style_winner" in prompt
        items = []
        for pair_id in ids:
            if self.rng.random() < 0.1:
                continue  # simulate a missing verdict
            pick = lambda: self.rng.choice(["A", "B", "tie"])  # noqa: E731
            item = {"pair_id": pair_id, "reason": "mock"}
            if persona:
                item.update(style_winner=pick(), quality_winner=pick())
            else:
                item["winner"] = pick()
            items.append(item)
        return "```json\n" + json.dumps(items) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--failures_path", default=None, help="default: <output>_failures.jsonl")
    parser.add_argument("--summary_path", default=None, help="default: <output>_summary.json")
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--fallback_models", nargs="*", default=list(DEFAULT_FALLBACKS))
    parser.add_argument("--backend", choices=["kaggle", "mock"], default="kaggle")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=3, help="attempts per batch")
    parser.add_argument("--reuse_judgments", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    stem = output_path.with_suffix("")
    failures_path = Path(args.failures_path or f"{stem}_failures.jsonl")
    summary_path = Path(args.summary_path or f"{stem}_summary.json")
    pairs = read_jsonl(args.pairs_path)[: args.limit]
    if not pairs:
        raise SystemExit(f"no pairs found in {args.pairs_path}")
    rubric = pairs[0]["rubric"]
    criteria = CRITERIA[rubric]

    judged = {row["pair_id"]: row for row in read_jsonl(output_path)}
    if not args.reuse_judgments and not args.summarize_only:
        judged = {}
    done_ok = {pid for pid, row in judged.items() if row.get("status") == "ok"}
    pending = [pair for pair in pairs if pair["pair_id"] not in done_ok]
    print(f"[judge] {len(pairs)} pairs, {len(done_ok)} already judged, {len(pending)} pending, rubric={rubric}", flush=True)

    def checkpoint() -> None:
        ordered = [judged[pair["pair_id"]] for pair in pairs if pair["pair_id"] in judged]
        write_jsonl(output_path, ordered)
        write_json(summary_path, summarize(ordered, len(pairs)))

    if args.summarize_only or not pending:
        checkpoint()
        print(f"[judge] summary -> {summary_path}", flush=True)
        return

    judge = MockJudge() if args.backend == "mock" else KaggleJudge(args.judge_model, args.fallback_models)
    sleep_seconds = 0.0 if args.backend == "mock" else max(5.0, args.sleep_seconds)
    first_request = True
    for batch_index in range(0, len(pending), args.batch_size):
        batch = pending[batch_index : batch_index + args.batch_size]
        unresolved = list(batch)
        attempts: dict[str, int] = {pair["pair_id"]: 0 for pair in batch}
        last_error: dict[str, str] = {}
        for attempt in range(1, args.max_retries + 1):
            if not unresolved:
                break
            if not first_request:
                time.sleep(sleep_seconds)
            first_request = False
            for pair in unresolved:
                attempts[pair["pair_id"]] += 1
            try:
                text = judge(build_judge_prompt(unresolved, rubric), f"judge-{batch_index}-{attempt}")
            except Exception as error:  # proxy errors, timeouts, quota denials...
                message = f"{type(error).__name__}: {error}"
                print(f"[judge] batch {batch_index // args.batch_size} attempt {attempt} failed: {message}", flush=True)
                for pair in unresolved:
                    last_error[pair["pair_id"]] = "request_error: " + message[:500]
                continue
            verdicts = parse_judge_response(text, unresolved, rubric)
            still = []
            for pair in unresolved:
                verdict = verdicts.get(pair["pair_id"])
                if verdict is None:
                    last_error[pair["pair_id"]] = "unparseable_or_missing_verdict: " + text[:300].replace("\n", " ")
                    still.append(pair)
                    continue
                row = {key: pair[key] for key in ("pair_id", "id", "category", "rubric", "hidden_meta")}
                row.update(status="ok", judge_model=judge.name, attempts=attempts[pair["pair_id"]], reason=verdict["reason"])
                for criterion in criteria:
                    row[criterion] = verdict[criterion]
                    row[f"outcome_{criterion}"] = outcome_for_candidate(pair, verdict[criterion])
                    row[f"winner_model_{criterion}"] = (
                        pair["hidden_meta"][verdict[criterion]] if verdict[criterion] in ("A", "B") else "tie"
                    )
                judged[pair["pair_id"]] = row
            unresolved = still

        for pair in unresolved:  # out of attempts
            error = last_error.get(pair["pair_id"], "unknown")
            status = "failed" if error.startswith("request_error") else "unknown"
            row = {key: pair[key] for key in ("pair_id", "id", "category", "rubric", "hidden_meta")}
            row.update(status=status, judge_model=judge.name, attempts=attempts[pair["pair_id"]], error=error)
            for criterion in criteria:
                row[criterion] = None
                row[f"outcome_{criterion}"] = "unknown"
            judged[pair["pair_id"]] = row
            append_jsonl(failures_path, {**row, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        checkpoint()
        n_ok = sum(1 for row in judged.values() if row["status"] == "ok")
        print(f"[judge] batch {batch_index // args.batch_size + 1}/{-(-len(pending) // args.batch_size)} "
              f"saved; ok={n_ok}/{len(pairs)}", flush=True)

    checkpoint()
    summary = summarize([judged[p["pair_id"]] for p in pairs if p["pair_id"] in judged], len(pairs))
    for criterion, stats in summary["criteria"].items():
        print(f"[judge] {criterion}: wins={stats['wins']} losses={stats['losses']} ties={stats['ties']} "
              f"unknown={stats['unknown']} failures={stats['failures']} win_rate={stats['win_rate']}", flush=True)
    print(f"[judge] judged -> {output_path}\n[judge] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
