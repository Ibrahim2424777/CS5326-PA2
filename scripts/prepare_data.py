"""Download and prepare every dataset used by the assignment (Task 2-4 inputs).

Writes to ``--out_dir`` (default ``outputs/data``):
    sft_train.jsonl (30k)  sft_val.jsonl (1k)     UltraChat-200k, first user/assistant turn
    hh_train.jsonl  (15k)  hh_val.jsonl  (1k)     Anthropic/hh-rlhf, single-turn pairs
    uf_train.jsonl  (15k)  uf_val.jsonl  (1k)     HuggingFaceH4/ultrafeedback_binarized
    data_stats.json                               filter counts for the report
and makes sure ``eval_sets/alpaca_eval_300_seed42.json`` exists.

Example:
    python scripts/prepare_data.py --out_dir outputs/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smalign import data as D  # noqa: E402
from smalign.utils import REPO_ROOT, read_json, set_seed, write_json, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="outputs/data")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    parser.add_argument("--sft_train", type=int, default=30000)
    parser.add_argument("--sft_val", type=int, default=1000)
    parser.add_argument("--dpo_train", type=int, default=15000)
    parser.add_argument("--dpo_val", type=int, default=1000)
    parser.add_argument("--only", nargs="*", choices=["sft", "hh", "uf", "alpaca"], default=None)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tracks = set(args.only or ["sft", "hh", "uf", "alpaca"])
    stats_path = out / "data_stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    stats["config"] = vars(args)

    alpaca_path = REPO_ROOT / "eval_sets" / "alpaca_eval_300_seed42.json"
    if "alpaca" in tracks and not alpaca_path.exists():
        write_json(alpaca_path, D.alpaca_eval_subset(300, 42))
    eval_prompts = set()
    for name in ("alpaca_eval_300_seed42.json", "hh_rlhf_eval_300_seed42.json"):
        path = REPO_ROOT / "eval_sets" / name
        if path.exists():
            eval_prompts |= {row["instruction"].strip() for row in read_json(path)}

    common = dict(
        seed=args.seed, tokenizer=tokenizer, max_prompt_tokens=args.max_prompt_tokens, exclude_prompts=eval_prompts
    )
    jobs = {
        "sft": ("HuggingFaceH4/ultrachat_200k", "train_sft", "test_sft", D.parse_ultrachat, args.sft_train, args.sft_val),
        "hh": ("Anthropic/hh-rlhf", "train", "test", D.parse_hh_rlhf, args.dpo_train, args.dpo_val),
        "uf": ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs", D.parse_ultrafeedback, args.dpo_train, args.dpo_val),
    }
    for track, (repo, train_split, val_split, parse, n_train, n_val) in jobs.items():
        if track not in tracks:
            continue
        for split_name, hf_split, n in (("train", train_split, n_train), ("val", val_split, n_val)):
            path = out / f"{track}_{split_name}.jsonl"
            print(f"[{track}] loading {repo}:{hf_split} -> {path}", flush=True)
            dataset = load_dataset(repo, split=hf_split)
            rows, split_stats = D.sample_rows(dataset, parse, n, id_prefix=f"{track}-{split_name}-", **common)
            if len(rows) < n:
                print(f"[{track}] WARNING: only {len(rows)} rows passed the filters (wanted {n})", flush=True)
            write_jsonl(path, rows)
            split_stats.update(source=f"{repo}:{hf_split}", source_rows=len(dataset))
            stats[f"{track}_{split_name}"] = split_stats
            print(f"[{track}] {split_name}: {split_stats}", flush=True)
            write_json(stats_path, stats)
    print("done ->", out.resolve())


if __name__ == "__main__":
    main()
