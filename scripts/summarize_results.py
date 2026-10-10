"""Task 8 - collect every number the report needs into one markdown file.

Sections written to ``<out_dir>/results.md``:
  1. experimental configuration (from each run's saved config)
  2. training summaries (best validation numbers, steps, minutes)
  3. DPO-vs-SFT judge results (+ per-category breakdown, position counts)
  4. generation statistics (length, share hitting the token limit)
  5. representative win / loss examples for the error analysis
Missing inputs are skipped, so it can be run at any stage.

Example:
    python scripts/summarize_results.py --runs_dir outputs --eval_dir outputs/eval \
        --gen_dir outputs/generations --out_dir outputs/report_assets
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smalign.utils import read_json, read_jsonl  # noqa: E402


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def config_table(runs_dir: Path) -> list[str]:
    lines = ["## Experimental configuration (values actually used)", ""]
    header = ("| Run | Params trained | Optimizer | Peak LR | Schedule / warmup | WD | Clip | Batch x accum = eff. | "
              "Precision | Max len (prompt) | Epochs / steps | beta | Reference | Seed |")
    lines += [header, "|" + "---|" * 14]
    for name in ("sft", "dpo_hh", "dpo_uf"):
        path = runs_dir / name / "run_config.json"
        if not path.exists():
            continue
        c = read_json(path)
        summary = read_json(runs_dir / name / "summary.json") if (runs_dir / name / "summary.json").exists() else {}
        is_sft = name == "sft"
        optimizer = "AdamW" if is_sft else {"rmsprop": "RMSprop", "adamw": "AdamW"}.get(c.get("optimizer", "").lower(), c.get("optimizer"))
        schedule = f"{c.get('scheduler')} / {c.get('warmup_ratio', c.get('warmup_steps'))}"
        batch = f"{c['per_device_batch_size']} x {c['grad_accum_steps']} = {c['per_device_batch_size'] * c['grad_accum_steps']}"
        precision = "fp32 weights + fp16 autocast" if c.get("fp16") else "fp32"
        if not is_sft:
            precision += f"; ref {c.get('reference_dtype')}"
        lines.append(
            f"| {name} | full (494M) | {optimizer} | {c['learning_rate']} | {schedule} | {c.get('weight_decay')} | "
            f"{c.get('max_grad_norm')} | {batch} | {precision} | {c['max_length']} ({c.get('max_prompt_length')}) | "
            f"{c['epochs']} / {summary.get('optimizer_steps', '-')} | {c.get('beta', '-')} | "
            f"{c.get('reference_path', '-') if not is_sft else '-'} | {c['seed']} |"
        )
    lora = runs_dir / "lora_persona" / "summary.json"
    if lora.exists():
        s = read_json(lora)
        c = s["config"]
        steps = s.get("final_eval", {}).get("step", "-")
        lines.append(
            f"| lora_persona | LoRA r={c['lora_r']} a={c['lora_alpha']} drop={c['lora_dropout']} "
            f"on {','.join(c['lora_target_modules'])} ({s['trainable_params']:,} params) | AdamW (TRL) | {c['learning_rate']} | "
            f"{c['lr_scheduler_type']} / {c['warmup_ratio']} | 0 | {c['max_grad_norm']} | "
            f"{c['per_device_train_batch_size']} x {c['gradient_accumulation_steps']} = "
            f"{c['per_device_train_batch_size'] * c['gradient_accumulation_steps']} | fp32 + fp16 AMP | {c['max_length']} | "
            f"{c['num_train_epochs']} / {steps} | {c['beta']} | base SmolLM2 (adapter off) | {c['seed']} |"
        )
    return lines + [""]


def training_summaries(runs_dir: Path) -> list[str]:
    lines = ["## Training summaries", "", "| Run | Steps | Best step | Best val metric | Minutes | Peak memory |", "|---|---|---|---|---|---|"]
    for name in ("sft", "dpo_hh", "dpo_uf"):
        path = runs_dir / name / "summary.json"
        if not path.exists():
            continue
        s = read_json(path)
        val_rows = [r for r in read_jsonl(runs_dir / name / "metrics.jsonl") if r.get("split") == "val"]
        if name == "sft":
            best_step = next((r["step"] for r in reversed(val_rows) if r.get("is_best")), "-")
            metric = f"val loss {fmt(s.get('best_val_loss'))}"
        else:
            best_step = s.get("best_step")
            metric = f"reward acc {fmt(s.get('best_val_reward_accuracy'))}, margin {fmt(s.get('best_val_reward_margin'), 4)}"
        lines.append(f"| {name} | {s.get('optimizer_steps')} | {best_step} | {metric} | {s.get('minutes')} | {s.get('peak_memory')} |")
        if val_rows and name != "sft":
            lines.append(f"|  ↳ step-0 sanity | | | loss {fmt(val_rows[0].get('loss'), 4)}, acc {fmt(val_rows[0].get('reward_accuracy'))} | | |")
    lora = runs_dir / "lora_persona" / "summary.json"
    if lora.exists():
        s = read_json(lora)
        fe, ie = s.get("final_eval", {}), s.get("initial_eval", {})
        lines.append(
            f"| lora_persona | - | final | eval reward acc {fmt(ie.get('eval_rewards/accuracies'))} -> "
            f"{fmt(fe.get('eval_rewards/accuracies'))}, margin {fmt(fe.get('eval_rewards/margins'))} | {s.get('minutes')} | - |"
        )
    return lines + [""]


def judge_tables(eval_dir: Path) -> list[str]:
    lines = ["## DPO vs SFT (and LoRA vs base) judge results", ""]
    summaries = sorted(eval_dir.glob("*_judged_summary.json"))
    if not summaries:
        return lines + ["(no judge summaries found yet)", ""]
    lines += [
        "| Comparison | Criterion | Judge | n | Wins | Ties | Losses | Unknown | Failures | Win rate (ties=loss) | "
        "Win rate (ties=½) | Win rate (excl. ties) | Retried pairs | Verdict A/B/tie |",
        "|" + "---|" * 14,
    ]
    breakdowns = []
    for path in summaries:
        s = read_json(path)
        if not s.get("n_pairs"):
            continue
        name = f"{s['candidate']} vs {s['baseline']}"
        for criterion, c in s["criteria"].items():
            pos = c.get("position_counts", {})
            lines.append(
                f"| {name} | {criterion} | {', '.join(s['judge_models'])} | {c['n']} | {c['wins']} | {c['ties']} | {c['losses']} | "
                f"{c['unknown']} | {c['failures']} | {fmt(c['win_rate'])} | {fmt(c['win_rate_ties_half'])} | "
                f"{fmt(c['win_rate_excl_ties'])} | {s.get('pairs_needing_retry')} | {pos.get('A')}/{pos.get('B')}/{pos.get('tie')} |"
            )
            breakdowns += [f"### {name} — {criterion} by category", "", "| Category | n | W | T | L | Win rate (ties=½) |", "|---|---|---|---|---|---|"]
            for category, b in c["by_category"].items():
                breakdowns.append(f"| {category} | {b['n']} | {b['wins']} | {b['ties']} | {b['losses']} | {fmt(b['win_rate_ties_half'])} |")
            breakdowns.append("")
    return lines + [""] + breakdowns


def generation_stats(gen_dir: Path) -> list[str]:
    lines = ["## Generation statistics", "", "| File | Generator | n | Mean words | Median words | Hit token limit |", "|---|---|---|---|---|---|"]
    for path in sorted(gen_dir.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        rows = read_json(path)
        if not rows:
            continue
        meta_path = path.with_suffix(".meta.json")
        limit = read_json(meta_path)["decoding"]["max_new_tokens"] if meta_path.exists() else None
        words = [len(row["output"].split()) for row in rows]
        capped = sum(1 for row in rows if limit and row.get("num_new_tokens", 0) >= limit - 1)
        lines.append(f"| {path.name} | {rows[0]['generator']} | {len(rows)} | {statistics.mean(words):.0f} | "
                     f"{statistics.median(words):.0f} | {capped / len(rows):.1%} |")
    return lines + [""]


def examples(eval_dir: Path, per_kind: int = 2, seed: int = 0) -> list[str]:
    lines = ["## Representative examples (for the error analysis)", ""]
    rng = random.Random(seed)
    for judged_path in sorted(eval_dir.glob("*_judged.jsonl")):
        pairs_path = judged_path.with_name(judged_path.name.replace("_judged.jsonl", "_pairs.jsonl"))
        if not pairs_path.exists():
            continue
        pairs = {p["pair_id"]: p for p in read_jsonl(pairs_path)}
        judged = [r for r in read_jsonl(judged_path) if r["status"] == "ok"]
        criterion = "quality_winner" if judged and "quality_winner" in judged[0] else "winner"
        for outcome in ("win", "loss", "tie"):
            chosen = [r for r in judged if r.get(f"outcome_{criterion}") == outcome]
            rng.shuffle(chosen)
            for row in chosen[:per_kind]:
                pair = pairs[row["pair_id"]]
                meta = pair["hidden_meta"]
                cand_slot = "A" if meta["A"] == meta["candidate"] else "B"
                cand = pair[f"response_{cand_slot.lower()}"]
                base = pair["response_b" if cand_slot == "A" else "response_a"]
                lines += [
                    f"### {judged_path.stem}: candidate {outcome.upper()} ({row.get('category')}, {row['pair_id']})",
                    f"**Prompt:** {pair['instruction'][:600]}",
                    "",
                    f"**{meta['candidate']}:** {cand[:700]}",
                    "",
                    f"**{meta['baseline']}:** {base[:700]}",
                    "",
                    f"**Judge reason:** {row.get('reason')}",
                    "",
                ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", default="outputs")
    parser.add_argument("--eval_dir", default="outputs/eval")
    parser.add_argument("--gen_dir", default="outputs/generations")
    parser.add_argument("--out_dir", default="outputs/report_assets")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Results summary (auto-generated by scripts/summarize_results.py)", ""]
    lines += config_table(Path(args.runs_dir))
    lines += training_summaries(Path(args.runs_dir))
    lines += judge_tables(Path(args.eval_dir))
    lines += generation_stats(Path(args.gen_dir))
    lines += examples(Path(args.eval_dir))
    (out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] wrote {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
