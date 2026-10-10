"""End-to-end preflight of every script on a tiny random Qwen2 model (CPU or GPU, ~2-4 min).

Runs, as subprocesses exactly like the notebooks do:
  train_sft -> train_dpo -> generate (x2) -> build_pairs -> judge (mock backend)
  -> plot_training -> summarize_results
on synthetic data, so it needs only the Qwen tokenizer download. It verifies
the plumbing (paths, configs, checkpoint save/load, resumable files), not quality.

Example:
    python scripts/smoke_test_pipeline.py --work_dir /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smalign.utils import read_json, read_jsonl, write_jsonl  # noqa: E402

WORDS = "the a cat dog sat ran on under table quickly slowly blue red happy sad answer question".split()


def sentence(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="outputs/smoke")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.eos_token_id,
    )
    tiny = work / "tiny_qwen2"
    Qwen2ForCausalLM(config).save_pretrained(tiny)
    tokenizer.save_pretrained(tiny)

    rng = random.Random(0)
    write_jsonl(data / "sft_train.jsonl", [{"id": f"s{i}", "prompt": sentence(rng, 6), "response": sentence(rng, 12)} for i in range(48)])
    write_jsonl(data / "sft_val.jsonl", [{"id": f"v{i}", "prompt": sentence(rng, 6), "response": sentence(rng, 12)} for i in range(8)])
    for split, n in (("train", 64), ("val", 8)):
        write_jsonl(data / f"hh_{split}.jsonl", [
            {"id": f"p{i}", "prompt": sentence(rng, 6), "chosen": "happy answer " + sentence(rng, 8), "rejected": "sad " + sentence(rng, 20)}
            for i in range(n)
        ])

    common = ["max_length=64", "max_prompt_length=32", "device=cpu", "fp16=true"]
    run(["scripts/train_sft.py", "--config", "configs/sft.yaml", "--set", f"model_name_or_path={tiny}",
         f"train_path={data / 'sft_train.jsonl'}", f"val_path={data / 'sft_val.jsonl'}", f"output_dir={work / 'sft'}",
         "grad_accum_steps=4", "eval_every_steps=3", "log_every_steps=1", "learning_rate=1e-3", *common])
    sft_best = work / "sft" / "best"
    assert (sft_best / "config.json").exists() and (work / "sft" / "final" / "config.json").exists()

    run(["scripts/train_dpo.py", "--config", "configs/dpo_hh.yaml", "--set", f"policy_init={sft_best}", f"reference_path={sft_best}",
         f"train_path={data / 'hh_train.jsonl'}", f"val_path={data / 'hh_val.jsonl'}", f"output_dir={work / 'dpo_hh'}",
         "grad_accum_steps=8", "eval_every_steps=2", "log_every_steps=1", "learning_rate=1e-3", "policy_device=cpu",
         "reference_device=cpu", "max_length=64", "max_prompt_length=32", "fp16=true"])
    val_rows = [r for r in read_jsonl(work / "dpo_hh" / "metrics.jsonl") if r["split"] == "val"]
    assert abs(val_rows[0]["loss"] - 0.693147) < 1e-4 and val_rows[0]["reward_accuracy"] == 0.0, val_rows[0]
    print(f"[smoke] DPO step-0 val: loss={val_rows[0]['loss']:.5f} acc={val_rows[0]['reward_accuracy']} (expected ln2, 0)")
    print(f"[smoke] DPO last val: {val_rows[-1]}")

    gens = work / "generations"
    for name, path in (("sft", sft_best), ("dpo_hh", work / "dpo_hh" / "best")):
        run(["scripts/generate.py", "--model_path", str(path), "--generator", name,
             "--eval_set", "eval_sets/hh_rlhf_eval_300_seed42.json", "--output_path", str(gens / f"{name}_hh.json"),
             "--limit", "6", "--batch_size", "4", "--max_new_tokens", "12", "--device", "cpu"])
    # rerun must be a no-op thanks to resumability
    run(["scripts/generate.py", "--model_path", str(sft_best), "--generator", "sft",
         "--eval_set", "eval_sets/hh_rlhf_eval_300_seed42.json", "--output_path", str(gens / "sft_hh.json"),
         "--limit", "6", "--device", "cpu"])

    ev = work / "eval"
    run(["scripts/build_pairs.py", "--candidate", str(gens / "dpo_hh_hh.json"), "--baseline", str(gens / "sft_hh.json"),
         "--rubric", "safety", "--eval_set", "eval_sets/hh_rlhf_eval_300_seed42.json", "--output_path", str(ev / "dpo_hh_vs_sft_pairs.jsonl")])
    judge = ["scripts/judge_pairwise_kaggle.py", "--pairs_path", str(ev / "dpo_hh_vs_sft_pairs.jsonl"),
             "--output_path", str(ev / "dpo_hh_vs_sft_judged.jsonl"), "--backend", "mock", "--batch_size", "2", "--reuse_judgments"]
    run(judge)
    run(judge)  # resume pass: retries only failed/unknown pairs
    pairs = read_jsonl(ev / "dpo_hh_vs_sft_pairs.jsonl")
    prompt_text = __import__("smalign.judge", fromlist=["x"]).build_judge_prompt(pairs[:2], "safety")
    assert "sft" not in prompt_text.lower() and "dpo" not in prompt_text.lower(), "model identity leaked into judge prompt"
    print(json.dumps(read_json(ev / "dpo_hh_vs_sft_judged_summary.json")["criteria"]["winner"], indent=1)[:600])

    run(["scripts/plot_training.py", "--sft", str(work / "sft"), "--dpo", str(work / "dpo_hh"), "--out_dir", str(work / "report_assets")])
    run(["scripts/summarize_results.py", "--runs_dir", str(work), "--eval_dir", str(ev), "--gen_dir", str(gens),
         "--out_dir", str(work / "report_assets")])
    print("\n[smoke] ALL STAGES PASSED")


if __name__ == "__main__":
    main()
