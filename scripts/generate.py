"""Tasks 2 & 6 - generate responses for a fixed evaluation set.

Output: a JSON list with one entry per prompt (original eval-set order)::

    {"id", "instruction", "output", "generator", "dataset", "category"}

plus ``<output>.meta.json`` recording the exact decoding settings. The file is
rewritten after every batch and completed ids are skipped on rerun, so a
disconnect never loses finished work.

Decoding (identical for every model in a comparison): greedy, repetition
penalty 1.1, at most 512 new tokens, stop on EOS / ``<|im_end|>``.

Examples:
    # sanity check on 3 prompts, printed to stdout
    python scripts/generate.py --model_path outputs/sft/best --generator sft \
        --eval_set eval_sets/alpaca_eval_300_seed42.json --output_path /tmp/x.json --limit 3 --print_samples
    # LoRA persona model (SmolLM2 chat template + adapter)
    python scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct --adapter_path outputs/lora_persona/adapter \
        --template chat_template --generator lora_dpo --eval_set eval_sets/persona_style_eval_40.json --output_path ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from smalign.prompting import STOP_STRINGS, TEMPLATE_NAME, format_prompt, stop_token_ids  # noqa: E402
from smalign.utils import read_json, resolve_device, set_seed, write_json  # noqa: E402

DTYPES = {"float32": torch.float32, "float16": torch.float16}


def build_prompt(tokenizer, instruction: str, template: str) -> str:
    if template == "chatml":
        return format_prompt(instruction)
    messages = [{"role": "user", "content": instruction.strip()}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean(text: str) -> str:
    for stop in STOP_STRINGS + ("<|im_start|>",):
        text = text.split(stop)[0]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="optional PEFT/LoRA adapter applied on top of model_path")
    parser.add_argument("--generator", required=True, help="model label stored in every entry, e.g. sft, dpo_uf")
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--template", choices=["chatml", "chat_template"], default="chatml",
                        help="chatml = our SFT/DPO template; chat_template = tokenizer's own (SmolLM2-Instruct)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity checks)")
    parser.add_argument("--print_samples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    device = resolve_device(args.device)
    eval_rows = read_json(args.eval_set)[: args.limit]
    out_path = Path(args.output_path)
    existing = {row["id"]: row for row in read_json(out_path)} if out_path.exists() else {}
    pending = [row for row in eval_rows if row["id"] not in existing]
    meta = {
        "generator": args.generator,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "eval_set": args.eval_set,
        "n_prompts": len(eval_rows),
        "template": TEMPLATE_NAME if args.template == "chatml" else "tokenizer.chat_template",
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "dtype": args.dtype,
    }
    write_json(out_path.with_suffix(".meta.json"), meta)
    if not pending:
        print(f"[gen] {out_path} already complete ({len(existing)} rows); skipping", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path or args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=DTYPES[args.dtype])
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.to(device).eval()
    model.config.use_cache = True
    eos_ids = stop_token_ids(tokenizer)
    print(f"[gen] {args.generator}: {len(pending)} prompts pending, stop ids {eos_ids}, template {meta['template']}", flush=True)

    # Longest-first batching keeps padding low; output order is restored when writing.
    pending.sort(key=lambda row: len(row["instruction"]), reverse=True)
    start = time.time()
    for begin in range(0, len(pending), args.batch_size):
        chunk = pending[begin : begin + args.batch_size]
        prompts = [build_prompt(tokenizer, row["instruction"], args.template) for row in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        for row, tokens in zip(chunk, new_tokens):
            text = clean(tokenizer.decode(tokens, skip_special_tokens=False))
            existing[row["id"]] = {
                "id": row["id"],
                "instruction": row["instruction"],
                "output": text,
                "generator": args.generator,
                "dataset": row.get("dataset", Path(args.eval_set).stem),
                "category": row.get("category"),
                "num_new_tokens": int((tokens != tokenizer.pad_token_id).sum()),
            }
            if args.print_samples:
                print(f"\n=== [{args.generator}] {row['id']}\nPROMPT: {row['instruction']}\nOUTPUT: {text}\n", flush=True)
        ordered = [existing[row["id"]] for row in eval_rows if row["id"] in existing]
        write_json(out_path, ordered)
        done = len(ordered)
        print(f"[gen] {args.generator}: {done}/{len(eval_rows)} done, {time.time() - start:.0f}s", flush=True)
    print(f"[gen] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
