"""Task 5 - persona style DPO with TRL ``DPOTrainer`` + PEFT LoRA (TRL is allowed only here).

Rows from ``data/persona_lora/ramsay_style_preferences.json`` are converted to
TRL's conversational preference format::

    {"prompt":   [{"role": "user", "content": ...}],
     "chosen":   [{"role": "assistant", "content": ...}],
     "rejected": [{"role": "assistant", "content": ...}]}

With ``peft_config`` and ``ref_model=None`` TRL uses the base model with the
adapter disabled as the frozen reference, so no second model copy is loaded.

Outputs: ``<output_dir>/adapter`` (LoRA weights), ``metrics.jsonl`` (TRL log
history), ``summary.json``.

Example:
    pip install -q "trl==0.29.1" "peft>=0.17,<0.19"
    python scripts/train_lora_dpo.py --config configs/lora_persona.yaml
"""

from __future__ import annotations

import argparse
import inspect
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from smalign.utils import append_jsonl, load_config, parse_overrides, read_json, set_seed, write_json  # noqa: E402


def to_conversational(row: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": row["prompt"].strip()}],
        "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
        "rejected": [{"role": "assistant", "content": row["rejected"].strip()}],
    }


def filtered_kwargs(cls, kwargs: dict) -> dict:
    """Keep only the kwargs this TRL version's class accepts (and report the dropped ones)."""

    accepted = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(key for key in kwargs if key not in accepted)
    if dropped:
        print(f"[lora] note: {cls.__name__} in this TRL version does not accept {dropped}; skipped", flush=True)
    return {key: value for key, value in kwargs.items() if key in accepted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/lora_persona.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    config = load_config(args.config, parse_overrides(args.set))
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])

    import peft
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__}
    print(f"[lora] versions {versions}", flush=True)

    rows = read_json(config["data_path"])
    order = list(range(len(rows)))
    random.Random(config["seed"]).shuffle(order)
    val_ids = set(order[: config["val_size"]])
    train = [to_conversational(rows[i]) for i in order if i not in val_ids]
    val = [to_conversational(rows[i]) for i in order if i in val_ids]
    write_json(out_dir / "split.json", {"val_source_indices": sorted(val_ids), "train_size": len(train)})
    print(f"[lora] train pairs={len(train)} val pairs={len(val)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_fp16 = bool(config["fp16"]) and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=list(config["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    dpo_kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        beta=config["beta"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        max_length=config["max_length"],
        max_grad_norm=config["max_grad_norm"],
        fp16=use_fp16,
        bf16=False,  # T4 has no fast bf16; TRL would otherwise default to bf16
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,  # progress bars become huge single lines in Kaggle logs
        seed=config["seed"],
        loss_type="sigmoid",
        remove_unused_columns=False,
    )
    if "eval_strategy" not in inspect.signature(DPOConfig.__init__).parameters:  # transformers < 4.41 naming
        dpo_kwargs["evaluation_strategy"] = dpo_kwargs.pop("eval_strategy")
    training_args = DPOConfig(**filtered_kwargs(DPOConfig, dpo_kwargs))

    logged: list[dict] = []

    class JsonlLogger(TrainerCallback):
        # TRL attaches the reward metrics inside log(), so the callback (not evaluate()'s return value)
        # is where the complete train/eval records are available.
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                record = {"step": state.global_step, "epoch": state.epoch or 0.0, **logs}
                logged.append(record)
                append_jsonl(out_dir / "metrics.jsonl", record)

    (out_dir / "metrics.jsonl").unlink(missing_ok=True)
    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(val),
        peft_config=lora_config,
        callbacks=[JsonlLogger()],
    )
    if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.model.print_trainable_parameters()

    start = time.time()
    trainer.evaluate()  # step-0 sanity check: loss = ln 2, rewards = 0
    trainer.train()
    trainer.evaluate()
    eval_logs = [record for record in logged if "eval_loss" in record]
    initial_eval, final_eval = eval_logs[0], eval_logs[-1]

    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    summary = {
        "run_name": config["run_name"],
        "versions": versions,
        "trainable_params": trainable,
        "total_params": total,
        "train_pairs": len(train),
        "val_pairs": len(val),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "minutes": round((time.time() - start) / 60, 2),
        "config": config,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[lora] done: eval reward acc {initial_eval.get('eval_rewards/accuracies')} -> "
          f"{final_eval.get('eval_rewards/accuracies')}, margin -> {final_eval.get('eval_rewards/margins')}", flush=True)


if __name__ == "__main__":
    main()
