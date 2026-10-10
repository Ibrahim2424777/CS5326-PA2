"""Task 3 - supervised fine-tuning with a hand-written PyTorch loop.

Implements: chat-template prompt formatting, response-only loss masking,
padding + attention masks, gradient accumulation, fp16 autocast with fp32
master weights and GradScaler, gradient clipping, warmup + cosine schedule,
periodic validation loss, and best/final checkpoints. Metrics are appended to
``<output_dir>/metrics.jsonl`` after every log/eval step.

Example:
    python scripts/train_sft.py --config configs/sft.yaml
    python scripts/train_sft.py --config configs/sft.yaml --set max_train_examples=256 eval_every_steps=4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from smalign.batching import build_sft_batch  # noqa: E402
from smalign.losses import response_logprobs_from_batch, sft_loss_from_batch  # noqa: E402
from smalign.prompting import TEMPLATE_NAME, format_prompt  # noqa: E402
from smalign.training import (  # noqa: E402
    autocast,
    batches,
    decay_param_groups,
    dump_config,
    grad_scaler,
    load_causal_lm,
    load_split,
    load_tokenizer,
    lr_lambda,
    save_checkpoint,
)
from smalign.utils import MetricsLogger, gpu_memory_summary, load_config, parse_overrides, resolve_device, set_seed, write_json  # noqa: E402


def collate(tokenizer, rows, config):
    return build_sft_batch(
        tokenizer,
        [format_prompt(row["prompt"]) for row in rows],
        [row["response"] for row in rows],
        config["max_length"],
        config.get("max_prompt_length"),
    )


@torch.no_grad()
def evaluate(model, tokenizer, rows, config, device) -> dict:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for chunk in batches(rows, config["eval_batch_size"], shuffle=False, seed=0):
        batch = collate(tokenizer, chunk, config)
        with autocast(device, config["fp16"]):
            logps, counts = response_logprobs_from_batch(model, batch)
        total_nll += -logps.sum().item()
        total_tokens += counts.sum().item()
    model.train()
    loss = total_nll / max(1, total_tokens)
    return {"loss": loss, "perplexity": math.exp(min(loss, 50)), "tokens": total_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. learning_rate=1e-5")
    args = parser.parse_args()
    config = load_config(args.config, parse_overrides(args.set))
    config["prompt_template"] = TEMPLATE_NAME
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir)
    set_seed(config["seed"])

    device = resolve_device(config["device"])
    use_fp16 = bool(config["fp16"]) and device.type == "cuda"
    print(f"[sft] device={device} fp16={use_fp16} config={config}", flush=True)

    tokenizer = load_tokenizer(config["model_name_or_path"])
    model = load_causal_lm(config["model_name_or_path"], torch.float32, device)  # fp32 master weights
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    train_rows = load_split(config["train_path"], config.get("max_train_examples"))
    val_rows = load_split(config["val_path"], config.get("max_val_examples"))
    accum = config["grad_accum_steps"]
    micro_per_epoch = math.ceil(len(train_rows) / config["per_device_batch_size"])
    total_steps = math.ceil(micro_per_epoch / accum) * config["epochs"]
    if config.get("max_steps"):
        total_steps = min(total_steps, int(config["max_steps"]))
    warmup_steps = int(round(config["warmup_ratio"] * total_steps))
    print(f"[sft] train={len(train_rows)} val={len(val_rows)} optimizer_steps={total_steps} warmup={warmup_steps}", flush=True)

    optimizer = torch.optim.AdamW(
        decay_param_groups(model, config["weight_decay"]),
        lr=config["learning_rate"],
        betas=tuple(config.get("adam_betas", (0.9, 0.999))),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(config["scheduler"], warmup_steps, total_steps))
    scaler = grad_scaler(use_fp16)
    logger = MetricsLogger(out_dir)

    step, micro, best_val = 0, 0, float("inf")
    window_nll, window_tokens = 0.0, 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    done = False
    for epoch in range(config["epochs"]):
        epoch_batches = list(batches(train_rows, config["per_device_batch_size"], shuffle=True, seed=config["seed"] + epoch))
        for index, rows in enumerate(epoch_batches):
            batch = collate(tokenizer, rows, config)
            with autocast(device, use_fp16):
                loss, n_tokens = sft_loss_from_batch(model, batch)
            scaler.scale(loss / accum).backward()
            window_nll += loss.item() * n_tokens.item()
            window_tokens += n_tokens.item()
            micro += 1

            last_micro = index == len(epoch_batches) - 1
            if micro % accum != 0 and not last_micro:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            micro = 0

            if step == 1:
                print(f"[sft] memory after first step: {gpu_memory_summary()}", flush=True)
            if step % config["log_every_steps"] == 0 or step == total_steps:
                train_loss = window_nll / max(1, window_tokens)
                elapsed = time.time() - start
                logger.log(
                    "train",
                    step,
                    loss=train_loss,
                    lr=scheduler.get_last_lr()[0],
                    grad_norm=float(grad_norm),
                    loss_scale=scaler.get_scale() if use_fp16 else 1.0,
                    epoch=epoch,
                )
                eta = elapsed / step * (total_steps - step)
                print(f"[sft] step {step}/{total_steps} loss {train_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"gnorm {float(grad_norm):.3f} elapsed {elapsed/60:.1f}m eta {eta/60:.1f}m", flush=True)
                window_nll, window_tokens = 0.0, 0

            if step % config["eval_every_steps"] == 0 or step == total_steps:
                val = evaluate(model, tokenizer, val_rows, config, device)
                is_best = val["loss"] < best_val
                logger.log("val", step, **val, is_best=is_best)
                print(f"[sft] step {step} val_loss {val['loss']:.4f} ppl {val['perplexity']:.2f} best={is_best}", flush=True)
                if is_best:
                    best_val = val["loss"]
                    save_checkpoint(model, tokenizer, out_dir / "best", {"step": step, "val_loss": best_val, "config": config})
            if step >= total_steps:
                done = True
                break
        if done:
            break

    save_checkpoint(model, tokenizer, out_dir / "final", {"step": step, "config": config})
    summary = {
        "run_name": config["run_name"],
        "optimizer_steps": step,
        "best_val_loss": best_val,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "minutes": round((time.time() - start) / 60, 1),
        "peak_memory": gpu_memory_summary(),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[sft] done: {summary}", flush=True)


if __name__ == "__main__":
    main()
