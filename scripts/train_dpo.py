"""Task 4 - from-scratch DPO (one implementation, used for both HH-RLHF and UltraFeedback).

* policy  = SFT best checkpoint, trainable, fp32 weights + fp16 autocast + GradScaler (cuda:0)
* reference = a second, frozen copy of the same checkpoint, run under ``torch.no_grad`` (cuda:1)
* loss = mean(-log sigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))) over response-only logps
* tracked: DPO loss, implicit rewards, strict reward accuracy, reward margin (train + val)
* checkpoints: ``best`` by validation reward accuracy (ties -> larger margin) and ``final``

Reference log-probs for the validation set never change, so they are computed
once at start-up and cached.

Example:
    python scripts/train_dpo.py --config configs/dpo_hh.yaml --set policy_init=/path/to/sft/best reference_path=/path/to/sft/best
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from smalign.batching import build_dpo_batch  # noqa: E402
from smalign.losses import dpo_loss_from_logps, dpo_metrics, policy_dpo_logps, reference_dpo_logps  # noqa: E402
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

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def collate(tokenizer, rows, config):
    return build_dpo_batch(
        tokenizer,
        [format_prompt(row["prompt"]) for row in rows],
        [row["chosen"] for row in rows],
        [row["rejected"] for row in rows],
        config["max_length"],
        config.get("max_prompt_length"),
    )


@torch.no_grad()
def reference_logps_for(reference, tokenizer, rows, config, device) -> tuple[torch.Tensor, torch.Tensor]:
    chosen, rejected = [], []
    for chunk in batches(rows, config["eval_batch_size"], shuffle=False, seed=0):
        with autocast(device, config["fp16"]):
            c, r = reference_dpo_logps(reference, collate(tokenizer, chunk, config))
        chosen.append(c.float().cpu())
        rejected.append(r.float().cpu())
    return torch.cat(chosen), torch.cat(rejected)


@torch.no_grad()
def evaluate(policy, tokenizer, rows, ref_chosen, ref_rejected, config, device) -> dict:
    policy.eval()
    chosen, rejected = [], []
    for chunk in batches(rows, config["eval_batch_size"], shuffle=False, seed=0):
        with autocast(device, config["fp16"]):
            c, r = policy_dpo_logps(policy, collate(tokenizer, chunk, config))
        chosen.append(c.float().cpu())
        rejected.append(r.float().cpu())
    policy.train()
    stats = dpo_metrics(torch.cat(chosen), torch.cat(rejected), ref_chosen, ref_rejected, config["beta"])
    return {
        "loss": stats["loss"].mean().item(),
        "reward_accuracy": stats["reward_accuracy"].mean().item(),
        "reward_margin": stats["reward_margin"].mean().item(),
        "reward_chosen": stats["reward_chosen"].mean().item(),
        "reward_rejected": stats["reward_rejected"].mean().item(),
        "reward_ties": (stats["reward_margin"] == 0).float().mean().item(),
    }


def build_optimizer(model, config):
    if config["optimizer"].lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if config["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(decay_param_groups(model, config["weight_decay"]), lr=config["learning_rate"])
    raise ValueError(f"unknown optimizer {config['optimizer']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. beta=0.2")
    args = parser.parse_args()
    config = load_config(args.config, parse_overrides(args.set))
    config["prompt_template"] = TEMPLATE_NAME
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, out_dir)
    set_seed(config["seed"])

    policy_device = resolve_device(config["policy_device"])
    reference_device = resolve_device(config["reference_device"])
    use_fp16 = bool(config["fp16"]) and policy_device.type == "cuda"
    config["fp16"] = use_fp16
    print(f"[dpo] policy on {policy_device}, reference on {reference_device}, fp16={use_fp16}", flush=True)
    print(f"[dpo] config={config}", flush=True)

    tokenizer = load_tokenizer(config["policy_init"])
    policy = load_causal_lm(config["policy_init"], torch.float32, policy_device)
    if config.get("gradient_checkpointing"):
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.train()
    reference = load_causal_lm(config["reference_path"], DTYPES[config["reference_dtype"]], reference_device)
    reference.eval()
    reference.requires_grad_(False)

    train_rows = load_split(config["train_path"], config.get("max_train_examples"))
    val_rows = load_split(config["val_path"], config.get("max_val_examples"))
    accum = config["grad_accum_steps"]
    total_steps = math.ceil(math.ceil(len(train_rows) / config["per_device_batch_size"]) / accum) * config["epochs"]
    if config.get("max_steps"):
        total_steps = min(total_steps, int(config["max_steps"]))
    print(f"[dpo] train pairs={len(train_rows)} val pairs={len(val_rows)} optimizer_steps={total_steps}", flush=True)

    optimizer = build_optimizer(policy, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(config["scheduler"], config.get("warmup_steps", 0), total_steps)
    )
    scaler = grad_scaler(use_fp16)
    logger = MetricsLogger(out_dir)

    t0 = time.time()
    ref_val_chosen, ref_val_rejected = reference_logps_for(reference, tokenizer, val_rows, config, reference_device)
    print(f"[dpo] cached reference val logps in {time.time() - t0:.0f}s", flush=True)

    best = {"reward_accuracy": -1.0, "reward_margin": -float("inf"), "step": None}

    def run_eval(step: int) -> None:
        val = evaluate(policy, tokenizer, val_rows, ref_val_chosen, ref_val_rejected, config, policy_device)
        better = step > 0 and (
            val["reward_accuracy"] > best["reward_accuracy"]
            or (val["reward_accuracy"] == best["reward_accuracy"] and val["reward_margin"] > best["reward_margin"])
        )
        logger.log("val", step, **val, is_best=better)
        print(f"[dpo] step {step} val loss {val['loss']:.4f} acc {val['reward_accuracy']:.3f} "
              f"margin {val['reward_margin']:.4f} best={better}", flush=True)
        if better:
            best.update(reward_accuracy=val["reward_accuracy"], reward_margin=val["reward_margin"], step=step)
            save_checkpoint(policy, tokenizer, out_dir / "best", {"step": step, "val": val, "config": config})

    if config.get("eval_at_start"):
        run_eval(0)

    step, micro = 0, 0
    window: dict[str, list[float]] = {}
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    done = False
    for epoch in range(config["epochs"]):
        epoch_batches = list(batches(train_rows, config["per_device_batch_size"], shuffle=True, seed=config["seed"] + epoch))
        for index, rows in enumerate(epoch_batches):
            batch = collate(tokenizer, rows, config)
            with autocast(reference_device, use_fp16):
                ref_chosen, ref_rejected = reference_dpo_logps(reference, batch)
            with autocast(policy_device, use_fp16):
                pol_chosen, pol_rejected = policy_dpo_logps(policy, batch)
            loss = dpo_loss_from_logps(pol_chosen, pol_rejected, ref_chosen, ref_rejected, config["beta"])
            scaler.scale(loss / accum).backward()

            stats = dpo_metrics(pol_chosen.detach(), pol_rejected.detach(), ref_chosen, ref_rejected, config["beta"])
            for key, value in stats.items():
                window.setdefault(key, []).extend(value.float().cpu().tolist())
            micro += 1

            last_micro = index == len(epoch_batches) - 1
            if micro % accum != 0 and not last_micro:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            micro = 0

            if step == 1:
                print(f"[dpo] memory after first step: {gpu_memory_summary()}", flush=True)
            if step % config["log_every_steps"] == 0 or step == total_steps:
                means = {key: sum(values) / len(values) for key, values in window.items()}
                logger.log("train", step, **means, grad_norm=float(grad_norm), lr=scheduler.get_last_lr()[0],
                           loss_scale=scaler.get_scale() if use_fp16 else 1.0, epoch=epoch)
                elapsed = time.time() - start
                print(f"[dpo] step {step}/{total_steps} loss {means['loss']:.4f} acc {means['reward_accuracy']:.3f} "
                      f"margin {means['reward_margin']:.4f} gnorm {float(grad_norm):.2f} "
                      f"elapsed {elapsed/60:.1f}m eta {elapsed/step*(total_steps-step)/60:.1f}m", flush=True)
                window = {}
            if step % config["eval_every_steps"] == 0 or step == total_steps:
                run_eval(step)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    if config.get("save_final", True):
        save_checkpoint(policy, tokenizer, out_dir / "final", {"step": step, "config": config})
    summary = {
        "run_name": config["run_name"],
        "optimizer_steps": step,
        "best_step": best["step"],
        "best_val_reward_accuracy": best["reward_accuracy"],
        "best_val_reward_margin": best["reward_margin"],
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "minutes": round((time.time() - start) / 60, 1),
        "peak_memory": gpu_memory_summary(),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[dpo] done: {summary}", flush=True)


if __name__ == "__main__":
    main()
