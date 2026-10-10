"""Task 8 - training plots for the report (SFT loss, DPO loss / reward accuracy / margin, LoRA-DPO).

Reads ``metrics.jsonl`` from each run directory and writes PNGs to ``--out_dir``.
Training curves are shown raw (faint) and with an exponential moving average
(alpha given in the legend); validation points are markers; the selected
"best" checkpoint is a dashed vertical line.

Example:
    python scripts/plot_training.py --sft outputs/sft --dpo outputs/dpo_hh outputs/dpo_uf \
        --lora outputs/lora_persona --out_dir outputs/report_assets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from smalign.utils import read_jsonl  # noqa: E402

EMA = 0.2
TRAIN_COLOR, VAL_COLOR, BEST_COLOR = "#2a6fb0", "#d1495b", "#555555"


def ema(values, alpha=EMA):
    out, current = [], None
    for value in values:
        current = value if current is None else alpha * value + (1 - alpha) * current
        out.append(current)
    return out


def split(rows, name):
    return [row for row in rows if row.get("split") == name]


def best_step(rows):
    steps = [row["step"] for row in split(rows, "val") if row.get("is_best")]
    return steps[-1] if steps else None


def draw(ax, rows, key, title, ylabel, run_label):
    train, val = split(rows, "train"), split(rows, "val")
    if train and key in train[0]:
        steps = [row["step"] for row in train]
        values = [row[key] for row in train]
        ax.plot(steps, values, color=TRAIN_COLOR, alpha=0.25, linewidth=1)
        ax.plot(steps, ema(values), color=TRAIN_COLOR, linewidth=2, label=f"train (EMA {EMA})")
    if val and key in val[0]:
        ax.plot([row["step"] for row in val], [row[key] for row in val], "o-", color=VAL_COLOR, label="validation")
    chosen = best_step(rows)
    if chosen is not None:
        ax.axvline(chosen, color=BEST_COLOR, linestyle="--", linewidth=1, label=f"best ckpt (step {chosen})")
    ax.set_title(f"{run_label}: {title}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_sft(run_dir: Path, out_dir: Path):
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    draw(ax, rows, "loss", "response-token NLL", "loss (nats / token)", "SFT")
    fig.tight_layout()
    path = out_dir / "sft_loss.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_dpo(run_dir: Path, out_dir: Path):
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        return None
    label = {"dpo_hh": "HH-RLHF DPO", "dpo_uf": "UltraFeedback DPO"}.get(run_dir.name, run_dir.name)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    draw(axes[0], rows, "loss", "DPO loss", "loss", label)
    axes[0].axhline(0.6931, color="gray", linestyle=":", linewidth=1)
    draw(axes[1], rows, "reward_accuracy", "reward accuracy", "fraction r_w > r_l", label)
    draw(axes[2], rows, "reward_margin", "reward margin", "mean r_w - r_l", label)
    fig.tight_layout()
    path = out_dir / f"{run_dir.name}_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_lora(run_dir: Path, out_dir: Path):
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        return None
    train = [row for row in rows if "loss" in row and "eval_loss" not in row]
    val = [row for row in rows if "eval_loss" in row]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (train_key, val_key, title) in zip(
        axes,
        [("loss", "eval_loss", "DPO loss"), ("rewards/accuracies", "eval_rewards/accuracies", "reward accuracy"),
         ("rewards/margins", "eval_rewards/margins", "reward margin")],
    ):
        if train and train_key in train[0]:
            ax.plot([r["step"] for r in train], [r[train_key] for r in train], color=TRAIN_COLOR, label="train (logged mean)")
        if val and val_key in val[0]:
            ax.plot([r["step"] for r in val], [r[val_key] for r in val], "o-", color=VAL_COLOR, label="held-out 20 pairs")
        ax.set_title(f"LoRA-DPO persona: {title}")
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "lora_persona_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft", default=None)
    parser.add_argument("--dpo", nargs="*", default=[])
    parser.add_argument("--lora", default=None)
    parser.add_argument("--out_dir", default="outputs/report_assets")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if args.sft:
        written.append(plot_sft(Path(args.sft), out_dir))
    for run in args.dpo:
        written.append(plot_dpo(Path(run), out_dir))
    if args.lora:
        written.append(plot_lora(Path(args.lora), out_dir))
    for path in written:
        if path:
            print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()
