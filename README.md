# Small-Model Alignment: SFT + from-scratch DPO + LoRA-DPO (CS5326 PA2)

Post-training pipeline for `Qwen/Qwen2.5-0.5B`: supervised fine-tuning on 30k UltraChat examples, then two
from-scratch DPO runs (HH-RLHF and UltraFeedback), plus a TRL `DPOTrainer` + LoRA persona experiment on
`HuggingFaceTB/SmolLM2-360M-Instruct`. Evaluation uses blinded pairwise judgments from a Kaggle-proxied Gemini model.

## Repository layout

```text
submission_adapter.py        grading interface -> thin wrappers around smalign/
smalign/                     the implementation
  prompting.py               single prompt template used everywhere
  batching.py                tokenisation, EOS, truncation, padding, response-only labels
  losses.py                  response log-probs, SFT loss, DPO loss + implicit-reward metrics
  training.py                shared training-loop utilities (AMP, schedules, checkpoints)
  data.py                    dataset parsing / filtering / seeded sampling, AlpacaEval subset
  judge.py                   judge prompts, verdict parsing, win/loss summaries
  notebook.py                Kaggle helpers (fail-fast shell runner, background jobs, input lookup)
scripts/
  prepare_data.py            Task 2-4 data       train_sft.py       Task 3 (own loop)
  train_dpo.py               Task 4 (own loop)   train_lora_dpo.py  Task 5 (TRL + PEFT)
  generate.py                Tasks 2 & 6         build_pairs.py     Task 6 (blinded A/B files)
  judge_pairwise_kaggle.py   Task 7              plot_training.py / summarize_results.py   Task 8
  smoke_test_pipeline.py     3-minute end-to-end check on a tiny random model
configs/                     sft.yaml, dpo_hh.yaml, dpo_uf.yaml, lora_persona.yaml
notebooks/                   01-04 Kaggle notebooks (run in order)
eval_sets/                   hh_rlhf_eval_300_seed42.json (provided), alpaca_eval_300_seed42.json,
                             persona_style_eval_40.json
data/persona_lora/           ramsay_style_preferences.json (provided)
tests/                       public smoke tests + our edge-case and judge tests
report/                      final report (report.tex -> report.pdf)
```

## Key implementation choices

* **Prompt template** (SFT, DPO and evaluation): `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n`,
  then the response followed by `tokenizer.eos_token` (`<|endoftext|>`). There is no system prompt. Generation stops at
  `<|endoftext|>` or `<|im_end|>`. The persona track uses SmolLM2's own chat template, the same one TRL applies.
* **Masking**: prompt and response are tokenised separately. EOS is appended before truncation to `max_length`.
  Padding goes on the right. Labels are `-100` for prompt and padding positions, and masks are built from lengths
  (Qwen's pad token equals its EOS token).
* **SFT loss reduction**: mean over all unmasked response tokens in the batch.
* **DPO**: `-log σ(β[(log πθ(y_w|x) − log πθ(y_l|x)) − (log π_ref(y_w|x) − log π_ref(y_l|x))])`, computed with
  `F.logsigmoid`. Chosen and rejected share one forward pass. The reference runs under `torch.no_grad`. Reward
  accuracy uses a strict `r_w > r_l`. Reference log-probs on the validation set are cached once.

## Run it

Local checks (CPU only, no downloads apart from the tokenizer for the smoke test):

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/
python scripts/smoke_test_pipeline.py --work_dir /tmp/smoke
```

On Kaggle, run the notebooks in order. Each notebook attaches the previous one's output (manual §5.4):

| Notebook | Tasks | Accelerator | Time (T4 x2) |
|---|---|---|---|
| `01_setup_data_baseline_sft.ipynb` | setup checks, tests, data, baseline generations, SFT | GPU T4 x2 | ~2 h |
| `02_dpo_hh_uf.ipynb` | HH-RLHF DPO + UltraFeedback DPO | GPU T4 x2 | ~4.5–5 h (~2.5 h with `PARALLEL_DPO`) |
| `03_lora_generation_pairs.ipynb` | LoRA persona DPO, generations, blinded pair files | GPU T4 x2 | ~1 h |
| `04_judge_kaggle_benchmarks.ipynb` | Gemini pairwise judging, results tables | none (create from kaggle.com/benchmarks/tasks/new) | ~30–45 min |

Dependencies: Kaggle's preinstalled PyTorch is kept. The notebooks install `transformers>=4.56.2,<5` (inside the
manual's `>=4.43,<5` range; `trl==0.29.1` requires it), `datasets>=2.18,<5`, and `trl==0.29.1` + `peft` for Task 5.

## Where the results live (after copying `final_artifacts.zip` from notebook 04 into the repo)

| What | Path |
|---|---|
| Configs actually used | `configs/*.yaml`, `outputs/<run>/run_config.json` |
| Training metrics (JSONL, one line per log/eval step) | `outputs/{sft,dpo_hh,dpo_uf,lora_persona}/metrics.jsonl` |
| Run summaries | `outputs/<run>/summary.json`, `outputs/data/data_stats.json` |
| Generations (+ decoding settings) | `outputs/generations/*.json`, `*.meta.json` |
| Blinded pair files / judgments / failures / summaries | `outputs/eval/*_pairs.jsonl`, `*_judged.jsonl`, `*_judged_failures.jsonl`, `*_judged_summary.json` |
| Plots and report tables | `outputs/report_assets/*.png`, `outputs/report_assets/results.md` |
