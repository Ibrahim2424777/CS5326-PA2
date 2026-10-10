# Instruction Tuning and DPO on Qwen2.5-0.5B — CS5326 PA2 Report

**Name:** TODO · **Repository:** TODO (GitHub URL) · **Judge model:** TODO (exact name printed by notebook 04)

> Fill every `TODO` from `outputs/report_assets/results.md` (auto-generated) and the plots in
> `outputs/report_assets/`. Keep the PDF ≤ 8 pages including figures. Delete this note.

## 1. Repository, models and data (§14.1)

* **Base checkpoint:** `Qwen/Qwen2.5-0.5B` (494M parameters, tied embeddings). **Persona model:** `HuggingFaceTB/SmolLM2-360M-Instruct`.
* **Seeds:** 42 for every data shuffle, training run, pair-order randomisation and the AlpacaEval subset.

| Track | Source | Train / Val | Filtering |
|---|---|---|---|
| SFT | `HuggingFaceH4/ultrachat_200k` `train_sft` / `test_sft`, first user→assistant turn | 30,000 / 1,000 | non-empty; formatted prompt ≤ 256 tokens; prompts deduplicated; exact eval prompts excluded |
| HH-RLHF DPO | `Anthropic/hh-rlhf` `train` / `test` | 15,000 / 1,000 | single-turn only (one Human + one Assistant turn); chosen ≠ rejected; prompt ≤ 256 tokens |
| UltraFeedback DPO | `HuggingFaceH4/ultrafeedback_binarized` `train_prefs` / `test_prefs` | 15,000 / 1,000 | single-turn; score ties dropped; chosen ≠ rejected; prompt ≤ 256 tokens |
| LoRA persona DPO | `data/persona_lora/ramsay_style_preferences.json` | 200 / 20 (seeded split) | none |

Filter counts are in `outputs/data/data_stats.json` (TODO: quote, e.g. "N of M HH pairs were single-turn").
Sampling: a seeded shuffle, then the first N rows that pass the filters.

**Fixed evaluation sets:** AlpacaEval subset `eval_sets/alpaca_eval_300_seed42.json`
(`random.Random(42).sample(range(805), 300)`, kept in original order); the provided HH set
`eval_sets/hh_rlhf_eval_300_seed42.json` (300 prompts, 10 categories); the persona set
`eval_sets/persona_style_eval_40.json` (10 cooking, 10 coding, 10 general help, 10 safety-sensitive; written for
this report and disjoint from the 220 training prompts).

**Prompt format** (identical for SFT training, DPO training and evaluation):
`<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|endoftext|>`. There is no system
prompt. The prompt and response are tokenised separately, and EOS is appended before truncation. The persona track
uses SmolLM2's built-in chat template, applied by TRL in training and by `generate.py --template chat_template`
at evaluation.

## 2. Experimental configuration (§14.2)

TODO: paste the "Experimental configuration" table from `results.md`. Values in the guide configuration:

| | SFT | HH-RLHF DPO | UltraFeedback DPO | LoRA-DPO |
|---|---|---|---|---|
| Trainable params | full (494M) | full | full | LoRA r=16, α=32, dropout 0.05, q/k/v/o/gate/up/down (TODO count) |
| Optimizer / peak LR | AdamW (0.9, 0.999) / 2e-5 | RMSprop / 1e-6 | RMSprop / 1e-6 | AdamW / 5e-5 |
| Schedule / warmup | cosine / 3% | constant / none | constant / none | cosine / 10% |
| Weight decay / clip | 0.1 (not on bias/norm) / 1.0 | 0 / 1.0 | 0 / 1.0 | 0 / 1.0 |
| Batch × accum = eff. | 2 × 16 = 32 | 1 × 32 = 32 | 1 × 32 = 32 | 4 × 2 = 8 |
| Precision | fp32 weights, fp16 autocast + GradScaler | same; reference fp32 under the same autocast | same | fp32 + fp16 AMP |
| Max length (prompt) | 512 (256) | 512 (256) | 512 (256) | 512 |
| Epochs / steps | 1 / TODO (≈938) | 1 / TODO (≈469) | 1 / TODO (≈469) | 3 / 75 |
| β / reference | – | 0.1 / frozen `outputs/sft/best` (step TODO) | 0.1 / same | 0.1 / SmolLM2 base (adapter disabled) |
| Checkpoint selection | best val loss | best val reward accuracy (ties → margin) | same | final |

**Generation (all comparisons):** greedy (temperature / top-p not used), `repetition_penalty=1.1`,
`max_new_tokens=512` (256 for persona), fp32, stop at EOS / `<|im_end|>`. SFT and DPO always use identical settings.

## 3. Parameter choices and deviations (§14.3)

| Item | Guide | Used | Why / effect | Type |
|---|---|---|---|---|
| Reference model precision | fp16 allowed | fp32 + same fp16 autocast as the policy | The policy and the reference give bit-identical log-probs at step 0, so the step-0 val loss is exactly ln 2 and reward accuracy is exactly 0 (verified). fp16 weights add noise that makes early reward accuracy ≈ 0.5 by chance. The cost is ~1 GB more on cuda:1. | correctness |
| Decoding | greedy | greedy + repetition_penalty 1.1 | TODO: evidence from the base/SFT sanity check (repetition loops). Applied identically to both sides of every comparison. | quality |
| Prompt-length filter | max prompt 256 for DPO | also applied when sampling SFT and DPO data (filtered, not truncated) | Keeps ≥ 256 tokens for responses, so EOS is learned more often. | data |
| transformers version | ≥ 4.43, < 5 | ≥ 4.56.2, < 5 | Required by `trl==0.29.1` (newest TRL pinned to transformers < 5). | resource / compat |
| LoRA-DPO LR | not given | 5e-5 | LoRA updates a small low-rank subspace, so a 1e-6 LR barely moves it. TODO: pilot evidence if you ran one. | performance |
| (only if used) `PARALLEL_DPO` | policy cuda:0, reference cuda:1 | each run on its own GPU with policy and reference together | Halved wall time. Memory TODO GiB per GPU. | resource |
| TODO: any other change | | | | |

## 4. Training behaviour (§14.4)

Figures: `sft_loss.png`, `dpo_hh_curves.png`, `dpo_uf_curves.png`, `lora_persona_curves.png`. Light lines are
raw logged means, bold lines are EMA (α = 0.2), dots are validation points, and the dashed line is the selected
checkpoint.

TODO: 3–5 sentences per run on convergence, stability (grad norm, loss-scale drops), over-fitting, and why the
selected checkpoint was chosen. Include the step-0 DPO sanity values (val loss 0.6931, reward accuracy 0). Do not
infer response quality from loss.

## 5. DPO-versus-SFT results (§14.5)

TODO: paste the judge table from `results.md`. For each comparison report n, wins/ties/losses, win rate with ties
counted as losses, with ties counted as ½, and excluding ties, plus unknown/failed counts and retries.
Invalid-judgment handling: a pair with a missing or unparseable verdict is retried with the rest of its batch (up
to 3 attempts). If it is still unresolved it is reported as *unknown* (or *failure* for proxy errors), excluded
from the win-rate denominators, and logged in `*_judged_failures.jsonl`.

| Comparison | Eval set | n | W | T | L | Win rate (ties ½) |
|---|---|---|---|---|---|---|
| UltraFeedback-DPO vs SFT | AlpacaEval-300 (seed 42) | TODO | | | | |
| HH-RLHF-DPO vs SFT | HH synthetic eval (300) | TODO | | | | |

**LoRA persona (reported separately):** LoRA-DPO vs SmolLM2-Instruct on 40 prompts. The *style* verdict (persona
faithfulness) and the *quality* verdict (useful, correct, safe) are separate judge outputs. TODO: table plus the
per-category breakdown, safety-sensitive prompts in particular.

## 6. Interpretation and error analysis (§14.6)

TODO. For each eval set, answer which model won and by how much, separating **observations** (numbers, per-category
tables, length statistics from `results.md`) from **interpretation**. Relate the result to the preference data
(HH = harmlessness/refusals; UF = helpfulness/detail), the SFT baseline, β = 0.1, ~469 steps at a fixed LR of
1e-6, reward diagnostics (final margin/accuracy), and verbosity changes (mean words, share hitting the token limit).
Use 2–3 short examples from the "Representative examples" section of `results.md`: improvements, regressions and
cases with little change. Check for position bias (verdict A/B counts).

## 7. Limitations and reproducibility (§14.7)

* There is a single LLM judge, with possible self-preference, length and position biases. Order was randomised
  per pair, but each pair was judged in one order only.
* 300 prompts per set and a single seed, so no variance estimate across seeds. The persona set has only 40 prompts.
* Compute limits: one epoch, a small fixed DPO LR, and a 512-token context (TODO: % of SFT responses truncated).
* Reproduce with notebooks `01`→`04`. Configs: `configs/*.yaml` and `outputs/<run>/run_config.json`. Metrics:
  `outputs/<run>/metrics.jsonl`. Generations: `outputs/generations/`. Judgments and summaries: `outputs/eval/`.
  Tables: `outputs/report_assets/results.md`.

## References

Rafailov et al. 2023 (DPO); Ouyang et al. 2022 (InstructGPT); Bai et al. 2022 (HH-RLHF); Dubois et al. 2024
(length-controlled AlpacaEval); Hugging Face Transformers / TRL / PEFT documentation; Kaggle Benchmarks documentation.
