# Speculative Decoding — Empirical Reproduction (Resume Summary)

**Role:** Research Engineer (end-to-end, solo) · **Stack:** Python, PyTorch (CPU), Hugging Face Transformers, matplotlib, pytest
**Paper reproduced:** Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding* (ICML 2023)

## Project in one line
Built a complete, runnable ML-systems research repo that reproduces the core principles of speculative decoding and
statistically tests how draft length affects inference speedup on CPU.

## What I built
- Greedy + sampling speculative decoder (paper Alg. 1) with exact acceptance (`r < min(1, p/q)`) and residual
  resampling, plus a target-only autoregressive baseline — both sharing one forward-pass harness for fair timing.
- Experiment pipeline: 8-prompt suite (4 natural-language + 4 code) × 5 configs (baseline + γ ∈ {1, 2, 4, 8}) ×
  3 seeded trials = **120 measured generations**; raw JSON → aggregated CSVs (with 95% bootstrap CIs) → 7
  figures, all via one command (`scripts/analyze.py` + `scripts/make_readme_figures.py`).
- 7 unit tests (output-equivalence for any draft model, progress guarantee, theory identities, residual-sampler
  validity, bootstrap coverage, simulation consistency) + a no-download smoke test.

## Key results (all measured, CPU-only, gpt2-124M target / distilgpt2-82M draft, fp32)
- **Output equivalence: 120/120** speculative generations token-identical to the baseline.
- **Speedup peaks at small draft lengths, then declines:** median speedup γ=1: **1.06×** [0.98, 1.18] → γ=2: 1.07×
  → γ=4: 0.90× → γ=8: **0.69×** (significant slowdown, CI [0.47, 0.89]).
- **Target-model forward passes cut up to 5×** (40 → 8 per 40-token generation) with identical tokens.
- **Code prompts benefit more than prose** at every draft length (γ=2: 1.16× vs 0.97×; acceptance 0.70 vs 0.67).
- **Theory validation:** decomposed the paper's speedup formula and showed its cost half holds within ~10% while
  its i.i.d.-acceptance assumption underpredicts tokens/iteration at large γ (3.76 measured vs 1.54 predicted at
  γ=8) — i.e., the formula is a conservative lower bound under correlated acceptance in this regime.
- **Root-caused the small absolute gain:** measured on-hardware cost ratio c = 0.61 vs < 0.05 in the paper
  (draft only 1.5× smaller than target + no memory-bandwidth bottleneck on CPU) — a regime difference, not a
  contradiction.

## Rigor highlights (good resume / interview talking points)
- Pre-registered H1/H0 with a decision rule; reported a *nuanced* verdict (hump shape supported; +7% peak not
  significant at 95% with n=8; large-γ decline significant) instead of overstating.
- Prompts as the unit of analysis; trials median-averaged first; percentile bootstrap CIs; acceptance rate kept
  strictly separate from tokens/iteration; theoretical speedup never reported as measured.
- All figures script-generated from raw data; analysis re-runnable from saved JSON with one command.

## Suggested resume bullets (copy-ready)
- Reproduced Leviathan et al. (2023) speculative decoding on CPU (gpt2/distilgpt2, PyTorch + HF Transformers);
  verified **token-identical outputs across 120/120** speculative vs baseline generations.
- Ran a γ ∈ {1,2,4,8} ablation over 8 prompts × 3 trials; found peak **~1.07× throughput at γ=1–2** with up to
  **5× fewer target forward passes**, declining to 0.69× at γ=8 (95% bootstrap CIs).
- Validated the paper's speedup model against measurements: cost term accurate within ~10%; showed its i.i.d.
  assumption underpredicts tokens/iteration under correlated acceptance (3.76 vs 1.54 at γ=8).
- Shipped a one-command reproducible pipeline (tests → benchmark → analysis → figures + report) with 7 pytest
  correctness/theory tests and pre-registered hypotheses.
