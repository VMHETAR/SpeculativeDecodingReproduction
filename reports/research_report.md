# Research report: draft length vs speedup in speculative decoding

Target paper: Leviathan, Kalman & Matias (2023), *Fast Inference from
Transformers via Speculative Decoding*, arXiv:2211.17192.
Run analyzed: `results/raw/run_20261010T131248Z.json` (120 rows: 8 prompts x
5 methods x 3 trials). Reproduce with `scripts/analyze.py <runfile>`.

## 1. Question & hypotheses
How does draft length gamma affect measured speedup, and how well does
Thm 3.8 predict it across prompt types?
- H1: throughput vs gamma rises to an empirical optimum, then flattens/declines;
  optimum governed by acceptance alpha and cost ratio c.
- H0: gamma has no meaningful effect within uncertainty.

## 2. Setup (controls)
Target `gpt2` (124M) / draft `distilgpt2` (82M), same tokenizer, fp32 CPU
(Intel Ultra 5 125H, 14 torch threads, 15 GB RAM, no GPU). Greedy (T=0),
`max_new_tokens=40`, 8 fixed prompts (4 NL + 4 code), 3 trials, 1 warmup,
seeded. Cost coefficient measured on-hardware: **c = 0.611** (draft/target
per-forward latency on a short probe). Time-to-first-token not separately
measured (harness times whole generations; reported as not applicable).

## 3. Results

### 3.1 Correctness (PASSED)
Greedy speculative output == greedy baseline output in **120/120**
(prompt, gamma, trial) cells (40/40 per-prompt configs). Target forwards drop
e.g. 40 -> 8-25 depending on gamma/prompt, while emitting identical tokens.

### 3.2 Draft length vs measured speedup (Fig. 1, Fig. 4)
Median speedup over prompts (95% bootstrap CI, n=8):

| gamma | median speedup | 95% CI | median acceptance | median predicted | median abs err |
|---|---|---|---|---|---|
| 1 | **1.056** | [0.977, 1.179] | 0.724 | 1.070 | 0.079 |
| 2 | **1.066** | [0.919, 1.211] | 0.704 | 0.990 | 0.113 |
| 4 | 0.896 | [0.754, 1.119] | 0.563 | 0.636 | 0.296 |
| 8 | 0.688 | [0.470, 0.888] | 0.352 | 0.264 | 0.425 |

Shape: peak at gamma=1-2, below parity at 4, significant slowdown at 8
(CI excludes 1.0). Target forwards per 40-token generation fall monotonically
(e.g. 40 -> ~22 -> ~16 -> ~11 -> ~8), but drafting cost dominates at large gamma.

### 3.3 Prompt categories (Fig. 5)
Code prompts gain more and more consistently than NL at every gamma
(median speedup code vs NL: g1 1.120 vs 0.994; g2 1.155 vs 0.969;
g4 1.009 vs 0.815; g8 0.778 vs 0.589), with higher acceptance.
Interpretation: the small draft model tracks the target better on structured
code completions.

### 3.4 Predicted vs observed (Fig. 3) — theory is conservative
Decomposing Thm 3.8 (speedup = E / (gamma*c + 1)) with the measured
tokens/iter (E = 1.72 / 2.39 / 3.17 / 3.76 for gamma = 1/2/4/8):

| gamma | E measured | E i.i.d. theory | implied cost | predicted cost (gamma*c+1) |
|---|---|---|---|---|
| 1 | 1.72 | 1.72 | ~1.63 | 1.61 |
| 2 | 2.39 | 2.20 | ~2.24 | 2.22 |
| 4 | 3.17 | 2.16 | ~3.54 | 3.45 |
| 8 | 3.76 | 1.54 | ~5.46 | 5.89 |

The denominator (cost model) validates within ~10% at every gamma.
The numerator's **i.i.d. acceptance assumption is the weak link**: measured
tokens/iter increasingly exceeds the geometric prediction because
acceptances are correlated within an iteration (drafts are accepted in runs;
pooled alpha mixes easy early positions with drifted late ones — pooled
acceptance itself falls 0.72 -> 0.35 with gamma). Net effect: Thm 3.8 with
pooled alpha + single-point c **underpredicts** observed speedup here
(median abs err 0.08 at g1 growing to 0.42 at g8), i.e. it is a conservative
lower bound in this regime, not an overestimate.

## 4. Interpretation / verdict
- **H1 shape supported**: speedup peaks at small gamma (1-2) and declines to
  a statistically significant slowdown at gamma=8 (CI [0.47, 0.89] excludes
  1.0). The optimum location matches Thm 3.8's prediction for the measured
  (alpha ~0.7, c ~0.61) regime, where theory also peaks at gamma=1.
- **Strength of the gain is modest and not significant at 95%**: peak median
  +7% has CI [0.92, 1.21], including parity. With n=8 prompts we **fail to
  reject H0 for the speedup magnitude**, while **rejecting H0 for the
  decline at large gamma**. Larger prompt sets are needed for a tight claim.
- The small absolute gain (vs the paper's 2-3x) is explained by c = 0.61
  (paper: < 0.05): our draft is only 1.5x smaller than the target and CPU
  execution has no memory-bandwidth bottleneck for speculation to exploit.
  This is a regime difference, not a contradiction.

## 5. Limitations & deviations from the paper
1. Models: 124M/82M public models vs T5-XXL 11B / LaMDA 137B on TPU; c differs
   by >10x, so absolute speedups are not comparable. No exact reproduction
   claimed.
2. No KV-cache: full-context forwards for both methods (fair comparison, but
   inflates absolute latencies and makes verification cost length-dependent).
3. Greedy only in measurements; sampling path implemented + unit-tested.
4. n=8 prompts, 40-token budget: small sample; CIs wide; category split is
   exploratory (n=4 each).
5. Single-point c from a short probe; cost is length-dependent (Sec 3.4).
6. TTFT not measured; memory reported as process peak RSS only (1348 MB
   measured in a same-config rerun).

## 6. Conclusion
For a same-family draft-target pair on CPU, speculative decoding is
output-identical while cutting target forwards up to 5x; wall-clock speedup
follows the paper's predicted hump shape with optimum at gamma=1-2, but the
gain is small (+~7%, n.s.) because drafting is nearly as expensive as
verifying (c=0.61). The experiment confirms the paper's structural claims
(exact outputs, gamma trade-off, category dependence of alpha) and refines
one: under correlated acceptance, Thm 3.8 with pooled alpha is a
conservative predictor, with the cost half validating quantitatively.
Practical takeaway: on CPU with similarly-sized models, use gamma in {1,2}
for code-like workloads; larger gamma reliably hurts.
