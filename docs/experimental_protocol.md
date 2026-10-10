# Experimental protocol

## Hypotheses
- H1: for fixed draft-target pair + workload, throughput vs gamma rises to an
  empirical optimum then flattens/declines; optimum set by acceptance alpha
  and cost ratio c (Thm 3.8).
- H0: gamma has no meaningful effect on speedup within uncertainty.

## Variables
- Independent: gamma in {1,2,4,8}; prompt category {natural_language, code}.
- Dependent: throughput, latency, speedup, acceptance rate, tokens/iter,
  target forwards, exact-match, predicted-vs-observed error.
- Controlled: models (gpt2 target / distilgpt2 draft, fp32 CPU), temperature 0,
  max_new_tokens=40, seed per trial, same machine, warmup, n_trials=3.

## Metrics (see src/metrics.py for definitions)
Throughput (tok/s), end-to-end latency, median inter-token latency (=
latency/n_tokens), speedup (paired throughput ratio vs baseline, median over
prompts), acceptance_rate = accepted/proposed, mean_accepted_per_iter,
tokens_per_iter = mean_accepted+1, target/draft forwards, exact_match,
predicted speedup via Thm 3.8 with measured c, abs/rel error.

## Statistics
- Unit of analysis: prompt (n=8). Aggregate = median over prompts.
- Uncertainty: 95% percentile bootstrap CI over prompts (5000 resamples).
- Repeated trials of one prompt are averaged (median) first, never treated
  as independent samples.
- Decision rule: H1 supported if median speedup peaks at interior gamma with
  CI excluding 1.0 somewhere and declines/flattens after; else fail to reject H0.

## Acceptance criteria
- 100% greedy exact-match across all (prompt, gamma, trial) cells.
- Raw rows saved; analysis reproducible from raw JSON alone.
- All 5 required figures generated with units + uncertainty where supported.
- Limitations/deviations documented in final report.
