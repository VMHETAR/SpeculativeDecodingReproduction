"""Metric definitions and measurement procedures.

- time_to_first_token_s: wall time from request start to first emitted token.
  For the baseline this equals one forward-pass latency; for speculative
  decoding it covers one full iteration (gamma drafts + 1 verification).
  Measured but reported as exploratory on CPU (see protocol).
- inter_token_latency_s: median per-token time = end_to_end_latency_s / n_tokens.
- end_to_end_latency_s: wall time for the full generation (perf_counter).
- throughput_tok_s: n_generated_tokens / end_to_end_latency_s.
- speedup: throughput(method) / throughput(baseline) per prompt (paired),
  aggregated as median over prompts.
- acceptance_rate: n_accepted_drafts / n_proposed_drafts (NOT tokens/iter).
- mean_accepted_per_iter: mean of per-iteration accepted-draft counts.
- expected_tokens_per_iter: mean_accepted_per_iter + 1 (accepted + bonus).
- n_target_forwards / n_draft_forwards: counted forward calls.
- exact_match: greedy speculative output == greedy baseline output.
- predicted_speedup: theory (Thm 3.8) from measured alpha and c.
- abs_err / rel_err: |obs - pred|, abs_err / obs.
"""

from __future__ import annotations

import statistics


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "stdev": float("nan")}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def bootstrap_ci(values: list[float], stat: str = "median", n_boot: int = 5000, seed: int = 0, ci: float = 0.95):
    """Bootstrap CI over prompts (independent units). Percentile method."""
    import random

    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    fn = statistics.median if stat == "median" else statistics.fmean
    boots = sorted(fn(rng.choices(values, k=n)) for _ in range(n_boot))
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    return (boots[int(lo_q * n_boot)], boots[int((hi_q) * n_boot) - 1])
