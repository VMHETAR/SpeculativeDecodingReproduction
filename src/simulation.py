"""Pure-Python simulation of speculative decoding (LABELLED SIMULATION).

Uses a Bernoulli(alpha) acceptance process + synthetic per-forward latencies.
Never to be reported as a real inference benchmark; used only to illustrate
the theory curve (expected tokens/iter, Thm 3.8) and to validate analysis code.
"""

from __future__ import annotations

import random

from theory import expected_tokens_per_iter, predicted_speedup


def simulate(alpha: float, gamma: int, c: float, n_tokens: int = 500, seed: int = 0) -> dict:
    rng = random.Random(seed)
    produced, iters, proposed, accepted = 0, 0, 0, 0
    per_iter = []
    while produced < n_tokens:
        n_acc = 0
        for _ in range(gamma):
            proposed += 1
            if rng.random() < alpha:
                n_acc += 1
                accepted += 1
            else:
                break
        produced += n_acc + 1
        iters += 1
        per_iter.append(n_acc)
    sim_time = iters * (gamma * c + 1.0)  # in units of target forwards
    base_time = produced * 1.0
    return {
        "label": "simulation",
        "alpha": alpha,
        "gamma": gamma,
        "c": c,
        "n_tokens": produced,
        "n_iterations": iters,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "mean_accepted_per_iter": sum(per_iter) / len(per_iter),
        "expected_tokens_per_iter_theory": expected_tokens_per_iter(alpha, gamma),
        "speedup_sim": base_time / sim_time,
        "speedup_theory": predicted_speedup(alpha, gamma, c),
    }
