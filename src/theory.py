"""Theoretical analysis from Leviathan et al. (2023).

Eq. (1):  E[#tokens per iteration] = (1 - alpha^(gamma+1)) / (1 - alpha)
Thm 3.8:  expected walltime speedup = E / (gamma * c + 1),
           where c = time(draft forward) / time(target forward),
           alpha = mean draft-token acceptance probability.
"""

from __future__ import annotations


def expected_tokens_per_iter(alpha: float, gamma: int) -> float:
    if alpha >= 1.0:
        return float(gamma + 1)
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def predicted_speedup(alpha: float, gamma: int, c: float) -> float:
    return expected_tokens_per_iter(alpha, gamma) / (gamma * c + 1.0)


def optimal_gamma(alpha: float, c: float, gamma_max: int = 16) -> tuple[int, float]:
    best_g, best_s = 1, predicted_speedup(alpha, 1, c)
    for g in range(2, gamma_max + 1):
        s = predicted_speedup(alpha, g, c)
        if s > best_s:
            best_g, best_s = g, s
    return best_g, best_s
