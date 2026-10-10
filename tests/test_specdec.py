import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from baseline import greedy_generate
from speculative import speculative_generate, _residual_sample
from theory import expected_tokens_per_iter, predicted_speedup
from metrics import bootstrap_ci
from simulation import simulate


def tiny_pair(shared=True):
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=128)
    torch.manual_seed(0)
    d = GPT2LMHeadModel(cfg).eval()
    torch.manual_seed(1)
    t = GPT2LMHeadModel(cfg).eval()
    if shared:
        t.load_state_dict(d.state_dict())
    return d, t


def test_greedy_equivalence_shared_weights():
    d, t = tiny_pair(shared=True)
    prompt = [5, 6, 7]
    base, _ = greedy_generate(prompt, t, max_new_tokens=10)
    for g in (1, 2, 4):
        spec, _ = speculative_generate(prompt, d, t, gamma=g, max_new_tokens=10)
        assert spec == base, f"gamma={g}: {spec} != {base}"


def test_greedy_equivalence_distinct_weights():
    # Equivalence must hold for ANY draft model under greedy verification.
    d, t = tiny_pair(shared=False)
    prompt = [5, 6, 7]
    base, _ = greedy_generate(prompt, t, max_new_tokens=10)
    for g in (1, 3):
        spec, _ = speculative_generate(prompt, d, t, gamma=g, max_new_tokens=10)
        assert spec == base, f"gamma={g}: {spec} != {base}"


def test_progress_guarantee():
    d, t = tiny_pair(shared=False)
    out, stats = speculative_generate([1, 2], d, t, gamma=4, max_new_tokens=9)
    assert len(out) == 9  # truncated to budget; final iter may emit up to g+1
    assert stats.n_accepted + stats.n_extra_tokens >= 9
    assert stats.n_target_forwards <= 9  # at least 1 token per target forward


def test_theory_monotone_and_cap():
    assert abs(expected_tokens_per_iter(0.0, 4) - 1.0) < 1e-9
    assert abs(expected_tokens_per_iter(0.75, 1) - 1.75) < 1e-9
    assert expected_tokens_per_iter(0.9, 8) <= 9.0
    assert predicted_speedup(0.75, 1, 0.0) == expected_tokens_per_iter(0.75, 1)


def test_residual_is_valid_distribution_sample():
    torch.manual_seed(0)
    V = 20
    p = torch.softmax(torch.randn(V), -1)
    q = torch.softmax(torch.randn(V), -1)
    for _ in range(50):
        t = _residual_sample(p, q)
        assert 0 <= t < V


def test_bootstrap_ci_covers_median():
    lo, hi = bootstrap_ci([1.0, 1.1, 0.9, 1.05, 0.95], n_boot=500, seed=0)
    import statistics
    m = statistics.median([1.0, 1.1, 0.9, 1.05, 0.95])
    assert lo <= m <= hi


def test_simulation_labelled_and_consistent():
    s = simulate(0.7, 3, 0.05, n_tokens=200, seed=0)
    assert s["label"] == "simulation"
    assert abs(s["acceptance_rate"] - 0.7) < 0.15
    assert abs(s["speedup_sim"] - s["speedup_theory"]) / s["speedup_theory"] < 0.25
