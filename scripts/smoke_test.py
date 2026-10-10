"""Smoke test: tiny random GPT-2 pair, no downloads. Verifies equivalence + speedup path."""

import sys, time
sys.path.insert(0, "src")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from baseline import greedy_generate
from speculative import speculative_generate

torch.manual_seed(0)
cfg_d = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=256)
cfg_t = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=256)
draft = GPT2LMHeadModel(cfg_d).eval()
target = GPT2LMHeadModel(cfg_t).eval()
# share weights -> alpha should be high
target.load_state_dict(draft.state_dict())

prompt = [10, 20, 30, 40]
base, _ = greedy_generate(prompt, target, max_new_tokens=12)
ok = True
for g in (1, 2, 4):
    spec, stats = speculative_generate(prompt, draft, target, gamma=g, max_new_tokens=12)
    match = spec == base
    ok &= match
    print(f"gamma={g} match={match} accepted={stats.n_accepted}/{stats.n_draft_tokens} "
          f"tgt_fwds={stats.n_target_forwards}")
print("SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
