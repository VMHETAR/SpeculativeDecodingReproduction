"""Greedy + sampling speculative decoding (Leviathan et al. 2023, Alg. 1).

Implementation notes / deviations from the paper:
- Full-context forward passes (no KV-cache reuse across steps). Both the
  baseline and the speculative decoder use the same forward mechanism, so the
  comparison is fair, but absolute latencies are higher than a KV-cached
  implementation. Target verification of gamma drafts is a SINGLE forward
  call (parallel scoring of all gamma+1 positions), matching the paper's
  cost model: one target run per iteration.
- Greedy mode (temperature 0): accept draft token x_i iff
  argmax p_i == x_i (equivalent to the paper's Alg. 1 with argmax-adjusted
  distributions). Produces token-for-token identical output to greedy
  target-only decoding.
- Sampling mode: exact speculative-sampling acceptance
  r < min(1, p(x)/q(x)) with residual distribution
  norm(max(0, p - q)) on rejection (paper Sec. 2.3, App. A.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SpecStepStats:
    n_iterations: int = 0
    n_draft_tokens: int = 0          # total draft tokens proposed
    n_accepted: int = 0              # total draft tokens accepted
    n_target_forwards: int = 0       # target-model forward calls
    n_draft_forwards: int = 0        # draft-model forward calls
    n_extra_tokens: int = 0          # correction/bonus tokens from target
    per_iter_accepted: list = field(default_factory=list)


def _residual_sample(p: torch.Tensor, q: torch.Tensor, generator=None) -> int:
    """Sample from norm(max(0, p - q))."""
    resid = torch.clamp(p - q, min=0.0)
    s = resid.sum()
    if s.item() <= 0:
        return int(torch.argmax(p).item())
    resid = resid / s
    return int(torch.multinomial(resid, 1, generator=generator).item())


@torch.no_grad()
def speculative_generate(
    prompt_ids: list[int],
    draft_model,
    target_model,
    gamma: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
    seed: int = 0,
) -> tuple[list[int], SpecStepStats]:
    """Run speculative decoding for up to max_new_tokens new tokens.

    Returns (new_token_ids, stats). prompt_ids is not modified.
    """
    assert gamma >= 1, "gamma must be >= 1"
    gen = torch.Generator().manual_seed(seed) if temperature > 0 else None
    stats = SpecStepStats()
    prefix = list(prompt_ids)
    out: list[int] = []

    def _next(logits: torch.Tensor, g) -> tuple[int, torch.Tensor]:
        probs = torch.softmax(logits, dim=-1)
        if temperature == 0.0:
            return int(torch.argmax(logits).item()), probs
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, 1, generator=g).item()), probs

    while len(out) < max_new_tokens:
        g = min(gamma, max_new_tokens - len(out))
        # 1. Draft gamma guesses autoregressively with Mq.
        drafts: list[int] = []
        q_probs: list = []
        for _ in range(g):
            inp = torch.tensor([prefix + drafts], dtype=torch.long)
            logits = draft_model(inp).logits[0, -1]
            stats.n_draft_forwards += 1
            tok, probs = _next(logits, gen)
            drafts.append(tok)
            q_probs.append(probs)
        # 2. Verify in parallel: single target forward over prefix+drafts.
        inp = torch.tensor([prefix + drafts], dtype=torch.long)
        all_logits = target_model(inp).logits[0]  # [L+g, V]
        stats.n_target_forwards += 1
        L = len(prefix)
        # position L-1+j corresponds to distribution p_{j+1} (next after prefix+drafts[:j])
        n_accept = 0
        for i in range(g):
            p = torch.softmax(all_logits[L - 1 + i] / temperature if temperature > 0 else all_logits[L - 1 + i], dim=-1)
            if temperature == 0.0:
                # argmax-adjusted distributions: accept iff draft == target argmax
                if drafts[i] == int(torch.argmax(all_logits[L - 1 + i]).item()):
                    n_accept += 1
                else:
                    break
            else:
                q = q_probs[i]
                r = torch.rand((), generator=gen).item()
                ratio = (p[drafts[i]] / max(q[drafts[i]].item(), 1e-12)).item()
                if r < min(1.0, ratio):
                    n_accept += 1
                else:
                    break
        stats.n_iterations += 1
        stats.n_draft_tokens += g
        stats.n_accepted += n_accept
        stats.per_iter_accepted.append(n_accept)
        # 3. Emit accepted drafts + one extra token from target.
        out.extend(drafts[:n_accept])
        prefix.extend(drafts[:n_accept])
        p_next = all_logits[L - 1 + n_accept]
        if temperature == 0.0:
            t = int(torch.argmax(p_next).item())
        else:
            if n_accept < g:
                t = _residual_sample(
                    torch.softmax(p_next / temperature, dim=-1), q_probs[n_accept], gen
                )
            else:
                t = int(torch.multinomial(torch.softmax(p_next / temperature, dim=-1), 1, generator=gen).item())
        stats.n_extra_tokens += 1
        out.append(t)
        prefix.append(t)
        if len(out) > max_new_tokens:
            # Final iteration may propose more than the remaining budget;
            # truncate to the budget (standard practice; paper Alg. 1 emits up
            # to gamma+1 per iteration).
            del out[max_new_tokens:]
            break
        if eos_token_id is not None and t == eos_token_id:
            break
        if len(out) >= max_new_tokens:
            break
    return out[:max_new_tokens], stats
