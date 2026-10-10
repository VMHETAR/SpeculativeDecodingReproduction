"""Target-only autoregressive baseline (greedy / temperature sampling)."""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_generate(
    prompt_ids: list[int],
    model,
    max_new_tokens: int,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
    seed: int = 0,
) -> tuple[list[int], dict]:
    gen = torch.Generator().manual_seed(seed) if temperature > 0 else None
    prefix = list(prompt_ids)
    out: list[int] = []
    n_forwards = 0
    while len(out) < max_new_tokens:
        inp = torch.tensor([prefix], dtype=torch.long)
        logits = model(inp).logits[0, -1]
        n_forwards += 1
        if temperature == 0.0:
            t = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            t = int(torch.multinomial(probs, 1, generator=gen).item())
        out.append(t)
        prefix.append(t)
        if eos_token_id is not None and t == eos_token_id:
            break
    return out, {"n_target_forwards": n_forwards, "n_iterations": n_forwards}
