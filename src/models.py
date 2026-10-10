"""Model loading helpers (kept separate for testability / offline use)."""

from __future__ import annotations

import torch


def load_pair(target_id: str, draft_id: str, device: str = "cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    target = AutoModelForCausalLM.from_pretrained(target_id).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(draft_id).to(device).eval()
    return tok, target, draft


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
