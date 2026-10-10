# Paper notes: Leviathan, Kalman & Matias (2023), arXiv:2211.17192
*Fast Inference from Transformers via Speculative Decoding* (ICML 2023).

## Problem
Autoregressive decoding of K tokens needs K serial target-model runs.

## Algorithm (Alg. 1, greedy + sampling)
1. Draft model Mq autoregressively proposes gamma tokens x_1..x_gamma.
2. Target model Mp scores prefix + all drafts **in parallel** (one batched run
   yielding p_1..p_{gamma+1}).
3. Accept the longest prefix with r_i < p_i(x_i)/q_i(x_i) (greedy: draft ==
   target argmax); emit accepted drafts + one extra token from Mp
   (residual distribution norm(max(0, p-q)) after a rejection, bonus token if
   all accepted).
4. Guarantees: >= 1 new token per iteration; output distribution identical to
   Mp alone (App. A.1).

## Theory
- Acceptance rate beta per position; alpha = E[beta] = E[min(p,q)] (Thm 3.5,
  Cor. 3.6); i.i.d. assumption gives E[#tokens/iter] = (1-alpha^{g+1})/(1-alpha).
- Thm 3.8: expected walltime speedup = E / (gamma*c + 1), c = t(Mq)/t(Mp).
  If alpha > c, gamma=1 already improves by (1+alpha)/(1+c) (Cor. 3.9).
- Larger gamma raises expected tokens but also drafting cost -> interior
  optimum gamma* (Fig. 3); oracle-varying gamma adds up to ~60% (Sec. 3.5).
- Ops increase factor (Thm 3.11); memory traffic drops with E (Sec. 3.4).

## Reported results (reproduced claims, NOT our measurements)
- T5-XXL 11B + T5-small 77M: EnDe 3.4x (argmax, g=7, a=0.75) / 2.6x (T=1);
  CNNDM 3.1x / 2.3x (Table 2). Best draft ~2 orders of magnitude smaller.
- alpha 0.5-0.9 for small-model drafts; argmax > sampling; even bigram
  drafts give alpha ~0.2 -> 1.25x at c=0 (Table 3, Sec. 3.6).

## Reproduction checklist
- [ ] Greedy speculative == greedy baseline token-for-token.
- [ ] Sampling uses acceptance + residual distribution (exact).
- [ ] Report alpha (acceptance rate) separately from E (tokens/iter).
- [ ] Measure c on the actual hardware; predict via Thm 3.8; compare.
- [ ] Speedup from wall-clock throughput, not from theory.
- [ ] gamma ablation incl. optimum + diminishing returns.
- [ ] Controls: same prompts, budgets, precision, hardware; warmup; trials.
