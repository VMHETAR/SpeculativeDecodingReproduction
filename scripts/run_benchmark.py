"""Full benchmark: baseline + speculative (gammas) over prompt set, repeated trials.

Saves one JSON row per (prompt, method, trial) to results/raw/run_<ts>.json.
Usage: .venv/bin/python scripts/run_benchmark.py [--config configs/experiment.yaml]
"""

import argparse, csv, json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import torch
import yaml

from baseline import greedy_generate
from speculative import speculative_generate
from prompts import PROMPTS
from models import load_pair, count_params


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = cfg.get("device", "cpu")
    torch.manual_seed(cfg.get("seed", 1234))

    print(f"Loading {cfg['target_model']} + {cfg['draft_model']} ...", flush=True)
    tok, target, draft = load_pair(cfg["target_model"], cfg["draft_model"], device)
    print(f"params: target={count_params(target)/1e6:.1f}M draft={count_params(draft)/1e6:.1f}M", flush=True)
    eos = tok.eos_token_id
    prompts = PROMPTS[: args.max_prompts] if args.max_prompts else PROMPTS

    # micro-benchmark per-forward latency for cost coefficient c
    probe = tok.encode("Hello world", return_tensors="pt")[0].tolist()
    def fwd_lat(model, n=5):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(torch.tensor([probe]))
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2]
    c = fwd_lat(draft) / max(fwd_lat(target), 1e-9)
    print(f"cost coefficient c (draft/target per-forward) = {c:.4f}", flush=True)

    rows = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # warmup
    for _ in range(cfg.get("n_warmup", 1)):
        pid = tok.encode(prompts[0]["text"], return_tensors="pt")[0].tolist()
        greedy_generate(pid, target, max_new_tokens=8, eos_token_id=eos)
        speculative_generate(pid, draft, target, gamma=2, max_new_tokens=8, eos_token_id=eos)

    for p in prompts:
        pids = tok.encode(p["text"], return_tensors="pt")[0].tolist()
        for trial in range(cfg.get("n_trials", 3)):
            seed = cfg.get("seed", 0) + trial
            (bout, binfo), dt = timed(greedy_generate, pids, target,
                                      cfg["max_new_tokens"], cfg.get("temperature", 0.0), eos, seed)
            rows.append({"prompt_id": p["id"], "category": p["category"], "method": "baseline",
                         "gamma": 0, "trial": trial, "prompt_text": p["text"],
                         "output_text": tok.decode(bout, skip_special_tokens=True),
                         "output_ids": bout, "n_tokens": len(bout),
                         "latency_s": dt, "throughput": len(bout) / dt if dt > 0 else 0,
                         "n_target_forwards": binfo["n_target_forwards"], "n_draft_forwards": 0,
                         "n_accepted": None, "n_proposed": None, "exact_match": True})
            for g in cfg["gammas"]:
                (sout, st), dt2 = timed(speculative_generate, pids, draft, target, g,
                                        cfg["max_new_tokens"], cfg.get("temperature", 0.0), eos, seed)
                rows.append({"prompt_id": p["id"], "category": p["category"], "method": f"spec_g{g}",
                             "gamma": g, "trial": trial, "prompt_text": p["text"],
                             "output_text": tok.decode(sout, skip_special_tokens=True),
                             "output_ids": sout, "n_tokens": len(sout),
                             "latency_s": dt2, "throughput": len(sout) / dt2 if dt2 > 0 else 0,
                             "n_target_forwards": st.n_target_forwards, "n_draft_forwards": st.n_draft_forwards,
                             "n_accepted": st.n_accepted, "n_proposed": st.n_draft_tokens,
                             "n_iterations": st.n_iterations,
                             "mean_accepted_per_iter": (sum(st.per_iter_accepted) / len(st.per_iter_accepted)) if st.per_iter_accepted else 0,
                             "exact_match": sout == bout})
            print(f"done {p['id']} trial {trial}", flush=True)

    os.makedirs("results/raw", exist_ok=True)
    out = args.out or f"results/raw/run_{ts}.json"
    meta = {"config": cfg, "cost_c": c, "n_prompts": len(prompts),
            "device": device, "torch_threads": torch.get_num_threads()}
    with open(out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(f"saved {len(rows)} rows -> {out}")
    bad = [r for r in rows if r["method"] != "baseline" and not r["exact_match"]]
    print(f"equivalence failures: {len(bad)}/{len(rows)}")
    for r in bad:
        print("  MISMATCH", r["prompt_id"], r["method"], r["trial"])


if __name__ == "__main__":
    main()
