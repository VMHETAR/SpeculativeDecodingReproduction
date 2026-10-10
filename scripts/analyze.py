"""Analyze raw run JSON -> processed CSVs + figures + console summary.

Usage: .venv/bin/python scripts/analyze.py results/raw/run_*.json [--outdir results/processed]
"""

import argparse, csv, glob, json, os, statistics, sys
sys.path.insert(0, "src")

from metrics import summarize, bootstrap_ci
from theory import expected_tokens_per_iter, predicted_speedup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = json.load(open(path))
    return d["meta"], d["rows"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runfile")
    ap.add_argument("--outdir", default="results/processed")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()
    meta, rows = load(args.runfile)
    c = meta["cost_c"]
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    # per (prompt, method): median latency/throughput over trials
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["prompt_id"], r["category"], r["method"], r["gamma"])].append(r)
    per_prompt = []
    for (pid, cat, method, g), rs in sorted(agg.items()):
        lat = statistics.median(r["latency_s"] for r in rs)
        thr = statistics.median(r["throughput"] for r in rs)
        ntok = statistics.median(r["n_tokens"] for r in rs)
        tf = statistics.median(r["n_target_forwards"] for r in rs)
        row = {"prompt_id": pid, "category": cat, "method": method, "gamma": g,
               "latency_med": lat, "throughput_med": thr, "n_tokens_med": ntok,
               "n_target_forwards_med": tf,
               "exact_match_all": all(r["exact_match"] for r in rs)}
        if method != "baseline":
            acc = sum(r["n_accepted"] for r in rs)
            prop = sum(r["n_proposed"] for r in rs)
            row["acceptance_rate"] = acc / prop if prop else 0.0
            row["mean_accepted_per_iter"] = statistics.fmean(
                r["mean_accepted_per_iter"] for r in rs)
            row["tokens_per_iter"] = row["mean_accepted_per_iter"] + 1.0
        per_prompt.append(row)

    base_thr = {(r["prompt_id"]): r["throughput_med"] for r in per_prompt if r["method"] == "baseline"}
    base_lat = {(r["prompt_id"]): r["latency_med"] for r in per_prompt if r["method"] == "baseline"}
    for r in per_prompt:
        if r["method"] != "baseline":
            r["speedup"] = r["throughput_med"] / base_thr[r["prompt_id"]] if base_thr[r["prompt_id"]] else 0
            r["predicted_speedup"] = predicted_speedup(r["acceptance_rate"], r["gamma"], c)
            r["abs_err"] = abs(r["speedup"] - r["predicted_speedup"])
            r["rel_err"] = r["abs_err"] / r["speedup"] if r["speedup"] else float("nan")

    fields = list(per_prompt[0].keys())
    for r in per_prompt:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(f"{args.outdir}/per_prompt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(per_prompt)

    # aggregate by gamma: median speedup over prompts + bootstrap CI
    summary = []
    gammas = sorted({r["gamma"] for r in per_prompt if r["method"] != "baseline"})
    for g in gammas:
        sp = [r["speedup"] for r in per_prompt if r["gamma"] == g]
        acc = [r["acceptance_rate"] for r in per_prompt if r["gamma"] == g]
        pred = [r["predicted_speedup"] for r in per_prompt if r["gamma"] == g]
        ae = [r["abs_err"] for r in per_prompt if r["gamma"] == g]
        lo, hi = bootstrap_ci(sp)
        summary.append({"gamma": g, "median_speedup": statistics.median(sp),
                        "ci_lo": lo, "ci_hi": hi,
                        "median_acceptance": statistics.median(acc),
                        "median_predicted": statistics.median(pred),
                        "median_abs_err": statistics.median(ae),
                        "n_prompts": len(sp)})
    with open(f"{args.outdir}/summary_by_gamma.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # category breakdown
    cats = sorted({r["category"] for r in per_prompt})
    with open(f"{args.outdir}/summary_by_category.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "gamma", "median_speedup", "median_acceptance", "n"])
        w.writeheader()
        for cat in cats:
            for g in gammas:
                s = [r["speedup"] for r in per_prompt if r["category"] == cat and r["gamma"] == g]
                a = [r["acceptance_rate"] for r in per_prompt if r["category"] == cat and r["gamma"] == g]
                if s:
                    w.writerow({"category": cat, "gamma": g, "median_speedup": statistics.median(s),
                                "median_acceptance": statistics.median(a), "n": len(s)})

    # ---- figures ----
    gs = [s["gamma"] for s in summary]
    fig, ax = plt.subplots()
    ax.errorbar(gs, [s["median_speedup"] for s in summary],
                yerr=[[s["median_speedup"] - s["ci_lo"] for s in summary],
                      [s["ci_hi"] - s["median_speedup"] for s in summary]],
                marker="o", capsize=4)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("draft length (gamma)"); ax.set_ylabel("median speedup vs baseline")
    ax.set_title("Draft length vs measured speedup (95% bootstrap CI over prompts)")
    fig.savefig(f"{args.figdir}/1_speedup_vs_gamma.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(gs, [s["median_acceptance"] for s in summary], marker="o")
    ax.set_xlabel("draft length (gamma)"); ax.set_ylabel("acceptance rate")
    ax.set_title("Draft length vs acceptance rate")
    fig.savefig(f"{args.figdir}/2_acceptance_vs_gamma.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    obs = [r["speedup"] for r in per_prompt if r["method"] != "baseline"]
    pr = [r["predicted_speedup"] for r in per_prompt if r["method"] != "baseline"]
    ax.scatter(pr, obs)
    lo = min(min(pr), min(obs)); hi = max(max(pr), max(obs))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("predicted speedup (Thm 3.8)"); ax.set_ylabel("observed speedup")
    ax.set_title("Predicted vs observed speedup (per prompt)")
    fig.savefig(f"{args.figdir}/3_pred_vs_obs.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    x = list(range(len(gs)))
    base_med_thr = statistics.median(r["throughput_med"] for r in per_prompt if r["method"] == "baseline")
    thr = []
    for g in gammas:
        thr.append(statistics.median(r["throughput_med"] for r in per_prompt if r["gamma"] == g))
    ax.bar(["baseline"] + [f"g{g}" for g in gammas], [base_med_thr] + thr)
    ax.set_ylabel("tokens/s"); ax.set_title("Throughput vs baseline (median over prompts)")
    fig.savefig(f"{args.figdir}/4_throughput.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    w = 0.35
    for i, cat in enumerate(cats):
        vals = [statistics.median(r["speedup"] for r in per_prompt if r["category"] == cat and r["gamma"] == g) for g in gammas]
        ax.bar([j + (i - len(cats)/2 + 0.5) * w for j in range(len(gammas))], vals, width=w, label=cat)
    ax.set_xticks(range(len(gammas))); ax.set_xticklabels([f"g{g}" for g in gammas])
    ax.axhline(1.0, color="k", ls="--", lw=1); ax.legend()
    ax.set_ylabel("median speedup"); ax.set_title("Speedup by prompt category")
    fig.savefig(f"{args.figdir}/5_by_category.png", dpi=150); plt.close(fig)

    print(json.dumps({"cost_c": c, "summary": summary}, indent=2))
    fails = [r for r in per_prompt if not r["exact_match_all"]]
    print(f"equivalence: {len(per_prompt)-len(fails)}/{len(per_prompt)} per-prompt configs exact-match")
    for r in fails:
        print("  MISMATCH", r["prompt_id"], r["method"])


if __name__ == "__main__":
    main()
