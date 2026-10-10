"""README visuals: hero montage (2x2 of key results) + algorithm schematic.

Usage: .venv/bin/python scripts/make_readme_figures.py results/raw/<run>.json
Reads results/processed/*.csv + figures/*.png. Writes figures/README_hero.png
and figures/README_how_it_works.png.
"""

import csv
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RUN = sys.argv[1] if len(sys.argv) > 1 else None
FIG = "figures"
PROC = "results/processed"


def read_csv(name):
    with open(f"{PROC}/{name}") as f:
        return list(csv.DictReader(f))


def hero():
    summ = sorted(read_csv("summary_by_gamma.csv"), key=lambda r: int(r["gamma"]))
    per = read_csv("per_prompt.csv")
    cats = read_csv("summary_by_category.csv")
    gs = [int(s["gamma"]) for s in summ]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # A. speedup vs gamma
    a = ax[0, 0]
    a.errorbar(gs,
               [float(s["median_speedup"]) for s in summ],
               yerr=[[float(s["median_speedup"]) - float(s["ci_lo"]) for s in summ],
                     [float(s["ci_hi"]) - float(s["median_speedup"]) for s in summ]],
               marker="o", capsize=4)
    a.axhline(1.0, color="k", ls="--", lw=1)
    a.set_xlabel("draft length (gamma)")
    a.set_ylabel("median speedup vs baseline")
    a.set_title("A. Speedup peaks at gamma=1-2, then declines")
    # B. acceptance vs gamma
    b = ax[0, 1]
    b.plot(gs, [float(s["median_acceptance"]) for s in summ], marker="o")
    b.set_xlabel("draft length (gamma)")
    b.set_ylabel("acceptance rate")
    b.set_title("B. Acceptance falls as drafts get longer")
    # C. predicted vs observed
    c = ax[1, 0]
    obs = [float(r["speedup"]) for r in per if r["method"] != "baseline"]
    pr = [float(r["predicted_speedup"]) for r in per if r["method"] != "baseline"]
    c.scatter(pr, obs, s=18)
    lo, hi = min(min(pr), min(obs)), max(max(pr), max(obs))
    c.plot([lo, hi], [lo, hi], "k--", lw=1)
    c.set_xlabel("predicted speedup (paper Thm 3.8)")
    c.set_ylabel("observed speedup")
    c.set_title("C. Theory is conservative (points above diagonal)")
    # D. by category
    d = ax[1, 1]
    catnames = sorted({r["category"] for r in cats})
    w = 0.35
    for i, cat in enumerate(catnames):
        vals = [float(next(r["median_speedup"] for r in cats
                           if r["category"] == cat and int(r["gamma"]) == g)) for g in gs]
        d.bar([j + (i - len(catnames) / 2 + 0.5) * w for j in range(len(gs))],
              vals, width=w, label=cat)
    d.set_xticks(range(len(gs)))
    d.set_xticklabels([f"g{g}" for g in gs])
    d.axhline(1.0, color="k", ls="--", lw=1)
    d.legend(fontsize=8)
    d.set_ylabel("median speedup")
    d.set_title("D. Code prompts gain more than prose")
    fig.suptitle("Speculative decoding on CPU (gpt2 target / distilgpt2 draft, greedy, n=8 prompts)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG}/README_hero.png", dpi=150)
    plt.close(fig)


def schematic():
    fig, a = plt.subplots(figsize=(10, 3.2))
    a.set_xlim(0, 10)
    a.set_ylim(0, 3)
    a.axis("off")
    boxes = [
        (0.3, 1.0, 1.9, 1.0, "Draft model\n(distilgpt2)", "#cfe2ff"),
        (3.0, 1.0, 2.2, 1.0, "proposes gamma\ntokens", "#e2e3e5"),
        (6.0, 1.0, 1.9, 1.0, "Target model\n(gpt2) verifies", "#d1e7dd"),
        (8.7, 1.0, 1.0, 1.0, "accept +\n1 bonus", "#fff3cd"),
    ]
    for x, y, wdt, hgt, txt, col in boxes:
        a.add_patch(FancyBboxPatch((x, y), wdt, hgt, boxstyle="round,pad=0.05",
                                   fc=col, ec="black"))
        a.text(x + wdt / 2, y + hgt / 2, txt, ha="center", va="center", fontsize=9)
    for x0, x1 in ((2.2, 3.0), (5.2, 6.0), (7.9, 8.7)):
        a.add_patch(FancyArrowPatch((x0, 1.5), (x1, 1.5), arrowstyle="-|>",
                                    mutation_scale=12, lw=1.2))
    a.text(5, 0.45, "One target forward scores all gamma drafts in parallel; "
           "rejected draft replaced via residual distribution (paper Alg. 1).",
           ha="center", fontsize=8.5, style="italic")
    a.set_title("How speculative decoding works (one iteration)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/README_how_it_works.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    hero()
    schematic()
    print("wrote figures/README_hero.png figures/README_how_it_works.png")
