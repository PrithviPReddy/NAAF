"""
Synthetic v2 figure: Sybil votes accepted vs classifier-evasion rate in the
vote-stuffing scenario (500 Sybils behind 50 attack edges), one panel per
C_max, mean +/- std over 5 injections. NAAF-A is omitted: it overlaps NAAF
gated within 0.3 points at every f (see stuffing_*.json).

Usage: python 06_plot_evasion.py --eval test
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
SERIES = [  # (method, label, colour, marker, linestyle) -- palette slots 1-2 + neutral ink
    ("sumup", "SumUp (flow only)", INK, "*", "--"),
    ("classifier_only", "Classifier only", "#2a78d6", "o", "-"),
    ("naaf_gated_hard", "NAAF (gated)", "#eb6834", "s", "-"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", choices=["val", "test"], default="test")
    args = parser.parse_args()
    r = json.load(open(ROOT / "results" / "synthetic_v2" / "stuffing" / f"stuffing_{args.eval}.json"))

    fig, axes = plt.subplots(1, len(r["C_max"]), figsize=(5.5 * len(r["C_max"]), 4.4), sharey=True, facecolor=SURFACE)
    for ax, c_max in zip(axes, map(str, r["C_max"])):
        ax.set_facecolor(SURFACE)
        fs = [float(f) for f in r["summary"][c_max]]
        for method, label, colour, marker, ls in SERIES:
            mean = [100 * r["summary"][c_max][str(f)][method]["bot_vote_acceptance_rate"]["mean"] for f in fs]
            std = [100 * r["summary"][c_max][str(f)][method]["bot_vote_acceptance_rate"]["std"] for f in fs]
            x = [100 * f for f in fs]
            ax.fill_between(x, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                            color=colour, alpha=0.06 if method == "sumup" else 0.12, lw=0)
            ax.plot(x, mean, color=colour, lw=2, ls=ls, marker=marker, ms=7 if marker != "*" else 11,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
            honest = 100 * r["summary"][c_max]["0.0"][method]["honest_collection_rate"]["mean"]
            ax.annotate(f"honest {honest:.0f}%", (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=MUTED)
        ax.set_title(f"C_max = {c_max}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Sybil voters that evade the classifier (%)", color=INK)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlim(-4, 128)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("Sybil votes accepted (%)", color=INK)
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"Vote stuffing: 500 Sybils behind 50 attack edges — SNAP Facebook, {args.eval} split, mean ± std of 5 injections",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()

    out_dir = ROOT / "figures" / "synthetic_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"evasion_{args.eval}.{ext}", dpi=200, facecolor=SURFACE)
    print(f"Saved {out_dir / f'evasion_{args.eval}.png'} and .svg")
