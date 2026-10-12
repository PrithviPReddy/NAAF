"""
Real-data Step 6b figure: bot-vote acceptance vs honest collection rate
(mean over 5 splits) for every method, sweeping the threshold quantile q.

Lower-right is better. The shaded band is the agreed win region: honest
rate within 1 percentage point of SumUp's. Reading the figure: a NAAF curve
that sits BELOW the classifier-only curve at the same honest rate means the
SumUp flow adds value on top of the classifier.

Usage: python 06_plot_tradeoff.py --eval val --classifiers gatv2_social_skip_e600 mlp_e600
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# Categorical slots 1-6 of the validated default palette (light mode); the
# low-contrast slots are backed by distinct markers + legend + JSON tables.
SWEPT = [  # (method prefix, label, colour, marker)
    ("classifier_only", "Classifier only (no flow)", "#2a78d6", "o"),
    ("naaf_gated_hard", "NAAF gated (hard)", "#eb6834", "s"),
    ("naaf_b_edge", "NAAF-B edge throttle", "#1baf7a", "^"),
    ("naaf_c", "NAAF-C ramp", "#eda100", "D"),
]
SINGLE = [
    ("naaf_a", "NAAF-A (1 - P)", "#e87ba4", "P"),
    ("naaf_v1_edge", "NAAF-v1 edge (1 - P)", "#008300", "X"),
]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"


def panel(ax, summary, quantiles, title):
    def xy(name):
        e = summary[name]
        return 100 * e["honest_collection_rate"]["mean"], 100 * e["bot_vote_acceptance_rate"]["mean"]

    sx, sy = xy("sumup")
    ax.axvspan(sx - 1, 100.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(sx - 0.9, 3, "win region\n(honest ≥ SumUp − 1 pt)", color=MUTED, fontsize=8, va="bottom")
    for prefix, label, colour, marker in SWEPT:
        pts = sorted(xy(f"{prefix}@q{q}") for q in quantiles)
        ax.plot(*zip(*pts), color=colour, lw=2, marker=marker, ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
    for name, label, colour, marker in SINGLE:
        ax.plot(*xy(name), ls="none", color=colour, marker=marker, ms=9,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=4)
    ax.plot(sx, sy, ls="none", color=INK, marker="*", ms=14, markeredgecolor=SURFACE, label="SumUp", zorder=5)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Honest votes collected (%)", color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", choices=["val", "test"], default="val")
    parser.add_argument("--classifiers", nargs="+", required=True)
    args = parser.parse_args()

    results = [json.load(open(ROOT / "results" / "real" / "naaf" / f"naaf_{c}_{args.eval}.json"))
               for c in args.classifiers]
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=True, facecolor=SURFACE)
    axes = [axes] if len(results) == 1 else list(axes)
    for ax, r in zip(axes, results):
        ax.set_facecolor(SURFACE)
        panel(ax, r["summary_over_seeds"], r["quantiles"],
              f"{r['classifier']} — MGTAB {args.eval}, mean of 5 splits")
    axes[0].set_ylabel("Bot votes accepted (%)", color=INK)
    axes[0].set_ylim(0, 102)
    axes[-1].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()

    out_dir = ROOT / "figures" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"tradeoff_{args.eval}.{ext}", dpi=200, facecolor=SURFACE)
    print(f"Saved {out_dir / f'tradeoff_{args.eval}.png'} and .svg")
