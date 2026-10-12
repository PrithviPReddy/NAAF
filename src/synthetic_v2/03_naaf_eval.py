"""
Synthetic v2, Step 3: SumUp vs classifier-only vs NAAF variants on the
fresh Facebook injections -- identical protocol and code to MGTAB
(run_split / summarise imported from src/real/05_naaf_mgtab.py): collector
= median-degree TRAIN human, C_max = 1000, voters = 200 sampled humans +
all Sybils of the evaluated part, tau = quantile of P(bot) over val humans,
same win rule. Each seed is its own injected graph.

Usage: python 03_naaf_eval.py --scenario dense --classifier gatv2_social_skip --eval val
"""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import importlib.util

_spec = importlib.util.spec_from_file_location("naaf", ROOT / "src" / "real" / "05_naaf_mgtab.py")
naaf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naaf)

DATA = ROOT / "data" / "synthetic_v2"
SEEDS = [0, 1, 2, 3, 4]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["dense", "sparse"], required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--eval", choices=["val", "test"], default="val")
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])
    args = parser.parse_args()

    per_split = []
    for seed in SEEDS:
        d = DATA / f"{args.scenario}_seed{seed}"
        meta, split = json.load(open(d / "meta.json")), json.load(open(d / "split.json"))
        G = nx.read_edgelist(d / "graph.edgelist", nodetype=int)
        probs = np.load(ROOT / "results" / "synthetic_v2" / "classifier" /
                        f"p_bot_{args.scenario}_{args.classifier}_seed{seed}.npy")
        p_bot = dict(zip(sorted(G.nodes()), probs.tolist()))
        r = naaf.run_split(G, set(meta["bot_nodes"]), split, p_bot, args.quantiles, args.eval)
        per_split.append(r)
        print(f"[{args.scenario}/{args.classifier}] seed {seed} done: {r['n_bot_voters']} Sybil voters, "
              f"{r['n_attack_edges_of_voting_bots']} attack edges", flush=True)

    summary = naaf.summarise(per_split)
    print(f"\n[{args.scenario}/{args.classifier}] {args.eval} summary (mean over 5 injected runs):")
    print(f"  {'method':<24}{'attacker/edge':>14}{'bot votes acc.':>16}{'honest rate':>13}{'win(mean)':>11}{'runs won':>10}")
    for name, e in summary.items():
        print(f"  {name:<24}{e['attacker_capacity_per_attack_edge']['mean']:>14.4f}"
              f"{e['bot_vote_acceptance_rate']['mean']:>16.3f}{e['honest_collection_rate']['mean']:>13.3f}"
              f"{str(e['wins_on_mean']):>11}{e['splits_won']:>8}/5", flush=True)

    out_dir = ROOT / "results" / "synthetic_v2" / "naaf"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"naaf_{args.scenario}_{args.classifier}_{args.eval}"
    json.dump({"scenario": args.scenario, "classifier": f"{args.scenario}_{args.classifier}",
               "eval": args.eval, "C_max": naaf.C_MAX, "quantiles": args.quantiles,
               "win_rule": "attacker/edge < SumUp and honest rate >= SumUp - 0.01",
               "summary_over_seeds": summary, "per_split": per_split},
              open(out_dir / f"{stem}.json", "w"), indent=2)
    print(f"Saved {out_dir / f'{stem}.json'}")
