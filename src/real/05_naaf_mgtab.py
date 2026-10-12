"""
Real-data Step 6b: NAAF variants vs SumUp vs classifier-only on MGTAB.

Same SumUp code, collector rule, C_max and voter protocol as
03_sumup_mgtab.py; the only thing that changes between methods is how the
classifier's P(bot) is fused in. Every method is scored on the SAME split,
collector and voter sample, so differences are attributable to the fusion
rule alone.

  --eval val   (default) voters = 200 VAL humans + all VAL bots. All
               design choices are made here.
  --eval test  voters = 200 TEST humans + all TEST bots. Run ONCE for the
               method chosen on val -- never used for tuning.

Threshold tau = quantile q of P(bot) among ALL val humans (no bot labels
used to set it), so ~(1-q) of honest users fall above it. Calibrating on val
then reporting val makes the val honest rate optimistic by construction;
the test run is the out-of-sample check.

Methods (P = P(bot), voters each have 1 vote unless stated):
  sumup            plain SumUp ticket capacities
  classifier_only  no flow at all: a vote counts iff P < tau
  naaf_a           voter capacity 1 - P          (Option A, Facebook Step 9)
  naaf_gated_hard  voter capacity 0 if P >= tau else 1
  naaf_gated_soft  voter capacity 1 - P if P >= tau else 1
  naaf_c           voter capacity 1 - gate(P), linear ramp tau_lo(q=0.90) -> tau  (Option C)
  naaf_v1_edge     every edge capacity * (1 - P(child))                  (Facebook Step 6 v1)
  naaf_b_edge      edge capacity * (1 - P(child)) only if P(child) >= tau (Option B)

Win rule (agreed with the author): attacker capacity per attack edge below
SumUp's AND honest collection rate at most 1 percentage point below SumUp's.
"""
import argparse
import importlib.util
import json
import random
import statistics
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sumup", ROOT / "src" / "03_sumup_baseline.py")
sumup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sumup)
_spec = importlib.util.spec_from_file_location("sumup_mgtab", ROOT / "src" / "real" / "03_sumup_mgtab.py")
sumup_mgtab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sumup_mgtab)

SEEDS = [0, 1, 2, 3, 4]
C_MAX = 1000
N_HONEST_VOTERS = 200
Q_LO = 0.90
WIN_HONEST_TOLERANCE = 0.01


def votes_collected_weighted(flow_graph, collector, voter_weights):
    """Same as votes_collected_weighted in src/09_naaf_option_b.py: each
    voter gets its own SOURCE->voter capacity (weight in [0, 1])."""
    F = flow_graph.copy()
    F.add_node("SOURCE")
    for v, w in voter_weights.items():
        if w > 0:
            F.add_edge("SOURCE", v, capacity=w)
    value, _ = nx.maximum_flow(F, "SOURCE", collector)
    return value


def ramp(p, tau_lo, tau_hi):
    """Option C gate from src/09_naaf_option_b.py: 0 below tau_lo, 1 above tau_hi."""
    if tau_hi <= tau_lo:
        return 1.0 if p > tau_lo else 0.0
    return max(0.0, min(1.0, (p - tau_lo) / (tau_hi - tau_lo)))


def score(attacker, honest, n_attack_edges, n_bot_voters):
    return {
        "attacker_flow_collected": round(attacker, 3),
        "attacker_capacity_per_attack_edge": round(attacker / n_attack_edges, 4),
        "bot_vote_acceptance_rate": round(attacker / n_bot_voters, 4),
        "honest_collection_rate": round(honest / N_HONEST_VOTERS, 4),
    }


def run_split(G, bot_set, split, p_bot, quantiles, eval_part):
    seed = split["seed"]
    val_humans = [v for v in split["val"] if v not in bot_set]
    part_humans = [v for v in split[eval_part] if v not in bot_set]
    bot_voters = [v for v in split[eval_part] if v in bot_set]
    honest_voters = random.Random(seed).sample(part_humans, N_HONEST_VOTERS)
    n_attack_edges = sum(1 for b in bot_voters for u in G.neighbors(b) if u not in bot_set)

    collector = sumup_mgtab.pick_collector(G, [v for v in split["train"] if v not in bot_set])
    levels = sumup.bfs_levels(G, collector)
    cap_sumup = sumup.ticket_distribution(G, levels, collector, C_MAX)
    flow_sumup = sumup.build_flow_graph(G, levels, cap_sumup)

    def by_voter_weight(weight):
        return score(votes_collected_weighted(flow_sumup, collector, {v: weight(v) for v in bot_voters}),
                     votes_collected_weighted(flow_sumup, collector, {v: weight(v) for v in honest_voters}),
                     n_attack_edges, len(bot_voters))

    def by_edge_capacity(cap):
        flow = sumup.build_flow_graph(G, levels, cap)
        return score(sumup.votes_collected(flow, collector, bot_voters)[0],
                     sumup.votes_collected(flow, collector, honest_voters)[0],
                     n_attack_edges, len(bot_voters))

    out = {"seed": seed, "collector": collector, "n_bot_voters": len(bot_voters),
           "n_attack_edges_of_voting_bots": n_attack_edges, "methods": {}}
    m = out["methods"]
    m["sumup"] = by_voter_weight(lambda v: 1.0)
    m["naaf_a"] = by_voter_weight(lambda v: 1.0 - p_bot[v])
    m["naaf_v1_edge"] = by_edge_capacity({(u, v): c * (1 - p_bot[v]) for (u, v), c in cap_sumup.items()})

    tau_lo = float(np.quantile([p_bot[v] for v in val_humans], Q_LO))
    out["tau"] = {}
    for q in quantiles:
        tau = float(np.quantile([p_bot[v] for v in val_humans], q))
        out["tau"][str(q)] = round(tau, 4)
        s = f"@q{q}"
        m["classifier_only" + s] = score(sum(p_bot[v] < tau for v in bot_voters),
                                         sum(p_bot[v] < tau for v in honest_voters),
                                         n_attack_edges, len(bot_voters))
        m["naaf_gated_hard" + s] = by_voter_weight(lambda v: 0.0 if p_bot[v] >= tau else 1.0)
        m["naaf_gated_soft" + s] = by_voter_weight(lambda v: 1.0 - p_bot[v] if p_bot[v] >= tau else 1.0)
        m["naaf_c" + s] = by_voter_weight(lambda v: 1.0 - ramp(p_bot[v], tau_lo, tau))
        m["naaf_b_edge" + s] = by_edge_capacity(
            {(u, v): (c * (1 - p_bot[v]) if p_bot[v] >= tau else c) for (u, v), c in cap_sumup.items()})

    base = m["sumup"]
    for name, r in m.items():
        r["wins_vs_sumup"] = bool(
            r["attacker_capacity_per_attack_edge"] < base["attacker_capacity_per_attack_edge"]
            and r["honest_collection_rate"] >= base["honest_collection_rate"] - WIN_HONEST_TOLERANCE)
    return out


def summarise(per_split):
    summary = {}
    for name in per_split[0]["methods"]:
        rows = [s["methods"][name] for s in per_split]
        entry = {k: {"mean": round(statistics.mean(r[k] for r in rows), 4),
                     "std": round(statistics.stdev(r[k] for r in rows), 4)}
                 for k in ("attacker_capacity_per_attack_edge", "bot_vote_acceptance_rate",
                           "honest_collection_rate")} if len(rows) > 1 else \
            {k: {"mean": rows[0][k], "std": 0.0} for k in ("attacker_capacity_per_attack_edge",
                                                          "bot_vote_acceptance_rate", "honest_collection_rate")}
        entry["splits_won"] = sum(r["wins_vs_sumup"] for r in rows)
        summary[name] = entry
    base = summary["sumup"]
    for entry in summary.values():
        entry["wins_on_mean"] = bool(
            entry["attacker_capacity_per_attack_edge"]["mean"] < base["attacker_capacity_per_attack_edge"]["mean"]
            and entry["honest_collection_rate"]["mean"]
            >= base["honest_collection_rate"]["mean"] - WIN_HONEST_TOLERANCE)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifiers", nargs="+", required=True,
                        help="probability file stems in results/real/classifier, e.g. mlp_e600")
    parser.add_argument("--eval", choices=["val", "test"], default="val")
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.99, 0.995])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out-suffix", default="", help="e.g. _smoketest, to avoid overwriting real results")
    args = parser.parse_args()

    G = nx.read_edgelist(ROOT / "data" / "real" / "mgtab_social.edgelist", nodetype=int)
    meta = json.load(open(ROOT / "data" / "real" / "mgtab_meta.json"))
    bot_set = set(meta["bot_nodes"])
    nodes = sorted(meta["bot_nodes"] + meta["human_nodes"])  # row order of the p_bot_*.npy files

    out_dir = ROOT / "results" / "real" / "naaf"
    out_dir.mkdir(parents=True, exist_ok=True)
    for clf in args.classifiers:
        per_split = []
        for seed in args.seeds:
            split = json.load(open(ROOT / "results" / "real" / "splits" / f"split_seed{seed}.json"))
            probs = np.load(ROOT / "results" / "real" / "classifier" / f"p_bot_{clf}_seed{seed}.npy")
            p_bot = dict(zip(nodes, probs.tolist()))
            r = run_split(G, bot_set, split, p_bot, args.quantiles, args.eval)
            per_split.append(r)
            print(f"[{clf}] seed {seed} done (tau {r['tau']})", flush=True)
        summary = summarise(per_split)
        print(f"\n[{clf}] {args.eval} summary (mean over 5 splits):")
        print(f"  {'method':<24}{'attacker/edge':>14}{'bot votes acc.':>16}{'honest rate':>13}{'win(mean)':>11}{'splits won':>12}")
        for name, e in summary.items():
            print(f"  {name:<24}{e['attacker_capacity_per_attack_edge']['mean']:>14.4f}"
                  f"{e['bot_vote_acceptance_rate']['mean']:>16.3f}{e['honest_collection_rate']['mean']:>13.3f}"
                  f"{str(e['wins_on_mean']):>11}{e['splits_won']:>10}/{len(per_split)}", flush=True)
        json.dump({"classifier": clf, "eval": args.eval, "C_max": C_MAX, "quantiles": args.quantiles,
                   "win_rule": "attacker/edge < SumUp and honest rate >= SumUp - 0.01",
                   "summary_over_seeds": summary, "per_split": per_split},
                  open(out_dir / f"naaf_{clf}_{args.eval}{args.out_suffix}.json", "w"), indent=2)
        print(f"Saved {out_dir / f'naaf_{clf}_{args.eval}{args.out_suffix}.json'}")
