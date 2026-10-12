"""
Real-data Step 3: SumUp baseline on MGTAB (real bots, nothing injected).

Reuses the exact SumUp implementation from src/03_sumup_baseline.py
(imported, not copied), so the Facebook and MGTAB numbers come from the
same ticket-distribution + exact max-flow code.

Protocol (per frozen split, see 02_make_splits.py):
  - collector: TRAIN human closest to the median train-human degree
    (a trusted node; ties broken by lowest node id).
  - C_max = 1000 (same as the locked Facebook protocol) is the main
    result; 500 / 2000 / 5000 are a sensitivity check only.
  - voters: 200 honest users sampled from TEST humans (seeded) + ALL test
    bots, 1 vote each; honest and bot flow measured in separate max-flow
    runs, as in the original script.
  - attacker capacity per attack edge = bot votes collected / number of
    human-bot edges touching the VOTING bots. Also reported:
    bot_vote_acceptance_rate = bot votes collected / number of bot voters.
"""
import importlib.util
import json
import random
import statistics
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sumup", ROOT / "src" / "03_sumup_baseline.py")
sumup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sumup)

SEEDS = [0, 1, 2, 3, 4]
C_MAX_MAIN = 1000
C_MAX_SENSITIVITY = [500, 1000, 2000, 5000]
N_HONEST_VOTERS = 200


def pick_collector(G, train_humans):
    degrees = {v: G.degree(v) for v in train_humans}
    median_degree = sorted(degrees.values())[len(degrees) // 2]
    return min(sorted(train_humans), key=lambda v: abs(degrees[v] - median_degree))


def run_split(G, bot_set, split):
    seed = split["seed"]
    train_humans = [v for v in split["train"] if v not in bot_set]
    test_humans = [v for v in split["test"] if v not in bot_set]
    bot_voters = [v for v in split["test"] if v in bot_set]
    honest_voters = random.Random(seed).sample(test_humans, N_HONEST_VOTERS)
    n_attack_edges = sum(1 for b in bot_voters for u in G.neighbors(b) if u not in bot_set)

    collector = pick_collector(G, train_humans)
    levels = sumup.bfs_levels(G, collector)

    by_c_max = {}
    for c_max in C_MAX_SENSITIVITY:
        capacity = sumup.ticket_distribution(G, levels, collector, c_max)
        flow_graph = sumup.build_flow_graph(G, levels, capacity)
        attacker_flow, _ = sumup.votes_collected(flow_graph, collector, bot_voters)
        honest_flow, _ = sumup.votes_collected(flow_graph, collector, honest_voters)
        by_c_max[c_max] = {
            "attacker_flow_collected": attacker_flow,
            "attacker_capacity_per_attack_edge": round(attacker_flow / n_attack_edges, 4),
            "bot_vote_acceptance_rate": round(attacker_flow / len(bot_voters), 4),
            "honest_flow_collected": honest_flow,
            "honest_collection_rate": round(honest_flow / N_HONEST_VOTERS, 4),
        }
        print(f"  seed {seed}  C_max={c_max:>4}: attacker/attack-edge="
              f"{by_c_max[c_max]['attacker_capacity_per_attack_edge']:.4f}  "
              f"bot votes accepted={by_c_max[c_max]['bot_vote_acceptance_rate']:.3f}  "
              f"honest rate={by_c_max[c_max]['honest_collection_rate']:.3f}", flush=True)

    return {
        "seed": seed,
        "collector": collector,
        "collector_degree": G.degree(collector),
        "bfs_depth": max(levels.values()),
        "n_honest_voters": N_HONEST_VOTERS,
        "n_bot_voters": len(bot_voters),
        "n_attack_edges_of_voting_bots": n_attack_edges,
        "by_c_max": by_c_max,
    }


def mean_std(values):
    return {"mean": round(statistics.mean(values), 4), "std": round(statistics.stdev(values), 4)}


if __name__ == "__main__":
    G = nx.read_edgelist(ROOT / "data" / "real" / "mgtab_social.edgelist", nodetype=int)
    with open(ROOT / "data" / "real" / "mgtab_meta.json") as f:
        bot_set = set(json.load(f)["bot_nodes"])
    print(f"MGTAB social LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(bot_set)} bots")

    per_split = []
    for seed in SEEDS:
        with open(ROOT / "results" / "real" / "splits" / f"split_seed{seed}.json") as f:
            split = json.load(f)
        r = run_split(G, bot_set, split)
        print(f"seed {seed}: collector {r['collector']} (deg {r['collector_degree']}, depth {r['bfs_depth']}), "
              f"{r['n_bot_voters']} bot voters, {r['n_attack_edges_of_voting_bots']} attack edges", flush=True)
        per_split.append(r)

    summary = {}
    for c_max in C_MAX_SENSITIVITY:
        summary[c_max] = {
            metric: mean_std([r["by_c_max"][c_max][metric] for r in per_split])
            for metric in ("attacker_capacity_per_attack_edge", "bot_vote_acceptance_rate",
                           "honest_collection_rate")
        }
    results = {
        "dataset": "MGTAB social LCC (follower+friend)",
        "C_max_main": C_MAX_MAIN,
        "summary_over_seeds": summary,
        "per_split": per_split,
    }
    print("\nSummary (mean +/- std over 5 splits):")
    print(json.dumps(summary, indent=2))

    out = ROOT / "results" / "real" / "sumup_mgtab.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")
