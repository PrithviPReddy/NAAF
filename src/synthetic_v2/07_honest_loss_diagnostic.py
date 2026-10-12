"""
Synthetic v2, Step 7 (diagnostic, not a new result): why does SumUp's honest
collection on the Facebook graph fall with fewer attack edges
(about 51% wide cut, 27% narrow cut, 20% vote stuffing)?

Hypothesis: the injected Sybil region acts as a shortcut. Its attack edges
join random honest users across Facebook's community structure, so honest
votes can route through it and BFS levels are shallower. Test: for every
injected run, keep the SAME collector and the SAME 200 test honest voters and
recompute SumUp's honest collection (C_max = 1000) on
  (a) the full injected graph (matches the reported numbers), and
  (b) the honest-only graph (Sybil nodes and their edges removed).
Also reports the BFS depth and the share of honest voters within 2 hops of
the collector, with and without Sybils.
"""
import importlib.util
import json
import random
import statistics
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("naaf", ROOT / "src" / "real" / "05_naaf_mgtab.py")
naaf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naaf)

DATA = ROOT / "data" / "synthetic_v2"
SCENARIOS = ["dense", "sparse", "stuffing"]
SEEDS = [0, 1, 2, 3, 4]
C_MAX = 1000


def honest_rate(G, collector, voters):
    levels = naaf.sumup.bfs_levels(G, collector)
    flow = naaf.sumup.build_flow_graph(G, levels, naaf.sumup.ticket_distribution(G, levels, collector, C_MAX))
    collected, _ = naaf.sumup.votes_collected(flow, collector, voters)
    near = sum(1 for v in voters if levels.get(v, 99) <= 2) / len(voters)
    return collected / len(voters), max(levels.values()), near


if __name__ == "__main__":
    out = {}
    for scenario in SCENARIOS:
        rows = []
        for seed in SEEDS:
            d = DATA / f"{scenario}_seed{seed}"
            meta, split = json.load(open(d / "meta.json")), json.load(open(d / "split.json"))
            G = nx.read_edgelist(d / "graph.edgelist", nodetype=int)
            bots = set(meta["bot_nodes"])
            voters = random.Random(seed).sample([v for v in split["test"] if v not in bots], naaf.N_HONEST_VOTERS)
            collector = naaf.sumup_mgtab.pick_collector(G, [v for v in split["train"] if v not in bots])
            full = honest_rate(G, collector, voters)
            honest_only = honest_rate(G.subgraph(v for v in G if v not in bots).copy(), collector, voters)
            rows.append({"seed": seed, "collector": collector, "collector_degree": G.degree(collector),
                         "attack_edges": meta["n_attack_edges"],
                         "with_sybils": dict(zip(("honest_rate", "bfs_depth", "voters_within_2_hops"), full)),
                         "honest_only": dict(zip(("honest_rate", "bfs_depth", "voters_within_2_hops"), honest_only))})
            print(f"{scenario:>8} seed {seed}: collector {collector} (deg {G.degree(collector)}) | "
                  f"with Sybils: honest {full[0]:.3f}, depth {full[1]}, within 2 hops {full[2]:.2f} | "
                  f"honest-only: honest {honest_only[0]:.3f}, depth {honest_only[1]}, within 2 hops {honest_only[2]:.2f}",
                  flush=True)
        summary = {k: {m: round(statistics.mean(r[k][m] for r in rows), 3) for m in rows[0][k]}
                   for k in ("with_sybils", "honest_only")}
        out[scenario] = {"summary_mean_over_seeds": summary, "per_seed": rows}
        print(f"  {scenario} mean: {json.dumps(summary)}\n", flush=True)

    path = ROOT / "results" / "synthetic_v2" / "honest_loss_diagnostic.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved {path}")
