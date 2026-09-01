"""
Sparse-attack scenario: SumUp baseline, rerun on the sparse Sybil
injection (25 Sybils, 200 attack edges) instead of the dense one.

Reuses the EXACT ticket-distribution / max-flow machinery from
03_sumup_baseline.py (bfs_levels, ticket_distribution, build_flow_graph,
votes_collected) unmodified -- only the input graph/labels and output
path differ. Same locked protocol as the dense run: median-degree
collector, C_max=1000, 200 honest voters (seed 42) vs. all Sybil
voters (now 25, since that's the whole point of the sparse scenario),
1 vote cap each.

Does not read or write anything under the dense scenario's file names.
"""
import networkx as nx
import json
import random
from importlib import import_module

sumup = import_module("03_sumup_baseline")

if __name__ == "__main__":
    G = nx.read_edgelist("../data/combined_graph_sparse.edgelist", nodetype=int,
                          data=(("edge_type", str),))
    with open("../data/sybil_labels_sparse.json") as f:
        labels = json.load(f)
    sybil_nodes = set(labels["sybil_nodes"])
    n_attack_edges = len(labels["attack_edges"])
    honest_nodes = [n for n in G.nodes() if n not in sybil_nodes]

    degrees = dict(G.degree(honest_nodes))
    median_degree = sorted(degrees.values())[len(degrees) // 2]
    collector = min(honest_nodes, key=lambda n: abs(degrees[n] - median_degree))
    print(f"Collector node: {collector} (degree {G.degree(collector)}, "
          f"median honest degree is {median_degree})")

    levels = sumup.bfs_levels(G, collector)
    max_level = max(levels.values())
    level_sizes = {}
    for lvl in levels.values():
        level_sizes[lvl] = level_sizes.get(lvl, 0) + 1
    print(f"BFS depth from collector: {max_level} levels, sizes: {level_sizes}")

    C_MAX = 1000
    capacity = sumup.ticket_distribution(G, levels, collector, C_MAX)
    print(f"Ticket distribution assigned capacities to {len(capacity)} forward edges "
          f"(C_max={C_MAX})")

    flow_graph = sumup.build_flow_graph(G, levels, capacity)

    attacker_flow, n_sybil_reachable = sumup.votes_collected(flow_graph, collector, sybil_nodes)

    rng = random.Random(42)
    honest_voter_sample = rng.sample(honest_nodes, 200)  # same sample size as dense run
    honest_flow, n_honest_reachable = sumup.votes_collected(flow_graph, collector, honest_voter_sample)

    honest_flow_full, n_honest_full = sumup.votes_collected(flow_graph, collector, honest_nodes)

    results = {
        "scenario": "sparse",
        "n_sybil_nodes": len(sybil_nodes),
        "collector": collector,
        "C_max": C_MAX,
        "n_attack_edges": n_attack_edges,
        "n_sybil_voters": n_sybil_reachable,
        "attacker_flow_collected": attacker_flow,
        "attacker_capacity_per_attack_edge": round(attacker_flow / n_attack_edges, 4),
        "n_honest_voters_sampled": n_honest_reachable,
        "honest_flow_collected": honest_flow,
        "honest_collection_rate": round(honest_flow / n_honest_reachable, 4),
        "honest_collection_rate_full_turnout_4039_voters": round(honest_flow_full / n_honest_full, 4),
    }
    print(json.dumps(results, indent=2))

    with open("../results/sumup_sparse.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved ../results/sumup_sparse.json")
