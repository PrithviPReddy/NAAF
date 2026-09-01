"""
Additional controlled experiment (user-requested, post-review): compare
SumUp baseline vs. NAAF-raw (1 - P_sybil) vs. NAAF-percentile (rank-based
scaling) on BOTH the dense and sparse scenarios, side by side, using each
scenario's own already-locked collector/C_max/voter-sample/seed exactly.

This script is purely additive: it reads the existing locked graphs,
labels, and trained GATv2 models (dense: gatv2_model.pt, sparse:
gatv2_model_sparse.pt) and writes ONLY new files
(results/percentile_experiment_dense.json,
results/percentile_experiment_sparse.json). It does not import-and-rerun
anything that writes to naaf_fusion*.json or ablation_study*.json, and
never opens any locked file in write mode.

--- Percentile-scaling formula ---

Given the GATv2 model's P(Sybil) output for every node v in the graph,
p(v) in [0, 1], let N be the total node count. Sort all N nodes ascending
by p(v) and let rank(v) in {0, 1, ..., N-1} be v's 0-indexed position in
that sorted order (ties broken by node id, stable sort -- with continuous
sigmoid outputs from a trained network, exact ties are essentially never
observed in practice).

    percentile(v) = rank(v) / (N - 1)                     in [0, 1]
    multiplier(v) = 1 - percentile(v)                     in [0, 1]
    capacity_NAAF_percentile(u -> v) = capacity_SumUp(u -> v) * multiplier(v)

The node with the single lowest P(Sybil) gets percentile=0 -> multiplier=1
(full SumUp capacity, untouched). The node with the single highest
P(Sybil) gets percentile=1 -> multiplier=0 (fully throttled). Every other
node's multiplier is its relative standing among all nodes, not its raw
probability -- so the multiplier is always spread across the full [0, 1]
range by construction, regardless of how small or clustered the
classifier's raw probabilities are in absolute terms.

Worked numeric example (N=5 toy nodes, P(Sybil) = [0.01, 0.03, 0.02, 0.20, 0.05]):

  node   raw P(Sybil)   rank (ascending)   percentile = rank/(N-1)   multiplier = 1-percentile   raw-scaling multiplier = 1-P
  A      0.01           0                  0/4 = 0.00                1.00                        0.99
  C      0.02           1                  1/4 = 0.25                0.75                        0.98
  B      0.03           2                  2/4 = 0.50                0.50                        0.97
  E      0.05           3                  3/4 = 0.75                0.25                        0.95
  D      0.20           4                  4/4 = 1.00                0.00                        0.80

Note the last column: with all raw probabilities small, raw scaling barely
differentiates the 5 nodes (multipliers bunched in [0.80, 0.99]) even
though the model's *ranking* of them is informative. Percentile scaling
uses that same ranking but spreads it across the full [0, 1] range -- this
is exactly the mechanism intended to fix the sparse-scenario calibration
problem found in 06_naaf_fusion_sparse.py / 07_ablation_study_sparse.py.
Whether it helps, hurts, or is neutral on the DENSE scenario (where raw
scaling already worked) is an empirical question this script answers, not
assumed.
"""
import networkx as nx
import json
import random
from importlib import import_module

sumup = import_module("03_sumup_baseline")
fusion_dense = import_module("06_naaf_fusion")
fusion_sparse = import_module("06_naaf_fusion_sparse")


def percentile_multiplier(p_sybil):
    """dict{node: raw P(Sybil)} -> dict{node: percentile-rank-based multiplier in [0,1]}."""
    ranked = sorted(p_sybil.items(), key=lambda kv: kv[1])  # ascending by P(Sybil)
    n = len(ranked)
    return {node: 1.0 - (rank / (n - 1)) for rank, (node, _) in enumerate(ranked)}


def run_scenario(scenario_name, graph_path, labels_path, get_p_sybil_fn):
    G = nx.read_edgelist(graph_path, nodetype=int, data=(("edge_type", str),))
    with open(labels_path) as f:
        labels = json.load(f)
    sybil_nodes = set(labels["sybil_nodes"])
    n_attack_edges = len(labels["attack_edges"])
    honest_nodes = [n for n in G.nodes() if n not in sybil_nodes]

    p_sybil = get_p_sybil_fn(G, sybil_nodes)
    mult_percentile = percentile_multiplier(p_sybil)

    # --- identical collector/C_max derivation as every prior script for this scenario ---
    degrees = dict(G.degree(honest_nodes))
    median_degree = sorted(degrees.values())[len(degrees) // 2]
    collector = min(honest_nodes, key=lambda n: abs(degrees[n] - median_degree))
    levels = sumup.bfs_levels(G, collector)
    C_MAX = 1000
    capacity_sumup = sumup.ticket_distribution(G, levels, collector, C_MAX)

    capacity_naaf_raw = {(u, v): cap * (1 - p_sybil[v]) for (u, v), cap in capacity_sumup.items()}
    capacity_naaf_pct = {(u, v): cap * mult_percentile[v] for (u, v), cap in capacity_sumup.items()}

    rng = random.Random(42)
    honest_voter_sample = rng.sample(honest_nodes, 200)  # identical seed/size as every prior script

    def evaluate(capacity_dict):
        flow_graph = sumup.build_flow_graph(G, levels, capacity_dict)
        attacker_flow, _ = sumup.votes_collected(flow_graph, collector, sybil_nodes)
        honest_flow, n_h = sumup.votes_collected(flow_graph, collector, honest_voter_sample)
        return {
            "attacker_flow_collected": round(attacker_flow, 3),
            "attacker_capacity_per_attack_edge": round(attacker_flow / n_attack_edges, 4),
            "honest_flow_collected": round(honest_flow, 3),
            "honest_collection_rate": round(honest_flow / n_h, 4),
        }

    result_sumup = evaluate(capacity_sumup)
    result_naaf_raw = evaluate(capacity_naaf_raw)
    result_naaf_pct = evaluate(capacity_naaf_pct)

    def compare(base, variant):
        return {
            "attacker_capacity_reduction_pct_vs_sumup": round(
                100 * (1 - variant["attacker_capacity_per_attack_edge"] /
                       max(base["attacker_capacity_per_attack_edge"], 1e-9)), 2),
            "honest_collection_change_pp_vs_sumup": round(
                100 * (variant["honest_collection_rate"] - base["honest_collection_rate"]), 2),
            "honest_collection_change_pct_vs_sumup": round(
                100 * (variant["honest_collection_rate"] - base["honest_collection_rate"]) /
                max(base["honest_collection_rate"], 1e-9), 2),
        }

    results = {
        "scenario": scenario_name,
        "n_total_nodes": G.number_of_nodes(),
        "n_sybil_nodes": len(sybil_nodes),
        "collector": collector,
        "C_max": C_MAX,
        "n_attack_edges": n_attack_edges,
        "p_sybil_stats": {
            "min": round(min(p_sybil.values()), 5),
            "max": round(max(p_sybil.values()), 5),
            "mean": round(sum(p_sybil.values()) / len(p_sybil), 5),
        },
        "sumup_baseline": result_sumup,
        "naaf_raw_1_minus_p": result_naaf_raw,
        "naaf_percentile_rank": result_naaf_pct,
        "naaf_raw_vs_sumup": compare(result_sumup, result_naaf_raw),
        "naaf_percentile_vs_sumup": compare(result_sumup, result_naaf_pct),
    }
    return results


if __name__ == "__main__":
    dense_results = run_scenario(
        "dense",
        "../data/combined_graph.edgelist",
        "../data/sybil_labels.json",
        lambda G, s: fusion_dense.get_sybil_probabilities(G, s),
    )
    sparse_results = run_scenario(
        "sparse",
        "../data/combined_graph_sparse.edgelist",
        "../data/sybil_labels_sparse.json",
        lambda G, s: fusion_sparse.get_sybil_probabilities(G, s, "../results/gatv2_model_sparse.pt"),
    )

    with open("../results/percentile_experiment_dense.json", "w") as f:
        json.dump(dense_results, f, indent=2)
    with open("../results/percentile_experiment_sparse.json", "w") as f:
        json.dump(sparse_results, f, indent=2)

    print(json.dumps({"dense": dense_results, "sparse": sparse_results}, indent=2))
    print("\nSaved ../results/percentile_experiment_dense.json and "
          "../results/percentile_experiment_sparse.json (new files only -- "
          "naaf_fusion*.json and ablation_study*.json untouched)")
