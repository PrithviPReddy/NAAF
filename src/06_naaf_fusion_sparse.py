"""
Sparse-attack scenario: NAAF fusion, rerun on the sparse Sybil injection
(25 Sybils, 200 attack edges) using the sparse-trained GATv2 model
(gatv2_model_sparse.pt from 05_gatv2_sparse.py) -- NOT the dense model,
since it was trained on a different graph.

capacity_NAAF(u, v) = capacity_SumUp(u, v) * (1 - P(Sybil | v))

Reuses the EXACT SumUp machinery (03_sumup_baseline) and the EXACT
NaafGATv2 model class (05_gatv2_layer) unmodified. The
get_sybil_probabilities() logic below is a parametrized copy of
06_naaf_fusion.py's version (model_path/graph-derived feature
construction as arguments instead of hardcoded dense paths), so the
dense scenario's script and locked results are never touched.
"""
import networkx as nx
import numpy as np
import json
import random
import torch
from importlib import import_module

sumup = import_module("03_sumup_baseline")
gat = import_module("05_gatv2_layer")


def get_sybil_probabilities(G, sybil_nodes, model_path):
    from torch_geometric.utils import from_networkx
    data = from_networkx(G)
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    degree = np.array([G.degree(v) for v in nodes], dtype=float)
    norm_degree = (degree - degree.mean()) / degree.std()

    rng = random.Random(42)
    honest_idx = [node_idx[v] for v in nodes if v not in sybil_nodes]
    revealed = set(rng.sample(honest_idx, int(0.2 * len(honest_idx))))  # fair variant: honest-only
    true_label = np.array([1.0 if v in sybil_nodes else 0.0 for v in nodes])
    revealed_label_feature = np.full(len(nodes), 0.5)
    for i in revealed:
        revealed_label_feature[i] = true_label[i]

    X = torch.tensor(np.stack([norm_degree, revealed_label_feature], axis=1), dtype=torch.float)
    model = gat.NaafGATv2()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(X, data.edge_index)
        probs = torch.sigmoid(logits).numpy()
    return {nodes[i]: float(probs[i]) for i in range(len(nodes))}


if __name__ == "__main__":
    G = nx.read_edgelist("../data/combined_graph_sparse.edgelist", nodetype=int,
                          data=(("edge_type", str),))
    with open("../data/sybil_labels_sparse.json") as f:
        sybil_labels = json.load(f)
    sybil_nodes = set(sybil_labels["sybil_nodes"])
    n_attack_edges = len(sybil_labels["attack_edges"])
    honest_nodes = [n for n in G.nodes() if n not in sybil_nodes]

    p_sybil = get_sybil_probabilities(G, sybil_nodes, "../results/gatv2_model_sparse.pt")
    print(f"P(Sybil) computed for {len(p_sybil)} nodes. "
          f"Mean P(Sybil) -- honest: {np.mean([p_sybil[n] for n in honest_nodes]):.4f}, "
          f"sybil: {np.mean([p_sybil[n] for n in sybil_nodes]):.4f}")

    degrees = dict(G.degree(honest_nodes))
    median_degree = sorted(degrees.values())[len(degrees) // 2]
    collector = min(honest_nodes, key=lambda n: abs(degrees[n] - median_degree))
    levels = sumup.bfs_levels(G, collector)
    C_MAX = 1000
    capacity_sumup = sumup.ticket_distribution(G, levels, collector, C_MAX)

    capacity_naaf = {(u, v): cap * (1 - p_sybil[v]) for (u, v), cap in capacity_sumup.items()}

    flow_graph_sumup = sumup.build_flow_graph(G, levels, capacity_sumup)
    flow_graph_naaf = sumup.build_flow_graph(G, levels, capacity_naaf)

    rng = random.Random(42)
    honest_voter_sample = rng.sample(honest_nodes, 200)

    def evaluate(flow_graph):
        attacker_flow, _ = sumup.votes_collected(flow_graph, collector, sybil_nodes)
        honest_flow, n_h = sumup.votes_collected(flow_graph, collector, honest_voter_sample)
        return {
            "attacker_flow_collected": attacker_flow,
            "attacker_capacity_per_attack_edge": round(attacker_flow / n_attack_edges, 4),
            "honest_flow_collected": honest_flow,
            "honest_collection_rate": round(honest_flow / n_h, 4),
        }

    THRESHOLD = 0.5
    capacity_naaf_v2 = {(u, v): cap if p_sybil[v] <= THRESHOLD else cap * (1 - p_sybil[v])
                          for (u, v), cap in capacity_sumup.items()}
    flow_graph_naaf_v2 = sumup.build_flow_graph(G, levels, capacity_naaf_v2)

    result_sumup = evaluate(flow_graph_sumup)
    result_naaf = evaluate(flow_graph_naaf)
    result_naaf_v2 = evaluate(flow_graph_naaf_v2)

    print("\n=== SumUp (recomputed on sparse graph, sanity check) ===")
    print(json.dumps(result_sumup, indent=2))
    print("\n=== NAAF v1 (continuous scaling, all edges) ===")
    print(json.dumps(result_naaf, indent=2))
    print("\n=== NAAF v2 (threshold-gated, only P(Sybil)>0.5) ===")
    print(json.dumps(result_naaf_v2, indent=2))

    def compare(base, variant):
        return {
            "attacker_capacity_reduction_pct": round(
                100 * (1 - variant["attacker_capacity_per_attack_edge"] /
                       max(base["attacker_capacity_per_attack_edge"], 1e-9)), 2),
            "honest_collection_change_pct_points": round(
                100 * (variant["honest_collection_rate"] - base["honest_collection_rate"]), 2),
        }

    results = {
        "scenario": "sparse",
        "n_sybil_nodes": len(sybil_nodes),
        "collector": collector,
        "C_max": C_MAX,
        "sybil_threshold_v2": THRESHOLD,
        "n_attack_edges": n_attack_edges,
        "sumup_baseline_recomputed": result_sumup,
        "naaf_v1_continuous": result_naaf,
        "naaf_v2_threshold_gated": result_naaf_v2,
        "improvement_v1_vs_sumup": compare(result_sumup, result_naaf),
        "improvement_v2_vs_sumup": compare(result_sumup, result_naaf_v2),
    }
    with open("../results/naaf_fusion_sparse.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved ../results/naaf_fusion_sparse.json")
