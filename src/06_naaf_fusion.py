"""
Stage 3, Step 6: NAAF fusion -- GATv2 Sybil-probability scales SumUp's
ticket-based capacities.

capacity_NAAF(u, v) = capacity_SumUp(u, v) * (1 - P(Sybil | v))

Reuses the EXACT SumUp machinery from Step 3 (same collector, same
C_max=1000, same 200-voter evaluation protocol) so this is a true
apples-to-apples comparison, not a different experiment. Reuses the
EXACT feature-construction from Step 5's fair (honest-only-seeded)
GATv2 run so the loaded model's predictions are valid for its own inputs.
"""
import networkx as nx
import numpy as np
import json
import random
import torch
from importlib import import_module

sumup = import_module("03_sumup_baseline")
gat = import_module("05_gatv2_layer")


def get_sybil_probabilities(G, sybil_nodes):
    """Rebuild the exact fair-variant features from Step 5, load the
    trained model, return P(Sybil) for every node."""
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
    model.load_state_dict(torch.load("../results/gatv2_model.pt"))
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(X, data.edge_index)
        probs = torch.sigmoid(logits).numpy()
    return {nodes[i]: float(probs[i]) for i in range(len(nodes))}


if __name__ == "__main__":
    G = nx.read_edgelist("../data/combined_graph.edgelist", nodetype=int, data=(("edge_type", str),))
    with open("../data/sybil_labels.json") as f:
        sybil_labels = json.load(f)
    sybil_nodes = set(sybil_labels["sybil_nodes"])
    n_attack_edges = len(sybil_labels["attack_edges"])
    honest_nodes = [n for n in G.nodes() if n not in sybil_nodes]

    p_sybil = get_sybil_probabilities(G, sybil_nodes)
    print(f"P(Sybil) computed for {len(p_sybil)} nodes. "
          f"Mean P(Sybil) -- honest: {np.mean([p_sybil[n] for n in honest_nodes]):.4f}, "
          f"sybil: {np.mean([p_sybil[n] for n in sybil_nodes]):.4f}")

    # --- identical protocol to Step 3 ---
    degrees = dict(G.degree(honest_nodes))
    median_degree = sorted(degrees.values())[len(degrees) // 2]
    collector = min(honest_nodes, key=lambda n: abs(degrees[n] - median_degree))
    levels = sumup.bfs_levels(G, collector)
    C_MAX = 1000
    capacity_sumup = sumup.ticket_distribution(G, levels, collector, C_MAX)

    # --- NAAF fusion: scale by (1 - P(Sybil| target node)) ---
    capacity_naaf = {(u, v): cap * (1 - p_sybil[v]) for (u, v), cap in capacity_sumup.items()}

    flow_graph_sumup = sumup.build_flow_graph(G, levels, capacity_sumup)
    flow_graph_naaf = sumup.build_flow_graph(G, levels, capacity_naaf)

    rng = random.Random(42)
    honest_voter_sample = rng.sample(honest_nodes, 200)  # identical sample as Step 3

    def evaluate(flow_graph, label):
        attacker_flow, _ = sumup.votes_collected(flow_graph, collector, sybil_nodes)
        honest_flow, n_h = sumup.votes_collected(flow_graph, collector, honest_voter_sample)
        return {
            "attacker_flow_collected": attacker_flow,
            "attacker_capacity_per_attack_edge": round(attacker_flow / n_attack_edges, 4),
            "honest_flow_collected": honest_flow,
            "honest_collection_rate": round(honest_flow / n_h, 4),
        }

    # Refined variant: threshold-gated scaling. Diagnostic showed 143/4039
    # honest nodes have a moderate P(Sybil) between 0.1-0.5 -- continuous
    # multiplicative scaling penalizes these on every edge they sit behind,
    # compounding across multi-hop paths. Only throttle when the model is
    # actually confident (P > 0.5), a standard decision boundary -- leave
    # everything else at full SumUp capacity.
    THRESHOLD = 0.5
    capacity_naaf_v2 = {(u, v): cap if p_sybil[v] <= THRESHOLD else cap * (1 - p_sybil[v])
                          for (u, v), cap in capacity_sumup.items()}
    flow_graph_naaf_v2 = sumup.build_flow_graph(G, levels, capacity_naaf_v2)

    result_sumup = evaluate(flow_graph_sumup, "SumUp (recomputed, should match Step 3)")
    result_naaf = evaluate(flow_graph_naaf, "NAAF v1 (continuous scaling, all edges)")
    result_naaf_v2 = evaluate(flow_graph_naaf_v2, "NAAF v2 (threshold-gated, P>0.5 only)")

    print("\n=== SumUp (recomputed on this script, sanity check vs Step 3) ===")
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
    with open("../results/naaf_fusion.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved ../results/naaf_fusion.json")
