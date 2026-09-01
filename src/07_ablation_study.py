"""
Stage 3, Step 6b: Controlled ablation of the NAAF capacity-scaling formula.

Diagnostic only -- does NOT overwrite the official Step 3 (SumUp) or
Step 6 (NAAF) locked results. Same graph, same collector, same C_max,
same 200-voter sample (seed=42), same attack edges throughout every
variant -- only the capacity-scaling FORMULA changes, so any difference
between rows is attributable to the formula alone, not the setup.
"""
import networkx as nx
import json
import random
from importlib import import_module

sumup = import_module("03_sumup_baseline")
fusion = import_module("06_naaf_fusion")

if __name__ == "__main__":
    G = nx.read_edgelist("../data/combined_graph.edgelist", nodetype=int, data=(("edge_type", str),))
    with open("../data/sybil_labels.json") as f:
        sybil_labels = json.load(f)
    sybil_nodes = set(sybil_labels["sybil_nodes"])
    n_attack_edges = len(sybil_labels["attack_edges"])
    honest_nodes = [n for n in G.nodes() if n not in sybil_nodes]

    p_sybil = fusion.get_sybil_probabilities(G, sybil_nodes)

    degrees = dict(G.degree(honest_nodes))
    median_degree = sorted(degrees.values())[len(degrees) // 2]
    collector = min(honest_nodes, key=lambda n: abs(degrees[n] - median_degree))
    levels = sumup.bfs_levels(G, collector)
    C_MAX = 1000
    capacity_sumup = sumup.ticket_distribution(G, levels, collector, C_MAX)

    rng = random.Random(42)
    honest_voter_sample = rng.sample(honest_nodes, 200)  # identical sample, every variant

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

    variants = {}

    # 1. SumUp baseline: no scaling at all
    variants["1_sumup_baseline"] = capacity_sumup

    # 2. Current NAAF: C = C_sumup * (1 - P_sybil)
    variants["2_naaf_current_alpha1.0"] = {
        (u, v): cap * (1 - p_sybil[v]) for (u, v), cap in capacity_sumup.items()
    }

    # 3. Thresholded: only penalize when P_sybil > 0.5
    variants["3_thresholded_p_gt_0.5"] = {
        (u, v): (cap if p_sybil[v] <= 0.5 else cap * (1 - p_sybil[v]))
        for (u, v), cap in capacity_sumup.items()
    }

    # 4. Milder scaling sweep: C = C_sumup * (1 - alpha * P_sybil)
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        variants[f"4_milder_alpha{alpha}"] = {
            (u, v): cap * (1 - alpha * p_sybil[v]) for (u, v), cap in capacity_sumup.items()
        }

    results = {}
    for name, cap_dict in variants.items():
        results[name] = evaluate(cap_dict)

    base = results["1_sumup_baseline"]
    for name, r in results.items():
        attacker_reduction_pct = round(
            100 * (1 - r["attacker_capacity_per_attack_edge"] /
                   max(base["attacker_capacity_per_attack_edge"], 1e-9)), 2)
        honest_change_pct = round(
            100 * (r["honest_collection_rate"] - base["honest_collection_rate"]) /
            max(base["honest_collection_rate"], 1e-9), 2)
        honest_loss_pp = round(
            (base["honest_collection_rate"] - r["honest_collection_rate"]) * 100, 3)
        trade_off_ratio = (round(attacker_reduction_pct / honest_loss_pp, 3)
                            if honest_loss_pp > 0.001 else None)
        r["attacker_influence_change_pct_vs_sumup"] = -attacker_reduction_pct
        r["honest_collection_change_pct_vs_sumup"] = honest_change_pct
        r["honest_collection_loss_pp_vs_sumup"] = honest_loss_pp
        r["attacker_reduction_per_1pp_honest_lost"] = trade_off_ratio

    print(f"{'variant':<26} {'atk/edge':>9} {'atk_flow':>9} {'honest_flow':>12} "
          f"{'honest_rate':>12} {'atk_chg%':>9} {'honest_chg%':>12} {'atk_cut_per_1pp':>16}")
    for name, r in results.items():
        print(f"{name:<26} {r['attacker_capacity_per_attack_edge']:>9} "
              f"{r['attacker_flow_collected']:>9} {r['honest_flow_collected']:>12} "
              f"{r['honest_collection_rate']:>12} "
              f"{r['attacker_influence_change_pct_vs_sumup']:>9} "
              f"{r['honest_collection_change_pct_vs_sumup']:>12} "
              f"{str(r['attacker_reduction_per_1pp_honest_lost']):>16}")

    with open("../results/ablation_study.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved ../results/ablation_study.json (diagnostic only -- official results untouched)")
