"""
Stage 3, Step 9: NAAF Option B (suspicious-boundary / attack-edge
adaptation) as the PRIMARY formulation, with Option A (source/voter-level
trust weighting) kept as a comparison/ablation and Option C (conservative,
confidence-gated variant of A) included as time permitted.

Reruns on BOTH the dense and sparse scenarios under the exact locked
evaluation protocol (median-degree collector, C_max=1000, 200-honest-
voter sample seed=42, all-Sybil-voters, 1 vote cap each). Writes ONLY
new files (results/naaf_option_b_dense.json,
results/naaf_option_b_sparse.json) -- never opens combined_graph*.edgelist,
sybil_labels*.json, or any existing results/*.json in write mode.

====================================================================
How an "estimated attack boundary edge" is identified, WITHOUT using
ground-truth Sybil labels
====================================================================

We need a threshold on GATv2's raw P(Sybil) output above which a node is
"confidently" Sybil-like, so capacity is only ever throttled on genuine
high-confidence detections (the whole point of Option B is to avoid what
sank NAAF v1: moderate/ambiguous scores, e.g. P in [0.1, 0.5], causing
honest collateral damage on shared edges).

REJECTED design: a single fixed constant threshold, e.g. tau=0.8, shared
across scenarios. This has the EXACT calibration flaw already diagnosed
for raw (1-P) scaling and for raw percentile-vs-magnitude scaling: in the
sparse scenario, GATv2's raw P(Sybil) output never exceeds ~0.022 for ANY
node in the graph (confirmed empirically in 07_ablation_study_sparse.py).
A fixed tau=0.8 would simply never fire there -- Option B would silently
degenerate to plain SumUp on the sparse graph, reproducing the exact
failure mode it exists to fix.

DESIGN USED HERE: a threshold calibrated PER SCENARIO from the
classifier's own output distribution, using only information already
legitimately available under this project's "fair" (honest-only-seeded)
semi-supervised setup from Step 5 -- the REVEALED HONEST node labels
(the same 20%-of-honest-nodes subset already used to train GATv2). This
uses ZERO Sybil ground truth, by construction: it only asks "how high
does P(Sybil) get among nodes the training setup itself already tells us
ARE honest?" and flags anything scoring higher than that as anomalous.

    revealed_honest   = the identical 20% honest-node subset from Step 5
                         (same seed=42 reconstruction, same node ordering)
    tau_hi(scenario)  = the q_hi-quantile of { P(Sybil|v) : v in revealed_honest }
    tau_lo(scenario)  = the q_lo-quantile of { P(Sybil|v) : v in revealed_honest }
                        (q_lo < q_hi; tau_lo is used only by Option C's
                        smooth gate, not by Option B)

Defaults: q_hi = 0.99, q_lo = 0.90. tau_hi is calibrated so that, AMONG
NODES WE ALREADY KNOW ARE HONEST, only ~1% would exceed it by chance --
i.e. it targets a fixed false-positive rate on known-honest nodes and
automatically adapts to each scenario's own probability scale (the wide
dense-scenario spread vs. the tiny-magnitude sparse-scenario spread)
instead of assuming a shared absolute cutoff.

Confident-Sybil set and boundary/attack-edge definition:

    S_hat = { v : P(Sybil|v) > tau_hi }

A forward edge (u, v) in the ticket-distribution tree (u closer to the
collector, v farther) is treated as an "estimated attack/boundary edge"
whenever its FAR endpoint v is in S_hat -- i.e. it is either a genuine
entry edge crossing into the confident-Sybil region, or an edge already
inside a confirmed-Sybil subtree (deliberately also throttled: by the
injection methodology itself, no honest voter's path to the collector
can ever need to route through a confident-Sybil node's subtree, so
throttling there costs nothing on the honest side -- see the design
review for the full structural argument).

--------------------------------------------------------------------
Formulas
--------------------------------------------------------------------

Option B (primary):
    capacity_NAAF-B(u,v) = capacity_SumUp(u,v) * (1 - P(Sybil|v))   if v in S_hat
                          = capacity_SumUp(u,v)                       otherwise

Option A (comparison/ablation) -- unchanged from the design review,
touches ONLY the source/voter entry point, never an intermediate edge:
    capacity_NAAF-A(SOURCE -> voter) = 1 * (1 - P(Sybil|voter))
    capacity_NAAF-A(u,v) = capacity_SumUp(u,v)   for every intermediate edge

Option C (conservative variant) -- smooth confidence gate at the voter
level, dead zone below tau_lo, saturating at tau_hi:
    g(p) = clip( (p - tau_lo) / (tau_hi - tau_lo), 0, 1 )
    capacity_NAAF-C(SOURCE -> voter) = 1 * (1 - g(P(Sybil|voter)))
    capacity_NAAF-C(u,v) = capacity_SumUp(u,v)   for every intermediate edge
"""
import networkx as nx
import numpy as np
import json
import random
from importlib import import_module

sumup = import_module("03_sumup_baseline")
fusion_dense = import_module("06_naaf_fusion")
fusion_sparse = import_module("06_naaf_fusion_sparse")


def get_revealed_honest_nodes(G, sybil_nodes, seed=42, frac=0.2):
    """Exact reconstruction of the 20% honest-node subset revealed to
    GATv2 during the 'fair' training run in Step 5 -- same node ordering
    (list(G.nodes())), same seed, same sampling call, so this recovers
    the identical node SET used for training (see reasoning in the
    accompanying writeup: random.sample's choice of positions depends
    only on population length and seed, and honest-node ordering here is
    reconstructed identically to Step 5/06's own internal logic)."""
    nodes = list(G.nodes())
    honest_nodes = [v for v in nodes if v not in sybil_nodes]
    rng = random.Random(seed)
    return set(rng.sample(honest_nodes, int(frac * len(honest_nodes))))


def calibrate_thresholds(p_sybil, revealed_honest, q_hi=0.99, q_lo=0.90):
    calib_scores = np.array([p_sybil[v] for v in revealed_honest])
    tau_hi = float(np.quantile(calib_scores, q_hi))
    tau_lo = float(np.quantile(calib_scores, q_lo))
    return tau_hi, tau_lo


def gate(p, tau_lo, tau_hi):
    if tau_hi <= tau_lo:
        return 1.0 if p > tau_lo else 0.0
    return max(0.0, min(1.0, (p - tau_lo) / (tau_hi - tau_lo)))


def votes_collected_weighted(flow_graph, collector, voter_weights):
    """Like sumup.votes_collected, but each voter gets its OWN capacity
    (dict voter -> weight in [0,1]) instead of a shared cap_per_voter."""
    F = flow_graph.copy()
    F.add_node("SOURCE")
    reachable = {v: w for v, w in voter_weights.items() if v in flow_graph.nodes()}
    for v, w in reachable.items():
        F.add_edge("SOURCE", v, capacity=w)
    value, _ = nx.maximum_flow(F, "SOURCE", collector)
    return value, len(reachable)


def run_scenario(scenario_name, graph_path, labels_path, get_p_sybil_fn,
                  q_hi=0.99, q_lo=0.90):
    G = nx.read_edgelist(graph_path, nodetype=int, data=(("edge_type", str),))
    with open(labels_path) as f:
        labels = json.load(f)
    sybil_nodes = set(labels["sybil_nodes"])
    n_attack_edges = len(labels["attack_edges"])
    honest_nodes = [n for n in G.nodes() if n not in sybil_nodes]

    p_sybil = get_p_sybil_fn(G, sybil_nodes)
    revealed_honest = get_revealed_honest_nodes(G, sybil_nodes)
    tau_hi, tau_lo = calibrate_thresholds(p_sybil, revealed_honest, q_hi, q_lo)
    S_hat = {v for v, p in p_sybil.items() if p > tau_hi}

    # --- diagnostic only (uses ground truth for EVALUATION of the boundary
    # detector's precision/recall -- never for choosing tau_hi/tau_lo above) ---
    true_attack_edge_set = {tuple(e) for e in labels["attack_edges"]} | \
                            {(e[1], e[0]) for e in labels["attack_edges"]}
    edge_types = nx.get_edge_attributes(G, "edge_type")

    # --- identical collector/C_max/levels derivation as every prior script ---
    degrees = dict(G.degree(honest_nodes))
    median_degree = sorted(degrees.values())[len(degrees) // 2]
    collector = min(honest_nodes, key=lambda n: abs(degrees[n] - median_degree))
    levels = sumup.bfs_levels(G, collector)
    C_MAX = 1000
    capacity_sumup = sumup.ticket_distribution(G, levels, collector, C_MAX)

    n_flagged_edges = 0
    n_flagged_true_attack = 0
    for (u, v) in capacity_sumup:
        if v in S_hat:
            n_flagged_edges += 1
            if (u, v) in true_attack_edge_set:
                n_flagged_true_attack += 1
    n_true_attack_edges_in_tree = sum(
        1 for (u, v) in capacity_sumup if (u, v) in true_attack_edge_set)

    capacity_naaf_v1 = {(u, v): cap * (1 - p_sybil[v]) for (u, v), cap in capacity_sumup.items()}
    capacity_naaf_b = {(u, v): (cap * (1 - p_sybil[v]) if v in S_hat else cap)
                        for (u, v), cap in capacity_sumup.items()}

    rng = random.Random(42)
    honest_voter_sample = rng.sample(honest_nodes, 200)  # identical seed/size, every prior script

    def evaluate_uniform(capacity_dict):
        flow_graph = sumup.build_flow_graph(G, levels, capacity_dict)
        attacker_flow, _ = sumup.votes_collected(flow_graph, collector, sybil_nodes)
        honest_flow, n_h = sumup.votes_collected(flow_graph, collector, honest_voter_sample)
        return {
            "attacker_flow_collected": round(attacker_flow, 3),
            "attacker_capacity_per_attack_edge": round(attacker_flow / n_attack_edges, 4),
            "honest_flow_collected": round(honest_flow, 3),
            "honest_collection_rate": round(honest_flow / n_h, 4),
        }

    def evaluate_source_weighted(weight_fn):
        # capacities are the ORIGINAL, untouched SumUp tree -- only the
        # SOURCE->voter edges are individually weighted (Options A, C)
        flow_graph = sumup.build_flow_graph(G, levels, capacity_sumup)
        attacker_weights = {v: weight_fn(v) for v in sybil_nodes}
        honest_weights = {v: weight_fn(v) for v in honest_voter_sample}
        attacker_flow, _ = votes_collected_weighted(flow_graph, collector, attacker_weights)
        honest_flow, n_h = votes_collected_weighted(flow_graph, collector, honest_weights)
        return {
            "attacker_flow_collected": round(attacker_flow, 3),
            "attacker_capacity_per_attack_edge": round(attacker_flow / n_attack_edges, 4),
            "honest_flow_collected": round(honest_flow, 3),
            "honest_collection_rate": round(honest_flow / n_h, 4),
        }

    result_sumup = evaluate_uniform(capacity_sumup)
    result_v1 = evaluate_uniform(capacity_naaf_v1)
    result_b = evaluate_uniform(capacity_naaf_b)
    result_a = evaluate_source_weighted(lambda v: 1 - p_sybil[v])
    result_c = evaluate_source_weighted(lambda v: 1 - gate(p_sybil[v], tau_lo, tau_hi))

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
        "p_sybil_stats_all_nodes": {
            "min": round(min(p_sybil.values()), 5),
            "max": round(max(p_sybil.values()), 5),
            "mean": round(sum(p_sybil.values()) / len(p_sybil), 5),
        },
        "threshold_calibration": {
            "method": "quantile of P(Sybil) among the revealed-HONEST training "
                      "subset only -- no Sybil ground truth used",
            "n_revealed_honest": len(revealed_honest),
            "q_hi": q_hi, "tau_hi": round(tau_hi, 6),
            "q_lo": q_lo, "tau_lo": round(tau_lo, 6),
            "n_confident_sybil_nodes_flagged": len(S_hat),
        },
        "boundary_detection_diagnostic_evaluation_only_not_used_to_set_tau": {
            "n_edges_flagged_far_endpoint_in_Shat": n_flagged_edges,
            "n_flagged_edges_that_are_true_attack_edges": n_flagged_true_attack,
            "precision_flagged_edges_are_attack_edges": (
                round(n_flagged_true_attack / n_flagged_edges, 4) if n_flagged_edges else None),
            "n_true_attack_edges_present_in_ticket_tree": n_true_attack_edges_in_tree,
            "recall_of_true_attack_edges_flagged": (
                round(n_flagged_true_attack / n_true_attack_edges_in_tree, 4)
                if n_true_attack_edges_in_tree else None),
        },
        "sumup_baseline": result_sumup,
        "naaf_v1_raw_all_edges": result_v1,
        "naaf_b_boundary_primary": result_b,
        "naaf_a_source_weighted_ablation": result_a,
        "naaf_c_conservative_variant": result_c,
        "naaf_v1_vs_sumup": compare(result_sumup, result_v1),
        "naaf_b_vs_sumup": compare(result_sumup, result_b),
        "naaf_a_vs_sumup": compare(result_sumup, result_a),
        "naaf_c_vs_sumup": compare(result_sumup, result_c),
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

    with open("../results/naaf_option_b_dense.json", "w") as f:
        json.dump(dense_results, f, indent=2)
    with open("../results/naaf_option_b_sparse.json", "w") as f:
        json.dump(sparse_results, f, indent=2)

    for r in (dense_results, sparse_results):
        print(f"\n=== {r['scenario'].upper()} ===")
        print(json.dumps(r, indent=2))

    print("\nSaved ../results/naaf_option_b_dense.json and "
          "../results/naaf_option_b_sparse.json (new files only)")
