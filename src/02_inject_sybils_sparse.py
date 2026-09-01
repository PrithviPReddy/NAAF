"""
Stage 3, Sparse-attack scenario: inject a SPARSE Sybil cluster onto the
same real honest graph, as a separate dataset alongside the dense one.

Rationale for the parameter choice (do not touch defaults casually --
this is deliberately justified from the literature, not tuned by feel):

  - attack_edges_per_sybil stays at 8, UNCHANGED from the dense scenario.
    This is SybilGAT's own "standard" synthetic-experiment setting, i.e.
    a literature-precedented rate rather than a knob we get to tune.
  - SybilRank's guarantee is explicitly conditioned on Sybils establishing
    "a limited number of attack edges due to the difficulty of soliciting
    and maintaining reciprocal social relationships" -- the sparse-cut
    assumption every one of our base papers (SumUp, SybilRank, SybilGAT)
    relies on is about attack edges being a SMALL MINORITY of the honest
    population, not about internal Sybil-cluster density (which costs the
    attacker nothing -- no literature reason to shrink that either, so
    internal_p stays at 0.3, unchanged).
  - The dense scenario (200 Sybils x 8 = 1,600 attack edges) covers 33.4%
    of the 4,039-node honest population -- nowhere near "sparse" by that
    definition.
  - Sparse scenario: reduce Sybil COUNT 200 -> 25 (8x reduction) instead,
    giving 25 x 8 = 200 total attack edges against the same 4,039-node
    honest population -- ~5% raw coverage before overlap, comfortably in
    the sparse-cut regime the theory assumes.

This script only ADDS new files (combined_graph_sparse.edgelist,
sybil_labels_sparse.json). It does not read, write, or otherwise touch
combined_graph.edgelist or sybil_labels.json (the dense scenario's
locked files).
"""
import networkx as nx
import json
from importlib import import_module

dense = import_module("02_inject_sybils")
inject_sybils = dense.inject_sybils

if __name__ == "__main__":
    G_honest = nx.read_edgelist("../data/honest_graph.edgelist", nodetype=int)

    N_SYBIL = 25                  # 8x fewer than dense scenario's 200
    ATTACK_EDGES_PER_SYBIL = 8    # unchanged -- SybilGAT's standard setting
    INTERNAL_P = 0.3              # unchanged -- internal density is free for the attacker

    G, sybil_nodes, attack_edges = inject_sybils(
        G_honest, N_SYBIL, ATTACK_EDGES_PER_SYBIL, internal_p=INTERNAL_P, seed=42
    )

    n_honest = G_honest.number_of_nodes()
    edge_types = nx.get_edge_attributes(G, "edge_type")
    counts = {"honest": 0, "sybil": 0, "attack": 0}
    for t in edge_types.values():
        counts[t] += 1

    stats = {
        "scenario": "sparse",
        "n_honest_nodes": n_honest,
        "n_sybil_nodes": len(sybil_nodes),
        "n_total_nodes": G.number_of_nodes(),
        "honest_honest_edges": counts["honest"],
        "sybil_sybil_edges": counts["sybil"],
        "attack_edges": counts["attack"],
        "expected_attack_edges": N_SYBIL * ATTACK_EDGES_PER_SYBIL,
        "attack_edge_coverage_of_honest_pop_pct": round(
            100 * counts["attack"] / n_honest, 2),
        "n_total_edges": G.number_of_edges(),
        "is_connected": nx.is_connected(G),
    }
    print(json.dumps(stats, indent=2))

    # --- verification gates (mirrors dense scenario's checks exactly) ---
    assert stats["n_total_nodes"] == n_honest + N_SYBIL
    assert stats["honest_honest_edges"] == G_honest.number_of_edges(), \
        "Original honest edges were altered -- bug in injection"
    assert stats["attack_edges"] == stats["expected_attack_edges"], \
        f"Attack edge count off: got {stats['attack_edges']}, expected {stats['expected_attack_edges']}"
    assert stats["is_connected"], "Graph became disconnected after injection -- unexpected"
    print("\nVERIFIED: node/edge counts and connectivity all check out.")

    nx.write_edgelist(G, "../data/combined_graph_sparse.edgelist", data=["edge_type"])
    with open("../data/sybil_labels_sparse.json", "w") as f:
        json.dump({"sybil_nodes": sybil_nodes,
                   "attack_edges": [list(e) for e in attack_edges]}, f)
    print("Saved ../data/combined_graph_sparse.edgelist and ../data/sybil_labels_sparse.json")
    print("\n(dense scenario files combined_graph.edgelist / sybil_labels.json untouched)")
