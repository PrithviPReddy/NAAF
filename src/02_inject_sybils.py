"""
Stage 3, Step 2: Inject a synthetic Sybil cluster onto the real honest graph.

Rationale (see Stage 1 lit review): SybilGAT and the resistance paper both
evaluate on real honest graphs with SYNTHETICALLY attached Sybil clusters,
specifically because ground-truth Sybil labels don't exist in the wild.
We follow the same methodology here, not as a shortcut but because it's
literally how our two base papers do it.

Design:
  - Sybil-Sybil edges: attacker fully controls these identities, so they
    can be densely interconnected for free -> Erdos-Renyi with high p.
  - Attack edges (honest -> Sybil): expensive for the attacker (requires
    fooling a real user), so these are deliberately SCARCE and
    configurable -- this is the parameter we'll sweep later (1/4/8/12
    per Sybil, matching SybilGAT's own experiment design).

Output: one combined graph with node attribute 'label' (0=honest,
1=sybil) and edge attribute 'edge_type' (honest / sybil / attack),
so every later stage (SumUp, SybilRank, GATv2 training, evaluation)
can read ground truth directly off the graph.
"""
import networkx as nx
import random
import json

def inject_sybils(G_honest, n_sybil, attack_edges_per_sybil,
                   internal_p=0.3, seed=42):
    rng = random.Random(seed)
    honest_nodes = list(G_honest.nodes())
    n_honest = len(honest_nodes)

    # Sybil node IDs continue after the honest ones so there's no collision
    sybil_nodes = list(range(n_honest, n_honest + n_sybil))

    G = nx.Graph()
    G.add_nodes_from(honest_nodes, label=0)
    G.add_edges_from(G_honest.edges(), edge_type="honest")
    G.add_nodes_from(sybil_nodes, label=1)

    # Sybil-Sybil edges: cheap, dense, attacker-controlled
    G_sybil_internal = nx.erdos_renyi_graph(n_sybil, internal_p, seed=seed)
    relabel = {i: sybil_nodes[i] for i in range(n_sybil)}
    G_sybil_internal = nx.relabel_nodes(G_sybil_internal, relabel)
    G.add_edges_from(G_sybil_internal.edges(), edge_type="sybil")

    # Attack edges: expensive, scarce, configurable -- each Sybil gets
    # exactly `attack_edges_per_sybil` distinct random honest targets
    attack_edges = set()
    for s in sybil_nodes:
        targets = rng.sample(honest_nodes, attack_edges_per_sybil)
        for h in targets:
            attack_edges.add((h, s))
    G.add_edges_from(attack_edges, edge_type="attack")

    return G, sybil_nodes, attack_edges


if __name__ == "__main__":
    G_honest = nx.read_edgelist("../data/honest_graph.edgelist", nodetype=int)

    N_SYBIL = 200                 # ~5% of 4,039 honest nodes
    ATTACK_EDGES_PER_SYBIL = 8    # SybilGAT's "standard" synthetic setting

    G, sybil_nodes, attack_edges = inject_sybils(
        G_honest, N_SYBIL, ATTACK_EDGES_PER_SYBIL
    )

    n_honest = G_honest.number_of_nodes()
    edge_types = nx.get_edge_attributes(G, "edge_type")
    counts = {"honest": 0, "sybil": 0, "attack": 0}
    for t in edge_types.values():
        counts[t] += 1

    stats = {
        "n_honest_nodes": n_honest,
        "n_sybil_nodes": len(sybil_nodes),
        "n_total_nodes": G.number_of_nodes(),
        "honest_honest_edges": counts["honest"],
        "sybil_sybil_edges": counts["sybil"],
        "attack_edges": counts["attack"],
        "expected_attack_edges": N_SYBIL * ATTACK_EDGES_PER_SYBIL,
        "n_total_edges": G.number_of_edges(),
        "is_connected": nx.is_connected(G),
    }
    print(json.dumps(stats, indent=2))

    # --- verification gates ---
    assert stats["n_total_nodes"] == n_honest + N_SYBIL
    assert stats["honest_honest_edges"] == G_honest.number_of_edges(), \
        "Original honest edges were altered -- bug in injection"
    # exact match is only guaranteed if no duplicate (h,s) pairs were sampled
    # across different sybils targeting the same honest node -- that's fine,
    # duplicates across DIFFERENT sybils are distinct edges, but same-sybil
    # sampling used rng.sample (no replacement) so is duplicate-free per sybil
    assert stats["attack_edges"] == stats["expected_attack_edges"], \
        f"Attack edge count off: got {stats['attack_edges']}, expected {stats['expected_attack_edges']}"
    assert stats["is_connected"], "Graph became disconnected after injection -- unexpected"
    print("\nVERIFIED: node/edge counts and connectivity all check out.")

    nx.write_gpickle = None
    nx.write_edgelist(G, "../data/combined_graph.edgelist", data=["edge_type"])
    with open("../data/sybil_labels.json", "w") as f:
        json.dump({"sybil_nodes": sybil_nodes,
                   "attack_edges": [list(e) for e in attack_edges]}, f)
    print("Saved ../data/combined_graph.edgelist and ../data/sybil_labels.json")
