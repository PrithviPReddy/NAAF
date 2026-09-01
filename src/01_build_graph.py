"""
Stage 3, Step 1 (revised): Load the REAL SNAP ego-Facebook graph.

The user manually downloaded facebook_combined.txt (SNAP: "Social circles:
Facebook", McAuley & Leskovec, NIPS 2012) and supplied it directly, so we
no longer need the Barabasi-Albert synthetic stand-in from before -- we
verify this real file loads correctly and matches the dataset's published
statistics (4,039 nodes; 88,234 edges) before building anything on it.
"""
import networkx as nx
import json

EDGELIST_PATH = "../data/facebook_combined.txt"
EXPECTED_NODES = 4039
EXPECTED_EDGES = 88234

G = nx.read_edgelist(EDGELIST_PATH, nodetype=int)

stats = {
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "avg_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 3),
    "is_connected": nx.is_connected(G),
    "num_connected_components": nx.number_connected_components(G),
    "self_loops": nx.number_of_selfloops(G),
}
print(json.dumps(stats, indent=2))

# --- verification gate: fail loudly if this doesn't match the published dataset ---
assert stats["n_nodes"] == EXPECTED_NODES, \
    f"Node count mismatch: got {stats['n_nodes']}, expected {EXPECTED_NODES}"
assert stats["n_edges"] == EXPECTED_EDGES, \
    f"Edge count mismatch: got {stats['n_edges']}, expected {EXPECTED_EDGES}"
print(f"\nVERIFIED: matches published SNAP ego-Facebook stats "
      f"({EXPECTED_NODES} nodes, {EXPECTED_EDGES} edges).")

nx.write_gpickle = None  # not using gpickle (deprecated); keep edgelist as canonical form
print("Graph is ready at", EDGELIST_PATH)
