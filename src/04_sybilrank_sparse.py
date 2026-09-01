"""
Sparse-attack scenario: SybilRank baseline, rerun on the sparse Sybil
injection (25 Sybils, 200 attack edges) instead of the dense one.

Reuses the EXACT power-iteration implementation from
04_sybilrank_baseline.py (the `sybilrank` function) unmodified -- same
K=100 trusted seeds, same early-termination at ceil(log2(n)) rounds,
same degree-normalization. Only the input graph/labels, output path,
and the resulting node count (n is now 4,064 instead of 4,239) differ,
so the iteration count is recomputed from the new n rather than copied.

Does not read or write anything under the dense scenario's file names.
"""
import networkx as nx
import numpy as np
import json
import random
from importlib import import_module
from sklearn.metrics import roc_auc_score

sybilrank_mod = import_module("04_sybilrank_baseline")
sybilrank = sybilrank_mod.sybilrank

if __name__ == "__main__":
    G = nx.read_edgelist("../data/combined_graph_sparse.edgelist", nodetype=int,
                          data=(("edge_type", str),))
    with open("../data/sybil_labels_sparse.json") as f:
        labels = json.load(f)
    sybil_nodes = set(labels["sybil_nodes"])
    honest_nodes = [n for n in G.nodes() if n not in sybil_nodes]

    rng = random.Random(42)
    K = 100
    seeds = rng.sample(honest_nodes, K)

    trust_scores, iters = sybilrank(G, seeds)
    print(f"Ran {iters} early-terminated power-iteration rounds with {K} seeds")

    y_true = []
    y_score = []
    for node, trust in trust_scores.items():
        y_true.append(1 if node in sybil_nodes else 0)
        y_score.append(trust)

    auc = roc_auc_score(y_true, y_score)

    nodes = list(G.nodes())
    degree = {n: G.degree(n) for n in nodes}
    raw_trust = {n: trust_scores[n] * degree[n] for n in nodes}
    y_score_raw = [raw_trust[n] for n, _ in trust_scores.items()]
    auc_raw = roc_auc_score(y_true, y_score_raw)

    ranked = sorted(trust_scores.items(), key=lambda kv: -kv[1])
    top_n = len(sybil_nodes)  # top-N by trust, N = actual Sybil count (25 here, not hardcoded 200)
    top_by_trust = [n for n, _ in ranked[:top_n]]
    top_sybil_frac = sum(1 for n in top_by_trust if n in sybil_nodes) / top_n

    results = {
        "scenario": "sparse",
        "n_sybil_nodes": len(sybil_nodes),
        "n_seeds": K,
        "iterations": iters,
        "roc_auc_degree_normalized": round(auc, 4),
        "roc_auc_raw_trust_pre_normalization": round(auc_raw, 4),
        f"top_{top_n}_highest_trust_sybil_fraction": round(top_sybil_frac, 4),
    }
    print(json.dumps(results, indent=2))

    with open("../results/sybilrank_sparse.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved ../results/sybilrank_sparse.json")
