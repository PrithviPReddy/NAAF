"""
Synthetic v2, Step 1: fresh Sybil injections on the real SNAP Facebook graph,
one per split seed, for the hybrid-story comparison with MGTAB.

Why v2 (decided with the author): the original Facebook pipeline used ONE
injection per scenario and a classifier trained on honest-only labels
(all-zero targets). v2 reuses the exact same attack model -- inject_sybils()
imported unchanged from src/02_inject_sybils.py, same parameters -- but:
  - injects a FRESH Sybil cluster per seed (0-4), so results average over
    5 different attacks and the sparse scenario's 25 Sybils/run add up to
    25 test Sybils across runs instead of 5;
  - freezes a stratified 70/10/20 train/val/test split per run, exactly as
    on MGTAB (src/real/02_make_splits.py);
  - computes graph-only structural features for the classifier (Facebook
    has no profile/text features): degree, log-degree, local clustering
    coefficient, k-core number, mean neighbour degree -- z-scored.

Scenarios (unchanged from the README): dense = 200 Sybils x 8 attack edges,
sparse = 25 Sybils x 8 attack edges, internal Erdos-Renyi p = 0.3.
Nothing under data/ from the original pipeline is read-modified; outputs go
to data/synthetic_v2/ (git-ignored, regenerable from this script).
"""
import importlib.util
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("inject", ROOT / "src" / "02_inject_sybils.py")
inject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inject)

SCENARIOS = {"dense": 200, "sparse": 25}
ATTACK_EDGES_PER_SYBIL = 8
SEEDS = [0, 1, 2, 3, 4]
OUT = ROOT / "data" / "synthetic_v2"


def structural_features(G, nodes):
    deg = np.array([G.degree(v) for v in nodes], dtype=float)
    clust = nx.clustering(G)
    core = nx.core_number(G)
    nbr = nx.average_neighbor_degree(G)
    X = np.stack([deg, np.log1p(deg), [clust[v] for v in nodes],
                  [core[v] for v in nodes], [nbr[v] for v in nodes]], axis=1)
    return (X - X.mean(0)) / X.std(0)  # unsupervised statistics, no labels used


if __name__ == "__main__":
    G_honest = nx.read_edgelist(ROOT / "data" / "honest_graph.edgelist", nodetype=int)
    for scenario, n_sybil in SCENARIOS.items():
        for seed in SEEDS:
            G, sybils, attack_edges = inject.inject_sybils(G_honest, n_sybil, ATTACK_EDGES_PER_SYBIL,
                                                           internal_p=0.3, seed=seed)
            assert len(attack_edges) == n_sybil * ATTACK_EDGES_PER_SYBIL and nx.is_connected(G)
            nodes = sorted(G.nodes())
            sybil_set = set(sybils)
            y = [1 if v in sybil_set else 0 for v in nodes]
            train, rest, _, y_rest = train_test_split(nodes, y, train_size=0.7, stratify=y, random_state=seed)
            val, test = train_test_split(rest, train_size=1 / 3, stratify=y_rest, random_state=seed)

            d = OUT / f"{scenario}_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            nx.write_edgelist(G, d / "graph.edgelist", data=False)
            np.save(d / "features.npy", structural_features(G, nodes))
            json.dump({"scenario": scenario, "seed": seed, "node_order": "sorted node ids",
                       "bot_nodes": sorted(sybils), "human_nodes": sorted(set(nodes) - sybil_set),
                       "n_attack_edges": len(attack_edges)}, open(d / "meta.json", "w"))
            json.dump({"seed": seed, "train": sorted(train), "val": sorted(val), "test": sorted(test)},
                      open(d / "split.json", "w"))
            n_syb = {k: sum(v in sybil_set for v in part) for k, part in
                     (("train", train), ("val", val), ("test", test))}
            print(f"{scenario} seed {seed}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                  f"{len(attack_edges)} attack edges; Sybils train/val/test = "
                  f"{n_syb['train']}/{n_syb['val']}/{n_syb['test']}", flush=True)
    print(f"Saved to {OUT}")
