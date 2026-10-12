"""
Synthetic v2, Step 4: "vote-stuffing" scenario -- the regime SumUp's
guarantee is actually about (Tran et al. 2009: the attacker's collected
votes are bounded by the number of attack edges, however many Sybil
identities sit behind them).

Why (see README / results/synthetic_v2): in the dense/sparse injections
each Sybil has 8 attack edges but casts 1 vote, so SumUp's per-attack-edge
cap never binds and NAAF reduced to its classifier. Here the attacker
instead creates MANY cheap identities behind FEW attack edges:
  - 500 Sybil identities, internal Erdos-Renyi p = 0.05 (~25 Sybil links
    each, close to the 26 median honest degree -- degree mimicry);
  - 50 attack edges in total: 50 distinct honest users each linked to one
    distinct "gateway" Sybil; the other 450 Sybils have no honest link.
Five fresh injections (seeds 0-4), stratified 70/10/20 split, structural
features -- same layout as 01_make_scenarios.py, so 02_classifier.py and the
shared NAAF code run on it unchanged. Output: data/synthetic_v2/stuffing_seed*/.
"""
import importlib.util
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("scen", ROOT / "src" / "synthetic_v2" / "01_make_scenarios.py")
scen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scen)

N_SYBIL = 500
N_ATTACK_EDGES = 50
INTERNAL_P = 0.05
SEEDS = [0, 1, 2, 3, 4]


def inject_stuffing(G_honest, seed):
    rng = random.Random(seed)
    honest = sorted(G_honest.nodes())
    sybils = list(range(len(honest), len(honest) + N_SYBIL))
    G = nx.Graph()
    G.add_nodes_from(honest)
    G.add_edges_from(G_honest.edges())
    internal = nx.relabel_nodes(nx.erdos_renyi_graph(N_SYBIL, INTERNAL_P, seed=seed),
                                dict(enumerate(sybils)))
    G.add_nodes_from(sybils)
    G.add_edges_from(internal.edges())
    attack_edges = list(zip(rng.sample(honest, N_ATTACK_EDGES), rng.sample(sybils, N_ATTACK_EDGES)))
    G.add_edges_from(attack_edges)
    return G, sybils, attack_edges


if __name__ == "__main__":
    G_honest = nx.read_edgelist(ROOT / "data" / "honest_graph.edgelist", nodetype=int)
    for seed in SEEDS:
        G, sybils, attack_edges = inject_stuffing(G_honest, seed)
        sybil_set = set(sybils)
        n_attack = sum(1 for u, v in G.edges() if (u in sybil_set) != (v in sybil_set))
        assert n_attack == N_ATTACK_EDGES, n_attack
        assert G.number_of_edges() - G_honest.number_of_edges() - n_attack == \
            G.subgraph(sybils).number_of_edges(), "honest edges altered"
        assert nx.is_connected(G), "Sybil region not reachable from the honest graph"

        nodes = sorted(G.nodes())
        y = [1 if v in sybil_set else 0 for v in nodes]
        train, rest, _, y_rest = train_test_split(nodes, y, train_size=0.7, stratify=y, random_state=seed)
        val, test = train_test_split(rest, train_size=1 / 3, stratify=y_rest, random_state=seed)

        d = scen.OUT / f"stuffing_seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        nx.write_edgelist(G, d / "graph.edgelist", data=False)
        np.save(d / "features.npy", scen.structural_features(G, nodes))
        json.dump({"scenario": "stuffing", "seed": seed, "node_order": "sorted node ids",
                   "bot_nodes": sorted(sybils), "human_nodes": sorted(set(nodes) - sybil_set),
                   "n_attack_edges": n_attack, "gateway_sybils": sorted(s for _, s in attack_edges)},
                  open(d / "meta.json", "w"))
        json.dump({"seed": seed, "train": sorted(train), "val": sorted(val), "test": sorted(test)},
                  open(d / "split.json", "w"))
        deg = [G.degree(s) for s in sybils]
        print(f"stuffing seed {seed}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{n_attack} attack edges, Sybil mean degree {np.mean(deg):.1f}; Sybils train/val/test = "
              f"{sum(v in sybil_set for v in train)}/{sum(v in sybil_set for v in val)}/"
              f"{sum(v in sybil_set for v in test)}", flush=True)
