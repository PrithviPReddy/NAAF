"""
Synthetic v2, Step 2: supervised Sybil classifier per injected run.

Same models and training loop as the MGTAB classifier (imported from
src/real/04_classifier_mgtab.py, not copied): BotGATv2Skip (the
heterophily-aware GATv2 picked on MGTAB val) as the main signal, and the
features-only MLP as the attention ablation. 600 epochs, epoch with the
lowest VAL loss kept, trained on TRAIN humans AND Sybils (the fix for the
original pipeline's all-negative "fair" training -- see project notes).
Only val metrics are computed. Note: sparse runs have only 2-3 val Sybils,
so per-run val AUC there is very noisy; it is reported, not tuned on.
"""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_undirected

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("clf", ROOT / "src" / "real" / "04_classifier_mgtab.py")
clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf)

DATA = ROOT / "data" / "synthetic_v2"
OUT = ROOT / "results" / "synthetic_v2" / "classifier"
OUT.mkdir(parents=True, exist_ok=True)
SCENARIOS = ["dense", "sparse"]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 600
VARIANTS = {"gatv2_social_skip": "gatv2_social_skip", "mlp": "mlp"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    results = {}
    for scenario in args.scenarios:
        for variant in args.variants:
            aucs = []
            for seed in SEEDS:
                d = DATA / f"{scenario}_seed{seed}"
                meta, split = json.load(open(d / "meta.json")), json.load(open(d / "split.json"))
                G = nx.read_edgelist(d / "graph.edgelist", nodetype=int)
                nodes = sorted(G.nodes())
                pos = {v: i for i, v in enumerate(nodes)}
                bots = set(meta["bot_nodes"])
                x = torch.tensor(np.load(d / "features.npy"), dtype=torch.float)
                y = torch.tensor([1.0 if v in bots else 0.0 for v in nodes])
                ei = torch.tensor([[pos[u], pos[v]] for u, v in G.edges()]).t()
                graphs = {"social": (to_undirected(ei, num_nodes=len(nodes)), None)}
                train_idx = torch.tensor([pos[v] for v in split["train"]])
                val_idx = torch.tensor([pos[v] for v in split["val"]])

                val_loss, probs, epoch = clf.train_one(variant, x, y, graphs, train_idx, val_idx,
                                                       seed, "cpu", EPOCHS)
                auc = roc_auc_score(y[val_idx].numpy(), probs[val_idx].numpy())
                aucs.append(auc)
                p = probs.numpy()
                print(f"{scenario:>6} {variant:>17} seed {seed}: epoch {epoch:>3}  val AUC {auc:.4f}  "
                      f"Sybil median P {np.median(p[[pos[v] for v in bots]]):.3f}  "
                      f"honest p99 P {np.quantile(p[[pos[v] for v in nodes if v not in bots]], 0.99):.3f}",
                      flush=True)
                np.save(OUT / f"p_bot_{scenario}_{variant}_seed{seed}.npy", p)
            results[f"{scenario}_{variant}"] = {"val_auc_mean": round(statistics.mean(aucs), 4),
                                                "val_auc_std": round(statistics.stdev(aucs), 4),
                                                "val_auc_per_seed": [round(a, 4) for a in aucs]}
    json.dump(results, open(OUT / f"classifier_val_{'_'.join(args.scenarios)}.json", "w"), indent=2)
    print("\nVal AUC summary:")
    for k, r in results.items():
        print(f"  {k:>26}: {r['val_auc_mean']:.4f} +/- {r['val_auc_std']:.4f}")
