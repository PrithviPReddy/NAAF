"""
Real-data Step 5: supervised bot classifier on MGTAB -- the learned signal
NAAF will fuse with SumUp.

Supervised setting (main result): BCE on the TRAIN split's human AND bot
labels. This replaces the Facebook "fair" variant, whose train set held
only honest nodes (all-zero targets) -- see README / project notes.
The honest-seeds-only (PU learning) variant is a later ablation.

Variants, all on the same 9,434-node social LCC, same features:
  - mlp:          features only, no graph (reference point)
  - gatv2_social: GATv2 over follower+friend edges (the SumUp flow graph)
  - gatv2_all:    GATv2 over all 7 MGTAB relations, relation type passed
                  as a one-hot edge feature (edge_dim=7)

Features: MGTAB's own 788-dim features.pt, already normalised by the
MGTAB authors (0/1 flags, standardised counts, text embeddings), used as-is.

Model selection: the epoch with the lowest VAL loss is kept. Only VAL
metrics are computed here -- test labels are never scored in this step.
Per-node P(bot) for every variant/seed is saved for the NAAF steps.

Usage: python 04_classifier_mgtab.py [--variants mlp gatv2_social gatv2_all] [--device cpu|cuda]
gatv2_all (2.8M edges) takes ~30 min/seed on a laptop CPU, so it was run on
a Colab GPU (see colab_gatv2_all.ipynb); mlp and gatv2_social run locally.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "real" / "MGTAB"
OUT = ROOT / "results" / "real" / "classifier"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 300
N_RELATIONS = 7


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index=None, edge_attr=None):
        x = F.dropout(F.elu(self.lin1(x)), p=0.3, training=self.training)
        return self.lin2(x).squeeze(-1)


class BotGATv2(torch.nn.Module):
    def __init__(self, in_dim, hidden=16, heads=4, edge_dim=None):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, 64)  # 788 -> 64 before attention keeps CPU memory sane
        self.conv1 = GATv2Conv(64, hidden, heads=heads, dropout=0.2, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden * heads, 1, heads=1, concat=False, dropout=0.2, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(F.elu(self.proj(x)), p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        return self.conv2(x, edge_index, edge_attr=edge_attr).squeeze(-1)


class BotGATv2Skip(torch.nn.Module):
    """Heterophily-aware variant (real-data Step 6a): bots' neighbours are
    overwhelmingly human, so mixing a node's own features into the
    neighbour average (BotGATv2 above) washes the bot signal out. Here the
    ego embedding bypasses attention, attention aggregates NEIGHBOURS only
    (no self-loops), and the final layer sees both side by side -- the
    ego/neighbour separation of H2GCN (Zhu et al. 2020)."""

    def __init__(self, in_dim, hidden=16, heads=4):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, 64)
        self.conv1 = GATv2Conv(64, hidden, heads=heads, dropout=0.2, add_self_loops=False)
        self.out = torch.nn.Linear(64 + hidden * heads, 1)

    def forward(self, x, edge_index, edge_attr=None):
        ego = F.dropout(F.elu(self.proj(x)), p=0.3, training=self.training)
        neigh = F.elu(self.conv1(ego, edge_index))
        return self.out(torch.cat([ego, neigh], dim=-1)).squeeze(-1)


def load_graphs():
    """Features/labels for the LCC nodes (re-indexed 0..n-1) + both edge sets."""
    meta = json.load(open(ROOT / "data" / "real" / "mgtab_meta.json"))
    nodes = sorted(meta["bot_nodes"] + meta["human_nodes"])  # MGTAB ids
    pos = {v: i for i, v in enumerate(nodes)}
    x = torch.load(RAW / "features.pt", weights_only=True).float()[nodes]
    y = torch.load(RAW / "labels_bot.pt", weights_only=True).float()[nodes]

    old_to_new = torch.full((torch.load(RAW / "labels_bot.pt", weights_only=True).numel(),), -1, dtype=torch.long)
    old_to_new[torch.tensor(nodes)] = torch.arange(len(nodes))
    ei = old_to_new[torch.load(RAW / "edge_index.pt", weights_only=True)]
    et = torch.load(RAW / "edge_type.pt", weights_only=True)
    keep = (ei[0] >= 0) & (ei[1] >= 0) & (ei[0] != ei[1])
    ei, et = ei[:, keep], et[keep]

    social = to_undirected(ei[:, et <= 1], num_nodes=len(nodes))
    ei_all = torch.cat([ei, ei.flip(0)], dim=1)  # both directions, relation kept
    attr_all = F.one_hot(torch.cat([et, et]), N_RELATIONS).float()
    return nodes, pos, x, y, {"social": (social, None), "all": (ei_all, attr_all)}


def train_one(variant, x, y, graphs, train_idx, val_idx, seed, device, epochs):
    torch.manual_seed(seed)
    if variant == "mlp":
        model, (ei, ea) = MLP(x.shape[1]), (None, None)
    elif variant == "gatv2_social":
        model, (ei, ea) = BotGATv2(x.shape[1]), graphs["social"]
    elif variant == "gatv2_social_skip":
        model, (ei, ea) = BotGATv2Skip(x.shape[1]), graphs["social"]
    else:
        model, (ei, ea) = BotGATv2(x.shape[1], edge_dim=N_RELATIONS), graphs["all"]
    model = model.to(device)
    x, y, train_idx, val_idx = x.to(device), y.to(device), train_idx.to(device), val_idx.to(device)
    ei = ei.to(device) if ei is not None else None
    ea = ea.to(device) if ea is not None else None
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    best = (float("inf"), None, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x, ei, ea)[train_idx], y[train_idx])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, ei, ea)
            val_loss = F.binary_cross_entropy_with_logits(logits[val_idx], y[val_idx]).item()
        if val_loss < best[0]:
            best = (val_loss, torch.sigmoid(logits).cpu(), epoch)
    return best


def val_metrics(p, y, val_idx):
    pv, yv = p[val_idx].numpy(), y[val_idx].numpy()
    return {
        "val_auc": round(float(roc_auc_score(yv, pv)), 4),
        "val_f1_at_0.5": round(float(f1_score(yv, pv > 0.5)), 4),
        "val_p_bot_humans": {q: round(float(np.quantile(pv[yv == 0], qq)), 4)
                             for q, qq in (("median", 0.5), ("p90", 0.9), ("p99", 0.99))},
        "val_p_bot_bots": {q: round(float(np.quantile(pv[yv == 1], qq)), 4)
                           for q, qq in (("p01", 0.01), ("p10", 0.1), ("median", 0.5))},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["mlp", "gatv2_social", "gatv2_all"],
                        choices=["mlp", "gatv2_social", "gatv2_social_skip", "gatv2_all"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    tag = "" if args.epochs == EPOCHS else f"_e{args.epochs}"  # keeps Step 5 outputs untouched

    nodes, pos, x, y, graphs = load_graphs()
    print(f"{len(nodes)} nodes; social edges (directed both ways): {graphs['social'][0].shape[1]}; "
          f"all-relation edges: {graphs['all'][0].shape[1]}; device: {args.device}", flush=True)

    results = {}
    for variant in args.variants:
        per_seed = []
        for seed in SEEDS:
            split = json.load(open(ROOT / "results" / "real" / "splits" / f"split_seed{seed}.json"))
            train_idx = torch.tensor([pos[v] for v in split["train"]])
            val_idx = torch.tensor([pos[v] for v in split["val"]])
            val_loss, probs, epoch = train_one(variant, x, y, graphs, train_idx, val_idx, seed,
                                               args.device, args.epochs)
            m = {"seed": seed, "best_epoch": epoch, "val_loss": round(val_loss, 4), **val_metrics(probs, y, val_idx)}
            per_seed.append(m)
            print(f"{variant:>13} seed {seed}: epoch {epoch:>3}  val AUC {m['val_auc']:.4f}  "
                  f"F1 {m['val_f1_at_0.5']:.4f}  humans p99 P(bot) {m['val_p_bot_humans']['p99']:.3f}  "
                  f"bots median P(bot) {m['val_p_bot_bots']['median']:.3f}", flush=True)
            np.save(OUT / f"p_bot_{variant}{tag}_seed{seed}.npy", probs.numpy())
        results[variant] = {
            "val_auc": {"mean": round(statistics.mean(r["val_auc"] for r in per_seed), 4),
                        "std": round(statistics.stdev(r["val_auc"] for r in per_seed), 4)},
            "val_f1_at_0.5": {"mean": round(statistics.mean(r["val_f1_at_0.5"] for r in per_seed), 4),
                              "std": round(statistics.stdev(r["val_f1_at_0.5"] for r in per_seed), 4)},
            "per_seed": per_seed,
        }
        json.dump({"variant": variant, "epochs": args.epochs, "device": args.device, "torch": torch.__version__,
                   "node_order": "sorted MGTAB ids of the social LCC (see mgtab_meta.json)", **results[variant]},
                  open(OUT / f"classifier_val_{variant}{tag}.json", "w"), indent=2)

    print("\nVal summary (mean +/- std over 5 splits):")
    for v, r in results.items():
        print(f"  {v:>13}: AUC {r['val_auc']['mean']:.4f} +/- {r['val_auc']['std']:.4f}   "
              f"F1 {r['val_f1_at_0.5']['mean']:.4f} +/- {r['val_f1_at_0.5']['std']:.4f}")
    print(f"Saved {OUT}")
