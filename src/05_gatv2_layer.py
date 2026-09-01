"""
Stage 3, Step 5: GATv2 attention layer -- the novel core of NAAF.
(Revised: fixed two real bugs found during testing -- see README.)

Bug 1: node ordering mismatch. from_networkx() internally orders nodes by
G's iteration order (file-read order), NOT sorted(G.nodes()). An earlier
version built features/labels using sorted() order, silently mismatching
every node's features against the wrong position in edge_index. Fixed by
using list(G.nodes()) everywhere, matching from_networkx exactly.

Bug 2: GATv2Conv adds self-loops by default before computing attention.
Verified (empirically, not assumed) that the first m columns of the
returned attn_edge_index match the original edge_index exactly and
self-loops are appended after -- so edge-type sanity checks slice off
the self-loop tail before aligning with data.edge_type.

Two outputs:
  1. Per-node Sybil-likelihood score (AUC-evaluated, comparable to
     SybilRank's 0.592 / 0.868 from Step 4).
  2. Per-edge attention coefficients for Step 6 (NAAF fusion).
"""
import networkx as nx
import numpy as np
import json
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_networkx
from sklearn.metrics import roc_auc_score
from collections import Counter

torch.manual_seed(42)

class NaafGATv2(torch.nn.Module):
    def __init__(self, in_dim=2, hidden=8, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden, heads=heads, dropout=0.2)
        self.conv2 = GATv2Conv(hidden * heads, 1, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x, (attn_edge_index, attn_weights) = self.conv2(
            x, edge_index, return_attention_weights=True)
        return x.squeeze(-1), attn_edge_index, attn_weights.squeeze(-1)


def run(reveal_sybil_labels, tag):
    G = nx.read_edgelist("../data/combined_graph.edgelist", nodetype=int, data=(("edge_type", str),))
    with open("../data/sybil_labels.json") as f:
        labels_data = json.load(f)
    sybil_nodes = set(labels_data["sybil_nodes"])

    data = from_networkx(G)                 # <-- source of truth for node order
    nodes = list(G.nodes())                  # <-- FIX: matches from_networkx exactly
    node_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    true_label = np.array([1.0 if node in sybil_nodes else 0.0 for node in nodes])
    degree = np.array([G.degree(v) for v in nodes], dtype=float)
    norm_degree = (degree - degree.mean()) / degree.std()

    rng = random.Random(42)
    honest_idx = [node_idx[v] for v in nodes if v not in sybil_nodes]
    sybil_idx = [node_idx[v] for v in nodes if v in sybil_nodes]
    revealed = set(rng.sample(honest_idx, int(0.2 * len(honest_idx))))
    if reveal_sybil_labels:
        revealed |= set(rng.sample(sybil_idx, int(0.2 * len(sybil_idx))))

    revealed_label_feature = np.full(n, 0.5)
    for i in revealed:
        revealed_label_feature[i] = true_label[i]

    X = torch.tensor(np.stack([norm_degree, revealed_label_feature], axis=1), dtype=torch.float)
    y = torch.tensor(true_label, dtype=torch.float)
    train_mask = torch.zeros(n, dtype=torch.bool)
    for i in revealed:
        train_mask[i] = True
    eval_mask = ~train_mask

    edge_index = data.edge_index
    model = NaafGATv2()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        logits, _, _ = model(X, edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, attn_edge_index, attn_weights = model(X, edge_index)
        probs = torch.sigmoid(logits).numpy()

    auc_eval = roc_auc_score(y[eval_mask].numpy(), probs[eval_mask])

    # --- attention-by-edge-type sanity check, self-loops sliced off ---
    m = edge_index.shape[1]
    assert torch.equal(attn_edge_index[:, :m], edge_index), "edge order assumption violated"
    real_edge_attn = attn_weights[:m].numpy()
    edge_types = np.array(data.edge_type)
    by_type = {}
    for t in ["honest", "sybil", "attack"]:
        vals = real_edge_attn[edge_types == t]
        by_type[t] = {"mean": round(float(vals.mean()), 5), "n": int(len(vals))}

    print(f"\n=== [{tag}] revealed sybil labels: {reveal_sybil_labels} ===")
    print(f"train/revealed: {train_mask.sum().item()}  held-out: {eval_mask.sum().item()}")
    print(f"final train loss: {loss.item():.4f}")
    print(f"AUC (held-out only): {auc_eval:.4f}")
    print("mean attention by edge type:", json.dumps(by_type))

    return {
        "tag": tag,
        "reveal_sybil_labels": reveal_sybil_labels,
        "n_train_revealed": int(train_mask.sum().item()),
        "n_eval_holdout": int(eval_mask.sum().item()),
        "final_train_loss": round(loss.item(), 4),
        "auc_holdout": round(float(auc_eval), 4),
        "attention_by_edge_type": by_type,
    }, (nodes, real_edge_attn, data.edge_index, model, X)


if __name__ == "__main__":
    # Fair-comparison variant: only HONEST labels revealed (matches SybilRank's
    # own constraint -- in a real deployment you don't have confirmed-Sybil
    # ground truth to train on; that's the whole problem).
    fair_result, _ = run(reveal_sybil_labels=False, tag="honest-seeds-only (fair vs SybilRank)")

    # Upper-bound variant: some confirmed-Sybil labels also revealed -- realistic
    # if a platform has moderation-confirmed bad accounts (SybilGAT's own
    # transductive setup does this too), kept as a documented secondary result.
    upper_result, (nodes, edge_attn, edge_index, model, X) = run(
        reveal_sybil_labels=True, tag="honest+sybil-seeds (upper bound, documented separately)")

    with open("../results/gatv2_layer.json", "w") as f:
        json.dump({"fair_honest_seeds_only": fair_result,
                   "upper_bound_with_sybil_seeds": upper_result}, f, indent=2)

    torch.save(model.state_dict(), "../results/gatv2_model.pt")
    edge_attn_out = {}
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for i in range(len(src)):
        u, v = nodes[src[i]], nodes[dst[i]]
        edge_attn_out[f"{u},{v}"] = float(edge_attn[i])
    with open("../data/edge_attention.json", "w") as f:
        json.dump(edge_attn_out, f)
    print("\nSaved ../results/gatv2_layer.json, gatv2_model.pt, ../data/edge_attention.json")
