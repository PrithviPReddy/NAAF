"""
Sparse-attack scenario: GATv2 attention layer, retrained from scratch on
the sparse Sybil injection (25 Sybils, 200 attack edges).

Reuses the exact NaafGATv2 model class (architecture, hidden width,
heads) and training procedure from 05_gatv2_layer.py unchanged -- same
2-layer GATv2Conv(hidden=8, heads=4), same 2 input features
[normalized degree, revealed-label-with-0.5-unknown], same 20%-revealed
semi-supervised split, same 100 training epochs, same bug fixes (node
order via list(G.nodes()) matching from_networkx; self-loop slicing on
the attention edge index). The `run()` logic here is a parametrized
copy (graph/labels paths are arguments instead of hardcoded) so the
dense scenario's script and outputs are never touched or re-executed.

A NEW model is trained here (gatv2_model_sparse.pt) rather than reusing
the dense model, because the sparse graph has different nodes (4,064 vs
4,239) and a different attack-edge distribution -- the dense model's
weights are not meaningful on this graph.
"""
import networkx as nx
import numpy as np
import json
import random
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from sklearn.metrics import roc_auc_score
from importlib import import_module

gat = import_module("05_gatv2_layer")
NaafGATv2 = gat.NaafGATv2

torch.manual_seed(42)


def run(graph_path, labels_path, reveal_sybil_labels, tag):
    G = nx.read_edgelist(graph_path, nodetype=int, data=(("edge_type", str),))
    with open(labels_path) as f:
        labels_data = json.load(f)
    sybil_nodes = set(labels_data["sybil_nodes"])

    data = from_networkx(G)
    nodes = list(G.nodes())
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
    GRAPH_PATH = "../data/combined_graph_sparse.edgelist"
    LABELS_PATH = "../data/sybil_labels_sparse.json"

    fair_result, _ = run(GRAPH_PATH, LABELS_PATH, reveal_sybil_labels=False,
                          tag="sparse: honest-seeds-only (fair vs SybilRank)")

    upper_result, (nodes, edge_attn, edge_index, model, X) = run(
        GRAPH_PATH, LABELS_PATH, reveal_sybil_labels=True,
        tag="sparse: honest+sybil-seeds (upper bound, documented separately)")

    with open("../results/gatv2_layer_sparse.json", "w") as f:
        json.dump({"scenario": "sparse",
                   "fair_honest_seeds_only": fair_result,
                   "upper_bound_with_sybil_seeds": upper_result}, f, indent=2)

    torch.save(model.state_dict(), "../results/gatv2_model_sparse.pt")
    edge_attn_out = {}
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for i in range(len(src)):
        u, v = nodes[src[i]], nodes[dst[i]]
        edge_attn_out[f"{u},{v}"] = float(edge_attn[i])
    with open("../data/edge_attention_sparse.json", "w") as f:
        json.dump(edge_attn_out, f)
    print("\nSaved ../results/gatv2_layer_sparse.json, gatv2_model_sparse.pt, "
          "../data/edge_attention_sparse.json")
