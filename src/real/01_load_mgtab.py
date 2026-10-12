"""
Real-data Step 1: load MGTAB (Shi et al. 2023) and build the social graph
that BOTH SumUp and NAAF route votes over.

Source: data/real/MGTAB/*.pt (downloaded from the official MGTAB Google
Drive; CC BY-NC-ND 4.0 -- the raw files are git-ignored, not redistributed).

Graph choice (decided with the author, see README):
  - flow graph = follower (type 0) + friend (type 1) relations only, made
    undirected, self-loops dropped -- the "real social tie" graph that
    SumUp's threat model is defined on. The other 5 relation types
    (mention/reply/quoted/URL/hashtag) are NOT trust links and are only
    available to the NAAF classifier later as extra signal.
  - keep the largest connected component (LCC): a voter outside the
    collector's component can never deliver a vote under ANY method, so
    including them would only add identical zeros to every method.

Node ids are MGTAB's own row indices, so features.pt[i] / labels_bot.pt[i]
index straight into the graph without a remapping table.
"""
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "real" / "MGTAB"
OUT = ROOT / "data" / "real"
SOCIAL_TYPES = [0, 1]  # followers, friends

edge_index = torch.load(RAW / "edge_index.pt", weights_only=True)
edge_type = torch.load(RAW / "edge_type.pt", weights_only=True)
labels_bot = torch.load(RAW / "labels_bot.pt", weights_only=True)  # 0 = human, 1 = bot
n = len(labels_bot)

mask = torch.isin(edge_type, torch.tensor(SOCIAL_TYPES))
src, dst = edge_index[0][mask].tolist(), edge_index[1][mask].tolist()
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from((u, v) for u, v in zip(src, dst) if u != v)
print(f"MGTAB: {n} users, {mask.sum().item()} follower/friend relations "
      f"-> {G.number_of_edges()} undirected social edges")

lcc = max(nx.connected_components(G), key=len)
G = G.subgraph(lcc).copy()
label = {v: int(labels_bot[v]) for v in G.nodes()}
bots = sorted(v for v in G if label[v] == 1)
humans = sorted(v for v in G if label[v] == 0)

edge_mix = Counter(tuple(sorted((label[u], label[v]))) for u, v in G.edges())
n_attack_edges = edge_mix[(0, 1)]
stats = {
    "n_nodes_lcc": G.number_of_nodes(),
    "n_edges_lcc": G.number_of_edges(),
    "n_humans": len(humans),
    "n_bots": len(bots),
    "n_human_human_edges": edge_mix[(0, 0)],
    "n_attack_edges_human_bot": n_attack_edges,
    "n_bot_bot_edges": edge_mix[(1, 1)],
    "attack_edges_per_bot": round(n_attack_edges / len(bots), 2),
    "n_dropped_outside_lcc": n - G.number_of_nodes(),
}
print(json.dumps(stats, indent=2))

nx.write_edgelist(G, OUT / "mgtab_social.edgelist", data=False)
with open(OUT / "mgtab_meta.json", "w") as f:
    json.dump({"social_relation_types": SOCIAL_TYPES, "stats": stats,
               "bot_nodes": bots, "human_nodes": humans}, f)
print(f"Saved {OUT / 'mgtab_social.edgelist'} and {OUT / 'mgtab_meta.json'}")
