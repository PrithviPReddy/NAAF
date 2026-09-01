"""
Stage 3, Step 3: SumUp baseline -- ticket distribution + capacity
assignment + max-flow evaluation.

Faithful to Tran et al. (NSDI 2009):
  - Collector distributes a fixed ticket budget breadth-first.
  - Each node consumes 1 ticket, splits the remainder EVENLY across its
    edges to next-level neighbors ("uniform ticket splitting" -- this is
    exactly the mechanism NAAF will later replace with learned attention).
  - Link capacity c(u,v) = 1 + tickets routed on that link.
  - Tickets never flow backward or across same-level links.

Deliberate simplification (documented, not hidden): we compute EXACT
max-flow with networkx rather than SumUp's greedy O(diameter)
approximation, since our graph (4,239 nodes) is small enough that exact
computation is fast, and it isolates the capacity-ASSIGNMENT policy
(what we're actually varying across SumUp / SybilRank / NAAF later) from
search-algorithm approximation noise.
"""
import networkx as nx
import json
from collections import deque

def bfs_levels(G, collector):
    """Standard BFS distance from collector to every reachable node."""
    return nx.single_source_shortest_path_length(G, collector)


def ticket_distribution(G, levels, collector, c_max):
    """
    Level-order ticket cascade. Returns dict {(u, v): capacity} for every
    edge with level[v] = level[u] + 1 (i.e. every 'forward' edge).
    """
    nodes_by_level = {}
    for node, lvl in levels.items():
        nodes_by_level.setdefault(lvl, []).append(node)
    max_level = max(nodes_by_level)

    tickets_in = {collector: c_max}
    capacity = {}

    for lvl in range(max_level + 1):
        for u in nodes_by_level.get(lvl, []):
            tickets = tickets_in.get(u, 0)
            remaining = max(tickets - 1, 0)  # u consumes exactly 1 ticket
            children = [v for v in G.neighbors(u) if levels.get(v) == lvl + 1]
            if not children:
                continue
            share, leftover = divmod(remaining, len(children))
            for i, v in enumerate(children):
                t = share + (1 if i < leftover else 0)
                capacity[(u, v)] = 1 + t
                tickets_in[v] = tickets_in.get(v, 0) + t
    return capacity


def build_flow_graph(G, levels, capacity):
    """Directed graph: every forward edge (u at level L, v at level L+1)
    becomes a directed edge v -> u (flow moves toward the collector)."""
    F = nx.DiGraph()
    F.add_nodes_from(G.nodes())
    for (u, v), cap in capacity.items():
        F.add_edge(v, u, capacity=cap)
    return F


def votes_collected(flow_graph, collector, voters, cap_per_voter=1):
    """Add a super-source wired to `voters` (capacity cap_per_voter each),
    compute exact max-flow to the collector, return the flow value."""
    F = flow_graph.copy()
    F.add_node("SOURCE")
    reachable_voters = [v for v in voters if v in flow_graph.nodes()]
    for v in reachable_voters:
        F.add_edge("SOURCE", v, capacity=cap_per_voter)
    value, _ = nx.maximum_flow(F, "SOURCE", collector)
    return value, len(reachable_voters)


if __name__ == "__main__":
    G = nx.read_edgelist("../data/combined_graph.edgelist", nodetype=int, data=(("edge_type", str),))
    with open("../data/sybil_labels.json") as f:
        labels = json.load(f)
    sybil_nodes = set(labels["sybil_nodes"])
    n_attack_edges = len(labels["attack_edges"])
    honest_nodes = [n for n in G.nodes() if n not in sybil_nodes]

    # Default collector: node closest to MEDIAN degree among honest nodes.
    # (Diagnostic: the max-degree node has degree 1,045 vs a median of 26 --
    # picking that as collector gives it a one-hop neighborhood covering 26%
    # of the whole graph, which is not a representative "vote collector."
    # We use median-degree as the representative default, and keep the
    # hub case available for a separate dense-region analysis later.)
    degrees = dict(G.degree(honest_nodes))
    median_degree = sorted(degrees.values())[len(degrees) // 2]
    collector = min(honest_nodes, key=lambda n: abs(degrees[n] - median_degree))
    print(f"Collector node: {collector} (degree {G.degree(collector)}, "
          f"median honest degree is {median_degree})")

    levels = bfs_levels(G, collector)
    max_level = max(levels.values())
    level_sizes = {}
    for lvl in levels.values():
        level_sizes[lvl] = level_sizes.get(lvl, 0) + 1
    print(f"BFS depth from collector: {max_level} levels, sizes: {level_sizes}")

    C_MAX = 1000
    capacity = ticket_distribution(G, levels, collector, C_MAX)
    print(f"Ticket distribution assigned capacities to {len(capacity)} forward edges "
          f"(C_max={C_MAX})")

    flow_graph = build_flow_graph(G, levels, capacity)

    attacker_flow, n_sybil_reachable = votes_collected(flow_graph, collector, sybil_nodes)

    # Match honest turnout to the Sybil pool size (200) for a fair head-to-head,
    # and to stay close to the literature's own tested condition ("<1% of users
    # vote per round", not 100% simultaneous turnout -- see note above).
    import random
    rng = random.Random(42)
    honest_voter_sample = rng.sample(honest_nodes, 200)
    honest_flow, n_honest_reachable = votes_collected(flow_graph, collector, honest_voter_sample)

    # Also record the full-turnout (all 4,039 honest nodes) case for transparency --
    # this is a harder stress test than the paper's own claim, kept for discussion.
    honest_flow_full, n_honest_full = votes_collected(flow_graph, collector, honest_nodes)

    results = {
        "collector": collector,
        "C_max": C_MAX,
        "n_attack_edges": n_attack_edges,
        "attacker_flow_collected": attacker_flow,
        "attacker_capacity_per_attack_edge": round(attacker_flow / n_attack_edges, 4),
        "n_honest_voters_sampled": n_honest_reachable,
        "honest_flow_collected": honest_flow,
        "honest_collection_rate": round(honest_flow / n_honest_reachable, 4),
        "honest_collection_rate_full_turnout_4039_voters": round(honest_flow_full / n_honest_full, 4),
    }
    print(json.dumps(results, indent=2))

    with open("../results/sumup_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved ../results/sumup_baseline.json")
