# Graph Report - Naaf-Implementation  (2026-10-12)

## Corpus Check
- Large corpus: 37 files · ~933,910 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 130 nodes · 239 edges · 8 communities (7 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.87)
- Token cost: 64,077 input · 0 output

## Community Hubs (Navigation)
- Pipeline Scripts & Sybil Injection
- NAAF Fusion Concepts & Variants
- SybilRank & GATv2 Models
- Dataset, Paper & Reproducibility
- SumUp Max-Flow Baseline
- Sybil Scenarios & Threat Model
- NAAF Option B Boundary Throttling
- NAAF Fusion Implementation

## God Nodes (most connected - your core abstractions)
1. `Stage-by-Stage Pipeline (Stages 1-9)` - 16 edges
2. `run_scenario()` - 10 edges
3. `NAAF Fusion: capacity_SumUp * (1 - P(Sybil|v))` - 10 edges
4. `Synthetic Sybil Cluster Injection (Erdos-Renyi p=0.3)` - 8 edges
5. `Python Dependencies (requirements.txt)` - 8 edges
6. `run_scenario()` - 7 edges
7. `SumUp (Tran et al. 2009)` - 7 edges
8. `NAAF Option A: Source/Voter-Level Trust Weighting` - 7 edges
9. `GATv2 Attention Layer (NaafGATv2)` - 6 edges
10. `Locked Evaluation Protocol` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Synthetic Sybil Cluster Injection (Erdos-Renyi p=0.3)` --shares_data_with--> `facebook_combined.txt (SNAP ego-Facebook edge list, 88,234 edges)`  [EXTRACTED]
  README.md → data/facebook_combined.txt
- `facebook_combined.txt (SNAP ego-Facebook edge list, 88,234 edges)` --implements--> `SNAP ego-Facebook Graph (McAuley & Leskovec, NIPS 2012)`  [EXTRACTED]
  data/facebook_combined.txt → README.md
- `honest_graph.edgelist Canonicalization (manual copy)` --shares_data_with--> `facebook_combined.txt (SNAP ego-Facebook edge list, 88,234 edges)`  [EXTRACTED]
  REPRODUCE.md → data/facebook_combined.txt
- `Stage-by-Stage Pipeline (Stages 1-9)` --references--> `facebook_combined.txt (SNAP ego-Facebook edge list, 88,234 edges)`  [EXTRACTED]
  REPRODUCE.md → data/facebook_combined.txt
- `run()` --calls--> `NaafGATv2`  [INFERRED]
  src/05_gatv2_sparse.py → src/05_gatv2_layer.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **NAAF capacity-fusion variants (v1, v2, A, B, C, percentile)** — readme_naaf_fusion, readme_naaf_v2_threshold_gated, readme_naaf_option_a, readme_naaf_option_b, readme_naaf_option_c, readme_percentile_rank_scaling [EXTRACTED 1.00]
- **Methods compared under the locked evaluation protocol** — readme_sumup, readme_sybilrank, readme_gatv2_attention_layer, readme_naaf_fusion, readme_locked_evaluation_protocol [EXTRACTED 1.00]
- **Sparse-regime calibration failure mode** — readme_sparse_scenario, readme_class_imbalance_calibration, readme_tau_hi_calibration, readme_naaf_option_b, readme_naaf_v2_threshold_gated [INFERRED 0.85]

## Communities (8 total, 1 thin omitted)

### Community 0 - "Pipeline Scripts & Sybil Injection"
Cohesion: 0.10
Nodes (3): inject_sybils(), percentile_multiplier(), run_scenario()

### Community 1 - "NAAF Fusion Concepts & Variants"
Cohesion: 0.15
Nodes (13): Capacity-Formula Ablation (alpha sweep), Attacker Capacity per Attack Edge, GATv2 Attention Layer (NaafGATv2), Honest Collection Rate, NAAF Fusion: capacity_SumUp * (1 - P(Sybil|v)), NAAF Option A: Source/Voter-Level Trust Weighting, NAAF Option B: Suspicious-Boundary Edge Throttling, NAAF Option C: Confidence-Gated Variant of A (+5 more)

### Community 2 - "SybilRank & GATv2 Models"
Cohesion: 0.15
Nodes (4): sybilrank(), NaafGATv2, run(), run()

### Community 3 - "Dataset, Paper & Reproducibility"
Cohesion: 0.21
Nodes (10): facebook_combined.txt (SNAP ego-Facebook edge list, 88,234 edges), Neural-Attentive Capacity Assignment: A Hybrid Paradigm for Sybil-Resilient Governance in Decentralized Information Systems, NAAF Implementation (BCSE497J Project-I), NAAF Reproducibility Guide, matplotlib>=3.8, networkx>=3.0, numpy>=1.24, Python Dependencies (requirements.txt) (+2 more)

### Community 4 - "SumUp Max-Flow Baseline"
Cohesion: 0.21
Nodes (4): bfs_levels(), build_flow_graph(), ticket_distribution(), votes_collected()

### Community 5 - "Sybil Scenarios & Threat Model"
Cohesion: 0.20
Nodes (8): Attack Edges, Dense Sybil Scenario (200 Sybils, 1,600 attack edges), SNAP ego-Facebook Graph (McAuley & Leskovec, NIPS 2012), Sparse Sybil Scenario (25 Sybils, 200 attack edges), Synthetic Sybil Cluster Injection (Erdos-Renyi p=0.3), SybilRank (Cao et al. 2012), Result File to Reported Claim Mapping, scikit-learn>=1.3

### Community 6 - "NAAF Option B Boundary Throttling"
Cohesion: 0.23
Nodes (6): calibrate_thresholds(), gate(), get_revealed_honest_nodes(), run_scenario(), evaluate_source_weighted(), votes_collected_weighted()

## Knowledge Gaps
- **4 isolated node(s):** `Neural-Attentive Capacity Assignment: A Hybrid Paradigm for Sybil-Resilient Governance in Decentralized Information Systems`, `NAAF v2 Threshold-Gated (P(Sybil) > 0.5)`, `Attack Edges`, `matplotlib>=3.8`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 45 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Stage-by-Stage Pipeline (Stages 1-9)` connect `NAAF Fusion Concepts & Variants` to `Dataset, Paper & Reproducibility`, `Sybil Scenarios & Threat Model`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **What connects `Neural-Attentive Capacity Assignment: A Hybrid Paradigm for Sybil-Resilient Governance in Decentralized Information Systems`, `NAAF v2 Threshold-Gated (P(Sybil) > 0.5)`, `Attack Edges` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Pipeline Scripts & Sybil Injection` be split into smaller, more focused modules?**
  _Cohesion score 0.09885057471264368 - nodes in this community are weakly interconnected._
- **Why does `run_scenario()` connect `NAAF Option B Boundary Throttling` to `Pipeline Scripts & Sybil Injection`, `SumUp Max-Flow Baseline`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `Synthetic Sybil Cluster Injection (Erdos-Renyi p=0.3)` connect `Sybil Scenarios & Threat Model` to `NAAF Fusion Concepts & Variants`, `Dataset, Paper & Reproducibility`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._