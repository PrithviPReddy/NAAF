# First 40 lines of README.md (of 344). Copied unchanged.
# Note: this README describes the earlier SNAP-Facebook pipeline (src/01-09); the MGTAB work lives in src/real/ and is summarised in NAAF_OUTCOMES.md.

# NAAF Implementation — BCSE497J Project-I

Working implementation for "Neural-Attentive Capacity Assignment: A Hybrid
Paradigm for Sybil-Resilient Governance in Decentralized Information Systems"
(Review 2, Sept 2026).

**Reproducing the results on a fresh machine?** See [`REPRODUCE.md`](REPRODUCE.md) for the
complete step-by-step runbook (setup, folder scaffold, exact commands, expected outputs,
and configuration parameters for every pipeline stage). This file below is the running
narrative/progress log — the "why" behind each step, not the "how to rerun it."

This folder is synced from the cloud workspace after each completed step, so
it should always reflect exactly what's been built and tested so far — safe
to open in VS Code and run locally (see Setup below).

## Structure

```
NAAF_Implementation/
├── README.md              <- this file, updated after every stage
├── requirements.txt        <- Python dependencies
├── data/
│   ├── facebook_combined.txt      <- raw SNAP ego-Facebook edge list (source of truth)
│   ├── honest_graph.edgelist      <- same data, canonical name used by all scripts
│   ├── combined_graph.edgelist    <- DENSE scenario: honest graph + Sybil cluster (200 sybils), edge_type tagged
│   ├── sybil_labels.json          <- DENSE scenario ground truth: sybil node ids + attack edge list
│   ├── edge_attention.json        <- DENSE scenario: learned per-edge attention weights (Step 5)
│   ├── combined_graph_sparse.edgelist  <- SPARSE scenario: honest graph + Sybil cluster (25 sybils)
│   ├── sybil_labels_sparse.json        <- SPARSE scenario ground truth
│   └── edge_attention_sparse.json      <- SPARSE scenario: learned per-edge attention weights
├── src/
│   ├── 01_build_graph.py          <- Stage 3 Step 1: load + verify the real graph
│   ├── 02_inject_sybils.py        <- Stage 3 Step 2: inject DENSE Sybil cluster (200) + attack edges
│   ├── 02_inject_sybils_sparse.py <- inject SPARSE Sybil cluster (25) + attack edges
│   ├── 03_sumup_baseline.py       <- Step 3: SumUp, dense scenario
│   ├── 03_sumup_sparse.py         <- SumUp, sparse scenario
│   ├── 04_sybilrank_baseline.py   <- Step 4: SybilRank, dense scenario
│   ├── 04_sybilrank_sparse.py     <- SybilRank, sparse scenario
│   ├── 05_gatv2_layer.py          <- Step 5: GATv2 layer, dense scenario
│   ├── 05_gatv2_sparse.py         <- GATv2 layer, sparse scenario (model retrained from scratch)
