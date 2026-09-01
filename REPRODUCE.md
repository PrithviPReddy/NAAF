# NAAF — Reproducibility Guide

This document is a complete, start-to-finish runbook for reproducing every result in the
NAAF Project-I report and presentation, starting from a **fresh machine with nothing but
`facebook_combined.txt`** (the raw SNAP ego-Facebook edge list). Follow it top to bottom in
order — every stage after Stage 0 depends on files written by an earlier stage.

For the narrative of *why* each step exists and what was learned at each stage, see
`README.md`. This document only covers *how to run it again and get the same numbers*.

---

## 0. Scope: what "reproducible" means here

- Every script fixes `seed=42` (Python `random`, and `torch.manual_seed(42)` where relevant)
  and uses **exact** computation (networkx's exact max-flow, not an approximation), so
  re-running the pipeline on the **same library versions** listed below reproduces the
  `results/*.json` values in this repo **exactly** for the graph-theoretic/flow numbers
  (integers and simple ratios), and to at least 3–4 decimal places for the GATv2/AUC numbers
  (floating-point, so a different PyTorch/BLAS build can shift the last digit or two — the
  qualitative findings and rounded headline numbers reported in the paper/deck are stable
  regardless).
- Nothing here needs a GPU. The GATv2 model is 2 layers, hidden width 8, trained for 100
  epochs on a ~4,000-node graph — each training script finishes in well under a minute on a
  normal laptop CPU.
- Total disk footprint end-to-end (raw data + both scenarios' generated graphs/results) is
  about 25 MB.

---

## 1. Prerequisites

| Requirement | Tested version | Notes |
|---|---|---|
| OS | Linux (also works on macOS / Windows) | No OS-specific code paths |
| Python | 3.11.15 | Any Python ≥ 3.9 should work; 3.11 is what every reported number was generated with |
| Disk | ~25 MB free | |
| GPU | Not required | Pure CPU run |

You need exactly one input file to start: **`facebook_combined.txt`** — the SNAP "Social
circles: Facebook" ego-network dataset (McAuley & Leskovec, NIPS 2012). If you don't already
have it:

```bash
# Either download it directly:
curl -O https://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip facebook_combined.txt.gz
# ...or use the copy already checked into this repo's data/ folder.
```

Expected raw-file stats (verified in Stage 1 below): **4,039 nodes, 88,234 edges**, one
connected component.

---

## 2. Folder scaffold (starting from just `facebook_combined.txt`)

Create the exact directory layout every script expects (all data/result paths in the code
are hardcoded as `../data/...` and `../results/...`, relative to `src/`):

```bash
mkdir -p NAAF_Implementation/{data,results,figures,src}
cd NAAF_Implementation
# place facebook_combined.txt here:
cp /path/to/facebook_combined.txt data/facebook_combined.txt
# place the 15 pipeline scripts here (see the file list in Stage-by-stage section below):
cp /path/to/src/*.py src/
cp /path/to/requirements.txt .
```

Final layout before running anything:

```
NAAF_Implementation/
├── requirements.txt
├── data/
│   └── facebook_combined.txt      <- the only data file that exists at the start
├── results/                        <- empty; every *.json / *.pt lands here
├── figures/                        <- unused by the reproduction pipeline itself
└── src/
    ├── 01_build_graph.py
    ├── 02_inject_sybils.py
    ├── 02_inject_sybils_sparse.py
    ├── 03_sumup_baseline.py
    ├── 03_sumup_sparse.py
    ├── 04_sybilrank_baseline.py
    ├── 04_sybilrank_sparse.py
    ├── 05_gatv2_layer.py
    ├── 05_gatv2_sparse.py
    ├── 06_naaf_fusion.py
    ├── 06_naaf_fusion_sparse.py
    ├── 07_ablation_study.py
    ├── 07_ablation_study_sparse.py
    ├── 08_percentile_experiment.py
    └── 09_naaf_option_b.py
```

**All commands below assume your shell's current directory is `src/`.** Every script uses
relative paths (`../data/...`, `../results/...`); running from anywhere else will raise
`FileNotFoundError`.

---

## 3. Environment setup

```bash
cd NAAF_Implementation
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins loose minimum versions; the exact versions every reported result in
this project was generated with are:

| Package | Minimum (requirements.txt) | Exact version used for reported results |
|---|---|---|
| networkx | >=3.0 | 3.6.1 |
| numpy | >=1.24 | 2.4.4 |
| matplotlib | >=3.8 | 3.10.9 *(only used by the standalone figure-generation scripts, not by the `src/` pipeline itself)* |
| scikit-learn | >=1.3 | 1.8.0 |
| torch | >=2.0 | 2.13.0 |
| torch_geometric | >=2.5 | 2.8.0.post1 |

If `pip install torch torch_geometric` has trouble resolving on your platform, install torch
first from [pytorch.org](https://pytorch.org)'s CPU instructions, then `pip install
torch_geometric` — the CPU wheel is all this project needs; no `torch-scatter`/`torch-sparse`
extension packages are required for the layer used here (`GATv2Conv`).

Verify the install:

```bash
python3 -c "import networkx, numpy, sklearn, torch, torch_geometric; print('OK')"
```

---

## 4. One manual step before Stage 1: canonicalize the raw file

No script performs this copy automatically — `01_build_graph.py` reads
`facebook_combined.txt` directly (for verification only), but every later script reads a
second, identically-named file, **`honest_graph.edgelist`**, which must exist first:

```bash
# from NAAF_Implementation/
cp data/facebook_combined.txt data/honest_graph.edgelist
```

(This is a byte-identical copy — the pipeline just uses a scenario-neutral canonical name
downstream so `02_inject_sybils*.py` doesn't need to know the original filename.)

---

## 5. Stage-by-stage pipeline

Run every command from `src/`. Follow this **exact order** — later stages load files
(model checkpoints, graphs, labels) written by earlier ones and will fail or silently use
stale data if you skip ahead.

### Stage 1 — Verify the raw graph

```bash
python3 01_build_graph.py
```
- **Reads:** `../data/facebook_combined.txt`
- **Writes:** nothing (verification only — prints stats and asserts them against the
  published dataset)
- **Parameters:** `EXPECTED_NODES=4039`, `EXPECTED_EDGES=88234` (hardcoded assertions)
- **Expected output:** `n_nodes: 4039`, `n_edges: 88234`, `avg_degree: 43.691`,
  `is_connected: true`, `num_connected_components: 1`, `self_loops: 0`. Script raises
  `AssertionError` and stops the pipeline here if these don't match — this is intentional,
  it means you don't have the correct SNAP file.

### Stage 2 — Inject synthetic Sybil clusters (dense + sparse scenarios)

```bash
python3 02_inject_sybils.py          # DENSE scenario: 200 Sybils
python3 02_inject_sybils_sparse.py   # SPARSE scenario: 25 Sybils
```
- **Reads:** `../data/honest_graph.edgelist` (both scripts, independently)
- **Writes:**
  - Dense: `../data/combined_graph.edgelist`, `../data/sybil_labels.json`
  - Sparse: `../data/combined_graph_sparse.edgelist`, `../data/sybil_labels_sparse.json`
- **Key parameters** (top of each script's `__main__`):

  | Parameter | Dense | Sparse | Meaning |
  |---|---|---|---|
  | `N_SYBIL` | 200 | 25 | number of injected Sybil nodes |
  | `ATTACK_EDGES_PER_SYBIL` | 8 | 8 | honest↔Sybil edges per Sybil (SybilGAT's standard setting; unchanged across scenarios by design) |
  | `internal_p` | 0.3 | 0.3 | Erdős–Rényi density of the Sybil-Sybil internal subgraph |
  | `seed` | 42 | 42 | |

- **Expected output (dense):** `n_total_nodes: 4239`, `honest_honest_edges: 88234`
  (untouched), `attack_edges: 1600`, `is_connected: true`.
- **Expected output (sparse):** `n_total_nodes: 4064`, `attack_edges: 200`,
  `attack_edge_coverage_of_honest_pop_pct: 4.95`, `is_connected: true`.
- The sparse script does **not** read or modify any dense-scenario file, and vice versa —
  running these two commands in either order gives identical results.

### Stage 3 — SumUp baseline (ticket distribution + exact max-flow)

```bash
python3 03_sumup_baseline.py   # dense
python3 03_sumup_sparse.py     # sparse
```
- **Reads:** `combined_graph[_sparse].edgelist`, `sybil_labels[_sparse].json`
- **Writes:** `../results/sumup_baseline.json`, `../results/sumup_sparse.json`
- **Parameters:**

  | Parameter | Value | Meaning |
  |---|---|---|
  | `C_MAX` | 1000 | ticket budget the collector starts with |
  | collector selection | node closest to **median** honest-node degree | see "Locked evaluation protocol" below |
  | honest voter sample | 200 honest nodes, `random.Random(42).sample(...)` | fixed across every stage from here on |
  | Sybil voters | **all** Sybil nodes in the scenario (200 dense / 25 sparse) | |
  | vote cap per voter | 1 | |

- **Expected output (dense):** collector node `62`; `attacker_capacity_per_attack_edge:
  0.0731`; `honest_collection_rate: 0.43`.
- **Expected output (sparse):** collector node `158`; `attacker_capacity_per_attack_edge:
  0.09`; `honest_collection_rate: 0.215`.
- This is the baseline every later method is compared against — these two numbers
  (`0.0731`/`0.43` dense, `0.09`/`0.215` sparse) recur throughout every downstream result
  file as `sumup_baseline` / `sumup_baseline_recomputed`.

### Stage 4 — SybilRank baseline (trust propagation, evaluated by ROC-AUC)

```bash
python3 04_sybilrank_baseline.py   # dense
python3 04_sybilrank_sparse.py     # sparse
```
- **Reads:** same combined graph + labels as Stage 3
- **Writes:** `../results/sybilrank_baseline.json`, `../results/sybilrank_sparse.json`
- **Parameters:** `K=100` trusted seed nodes (`random.Random(42).sample` of honest nodes),
  `iterations = ceil(log2(n))` (early-terminated power iteration — 13 rounds dense, 12
  sparse, computed from each scenario's own node count, not hardcoded).
- **Expected output (dense):** `roc_auc_degree_normalized: 0.5921`,
  `roc_auc_raw_trust_pre_normalization: 0.8675`.
- **Expected output (sparse):** `roc_auc_degree_normalized: 0.7048`,
  `roc_auc_raw_trust_pre_normalization: 0.3852`.
- This is a detection/ranking metric (AUC), evaluated on a completely different axis from
  SumUp/NAAF's capacity metric — not directly comparable row-for-row.

### Stage 5 — GATv2 attention layer (train the classifier NAAF fuses on)

```bash
python3 05_gatv2_layer.py    # dense — writes gatv2_model.pt
python3 05_gatv2_sparse.py   # sparse — writes gatv2_model_sparse.pt
```
- **Reads:** same combined graph + labels as Stage 3/4
- **Writes:**
  - `../results/gatv2_layer.json` / `gatv2_layer_sparse.json` — metrics for **both**
    variants below
  - `../results/gatv2_model.pt` / `gatv2_model_sparse.pt` — **model checkpoint**
  - `../data/edge_attention.json` / `edge_attention_sparse.json` — per-edge attention
    weights
- **Parameters:** `NaafGATv2`: 2-layer GATv2Conv, hidden width 8, 4 attention heads,
  dropout 0.2, 2 input features (`[normalized degree, revealed label ∈ {0, 0.5, 1}]`),
  Adam optimizer (`lr=0.01`, `weight_decay=5e-4`), 100 training epochs,
  `torch.manual_seed(42)`.
- **Each script's `__main__` trains TWO model variants, in this exact order, in a single
  process:**
  1. **Fair variant** (`reveal_sybil_labels=False`) — only a random 20% of *honest* nodes'
     true labels are shown as input features (the rest see the neutral placeholder `0.5`).
     This is the legitimate, real-world-deployable comparison point against SybilRank.
  2. **Upper-bound variant** (`reveal_sybil_labels=True`) — an additional random 20% of
     *Sybil* nodes' labels are also revealed. Realistic only if a platform already has some
     moderator-confirmed bad accounts.

  **⚠️ Important, non-obvious detail for exact reproduction:** because both variants run in
  the same process and only `torch.manual_seed(42)` is called once (at module import, before
  either `run()` call), the *fair* variant's training consumes RNG draws that affect the
  *upper-bound* variant's random weight initialization. **Only the upper-bound-trained
  model is what gets saved** to `gatv2_model.pt` / `edge_attention.json` (the `model`
  variable is overwritten by the second `run()` call) — reproducing the exact saved
  checkpoint requires running both variants, in this order, in one script execution, exactly
  as `05_gatv2_layer.py`'s `__main__` does. Do not call `run()` for only one variant in
  isolation and expect a bit-identical checkpoint.
- **Expected output (dense):** fair variant `auc_holdout: 0.9142`; upper-bound variant
  `auc_holdout: 1.0`.
- **Expected output (sparse):** fair variant `auc_holdout: 0.8534`; upper-bound variant
  `auc_holdout: 0.8474`.

### Stage 6 — NAAF fusion (the core contribution)

```bash
python3 06_naaf_fusion.py          # dense
python3 06_naaf_fusion_sparse.py   # sparse
```
- **Reads:** combined graph + labels (Stage 2) **and** the model checkpoint from Stage 5
  (`../results/gatv2_model.pt` / `gatv2_model_sparse.pt` — **must already exist**)
- **Writes:** `../results/naaf_fusion.json`, `../results/naaf_fusion_sparse.json`
- **Formula:** `capacity_NAAF(u, v) = capacity_SumUp(u, v) * (1 − P(Sybil | v))`, computed
  on top of the *exact same* ticket-distribution capacities from Stage 3 (same collector,
  same `C_MAX=1000`).
- **⚠️ Non-obvious detail:** `get_sybil_probabilities()` in this script re-derives the
  **fair-variant** input features (only 20% honest labels revealed, same seed/logic as
  Stage 5) and feeds them through the **upper-bound-trained** checkpoint saved by Stage 5.
  This exact (fair features → upper-bound weights) combination is what every downstream
  P(Sybil) value in this project is built on — it is how the shipped code behaves, not a
  bug to fix before reproducing.
- **Two capacity variants evaluated:**
  - `naaf_v1_continuous`: every edge scaled by `(1 − P(Sybil|v))`
  - `naaf_v2_threshold_gated`: only scaled when `P(Sybil|v) > 0.5` (`THRESHOLD=0.5`)
- **Expected output (dense):** SumUp recomputed `0.0731`/`0.43` (sanity check, must match
  Stage 3 exactly); NAAF v1 `attacker_capacity_per_attack_edge: 0.0338`,
  `honest_collection_rate: 0.2563` → **53.76% attacker-capacity reduction, −17.37pp honest
  cost**.
- **Expected output (sparse):** NAAF v1 `attacker_capacity_per_attack_edge: 0.0896`,
  `honest_collection_rate: 0.2139` → only **0.44% attacker-capacity reduction** — this is
  the sparse-regime failure mode investigated in Stages 7–9.

### Stage 7 — Ablation study on the capacity-scaling formula (diagnostic)

```bash
python3 07_ablation_study.py          # dense
python3 07_ablation_study_sparse.py   # sparse
```
- **Reads:** everything Stage 6 reads, **plus** imports `06_naaf_fusion[_sparse]` as a
  module (so it also needs the Stage 5 model checkpoint transitively)
- **Writes:** `../results/ablation_study.json`, `../results/ablation_study_sparse.json`
  (diagnostic only — never touches Stage 3/6's locked result files)
- **Parameters swept:** `C = C_sumup * (1 − alpha * P_sybil)` for
  `alpha ∈ {0.25, 0.5, 0.75, 1.0}` (dense); sparse additionally sweeps
  `alpha ∈ {5.0, 20.0, 50.0}` and adds a `5_percentile_rank_scaling` variant (percentile
  rank of P(Sybil) instead of the raw value).
- **Expected output (dense):** roughly linear trade-off, e.g. `alpha=0.25` →
  `-6.7%` attacker / `-4.26pp` honest; `alpha=1.0` reproduces Stage 6's `-53.76%`/`-17.37pp`
  exactly.
- **Expected output (sparse):** even `alpha=50.0` only reaches `-26.11%` attacker reduction;
  `5_percentile_rank_scaling` reaches `-56.67%` at `-12.66pp` honest cost — the key evidence
  that percentile-rank scaling, not amplitude scaling, is what fixes the sparse regime.

### Stage 8 — Percentile-rescaling experiment (dense vs. sparse, side by side)

```bash
python3 08_percentile_experiment.py
```
(Single script, no `_sparse` counterpart — it runs both scenarios internally.)
- **Reads:** both combined graphs/labels and **both** Stage 5 model checkpoints
- **Writes:** `../results/percentile_experiment_dense.json`,
  `../results/percentile_experiment_sparse.json` (additive only — never opens any earlier
  result file in write mode)
- **Formula:** `percentile(v) = rank(v) / (N−1)` (0-indexed ascending rank of P(Sybil) among
  all N nodes in the scenario); `multiplier(v) = 1 − percentile(v)`.
- **Expected output (dense):** percentile scaling → `93.43%` attacker reduction but
  `-33.49pp` honest cost (overcorrects — worse trade-off than raw `(1−P)` scaling here).
- **Expected output (sparse):** percentile scaling → `56.67%` attacker reduction at
  `-12.66pp` honest cost (fixes the near-0% no-op from Stage 6).

### Stage 9 — NAAF Option A / B / C (final fusion variants)

```bash
python3 09_naaf_option_b.py
```
(Single script; runs dense and sparse internally, same as Stage 8.)
- **Reads:** both combined graphs/labels and both Stage 5 model checkpoints
- **Writes:** `../results/naaf_option_b_dense.json`, `../results/naaf_option_b_sparse.json`
- **Parameters:** `q_hi=0.99`, `q_lo=0.90` — the boundary-detection threshold `tau_hi` is
  calibrated **per scenario** as the 99th percentile of P(Sybil) among the same 807 (dense)
  / 807 (sparse) revealed-honest nodes from Stage 5's fair variant — **zero Sybil ground
  truth is used to set the threshold** (ground truth is only used afterward, to *evaluate*
  the resulting boundary detector's precision/recall as a diagnostic).
- **Three formulas evaluated**, all reusing Stage 3's untouched SumUp capacities as the base:
  - **Option A** (source/voter-level trust weighting):
    `capacity(SOURCE→voter) = 1 × (1 − P(Sybil|voter))`, every intermediate edge untouched.
  - **Option B** (boundary/attack-edge throttling — primary formulation going in):
    `capacity(u,v) = capacity_SumUp(u,v) × (1 − P(Sybil|v))` only if `v` is in the
    confidently-flagged set `S_hat = {v : P(Sybil|v) > tau_hi}`, else untouched.
  - **Option C** (conservative, smooth-gated variant of A): same as A but scaled by
    `g(p) = clip((p − tau_lo)/(tau_hi − tau_lo), 0, 1)` instead of `(1−p)` directly.
- **Expected output (dense):** `tau_hi=0.151922`, 272 nodes flagged. **Option A is the
  headline result: 61.42% attacker-capacity reduction at only −0.06pp honest cost**
  (`naaf_a_source_weighted_ablation`). Option B ≈ Option v1's honest damage (−17.35pp).
  Option C: 100% attacker suppression at −0.31pp honest cost.
- **Expected output (sparse):** `tau_hi=0.018488`, 61 nodes flagged. Option B is an exact
  no-op (0.00% change). Option A: 0.89% reduction, −0.07pp honest. None of A/B/C solve the
  sparse regime — this is the project's stated open problem.

---

## 6. Run everything in one shot

Once your environment and folder scaffold from Sections 2–4 are in place:

```bash
cd NAAF_Implementation/src
python3 01_build_graph.py
python3 02_inject_sybils.py
python3 02_inject_sybils_sparse.py
python3 03_sumup_baseline.py
python3 03_sumup_sparse.py
python3 04_sybilrank_baseline.py
python3 04_sybilrank_sparse.py
python3 05_gatv2_layer.py
python3 05_gatv2_sparse.py
python3 06_naaf_fusion.py
python3 06_naaf_fusion_sparse.py
python3 07_ablation_study.py
python3 07_ablation_study_sparse.py
python3 08_percentile_experiment.py
python3 09_naaf_option_b.py
echo "Pipeline complete. Compare ../results/*.json against the shipped copies."
```

Total pipeline runtime on a normal laptop CPU: a few minutes, dominated by the two GATv2
training scripts (Stage 5) and the max-flow calls (which run once per capacity variant per
scenario, each on a ~4,000–4,200-node graph).

---

## 7. Locked evaluation protocol (applies identically to every stage from Stage 3 onward)

To keep every method — SumUp, SybilRank, GATv2, NAAF v1/v2/A/B/C, percentile scaling —
comparable, one protocol is fixed once (in Stage 3) and reused verbatim everywhere after:

| Element | Rule |
|---|---|
| Collector node | The honest node whose degree is closest to the **median** honest-node degree (not the max-degree hub — see README for why the hub was rejected as unrepresentative) |
| Ticket budget | `C_max = 1000` |
| Honest voter sample | 200 honest nodes, `random.Random(42).sample(honest_nodes, 200)` |
| Sybil voters | **all** Sybil nodes in the scenario vote (200 dense / 25 sparse) |
| Vote cap per voter | 1 |
| Flow computation | Exact max-flow (`networkx.maximum_flow`), not SumUp's own greedy approximation — a documented simplification, justified because the graph is small enough to compute exactly |

Collector nodes differ between scenarios because honest-node degrees shift slightly once a
different number of attack edges are attached: **node 62** (dense), **node 158** (sparse).

---

## 8. Which result file backs which reported number

| Report/PPT claim | Source file | Field |
|---|---|---|
| SNAP graph verified: 4,039 nodes / 88,234 edges | Stage 1 console output | `n_nodes`, `n_edges` |
| SumUp baseline: 0.0731 attacker cap/edge, 43.0% honest (dense) | `results/sumup_baseline.json` | `attacker_capacity_per_attack_edge`, `honest_collection_rate` |
| SybilRank AUC 0.592 (dense) / 0.705 (sparse) | `results/sybilrank_baseline.json`, `sybilrank_sparse.json` | `roc_auc_degree_normalized` |
| GATv2 AUC 0.914 dense / 0.853 sparse (fair variant) | `results/gatv2_layer.json`, `gatv2_layer_sparse.json` | `fair_honest_seeds_only.auc_holdout` |
| NAAF-v1: 53.76% attacker reduction, −17.37pp honest (dense) | `results/naaf_fusion.json` | `improvement_v1_vs_sumup` |
| NAAF-v1: 0.44% attacker reduction (sparse — the discovered failure) | `results/naaf_fusion_sparse.json` | `improvement_v1_vs_sumup` |
| Percentile rescaling: 56.67% (sparse) / 93.43% (dense, overcorrects) | `results/percentile_experiment_sparse.json`, `percentile_experiment_dense.json` | `naaf_percentile_vs_sumup` |
| **NAAF-A headline: 61.4% attacker reduction, −0.06pp honest (dense, best observed)** | `results/naaf_option_b_dense.json` | `naaf_a_vs_sumup` |
| NAAF-A on sparse: 0.89% reduction (open problem, unsolved) | `results/naaf_option_b_sparse.json` | `naaf_a_vs_sumup` |
| Ablation alpha-sweep trade-off curve | `results/ablation_study.json`, `ablation_study_sparse.json` | every `4_*` key |

---

## 9. Gotchas / non-obvious dependencies

- **Run from `src/`.** Every path is relative (`../data/...`, `../results/...`); running a
  script from the repo root or anywhere else raises `FileNotFoundError`.
- **`results/` and `data/` must already exist.** No script calls `os.makedirs` — if you
  deleted these folders, `mkdir -p ../data ../results` before rerunning.
- **Numeric-prefixed module imports are intentional.** Several scripts do
  `import_module("03_sumup_baseline")` etc. — this is valid Python (dynamic import doesn't
  require a syntactically-legal identifier the way a plain `import` statement would) and
  depends only on the `.py` file being present in the working directory / `sys.path`; it
  does **not** re-execute that file's `if __name__ == "__main__":` block, so importing
  `03_sumup_baseline` for its functions does not require `results/sumup_baseline.json` to
  already exist — only the *model checkpoints* (`.pt` files) are genuine cross-stage file
  dependencies (see the dependency list below).
- **Real inter-stage file dependencies** (beyond "the script file itself must be present"):
  - Stage 6, 7, 8, 9 all require `results/gatv2_model.pt` (dense) and/or
    `results/gatv2_model_sparse.pt` (sparse) to already exist — i.e., **Stage 5 must run
    before any of Stages 6–9**.
  - Everything from Stage 3 onward requires Stage 2's `combined_graph[_sparse].edgelist`
    and `sybil_labels[_sparse].json`.
- **Dense and sparse scenarios never share files.** Every script in the sparse family reads
  and writes only `*_sparse.*`-suffixed paths; this was verified in the original run via
  byte-for-byte (md5) comparison of every dense-scenario file before/after each sparse run.
  You can therefore run the two scenario tracks (dense: 02→03→04→05→06→07, sparse: the
  `_sparse` counterparts) in either order or interleaved — the only hard ordering constraint
  is *within* each track, plus Stage 8/9 needing both tracks' Stage-5 checkpoints.
- **`gatv2_model.pt` is the upper-bound-trained checkpoint, not the fair one** — see the
  callout in Stage 5. This is exactly what the shipped `results/*.json` files reflect;
  reproducing them requires preserving this behavior, not "fixing" it into two separate
  checkpoints.
- **torch/numpy version drift** can shift GATv2's AUC and P(Sybil) values in the 3rd–4th
  decimal place (different BLAS/LAPACK builds round differently). The exact-integer flow
  numbers (`attacker_flow_collected`, `honest_flow_collected` when they land on whole
  numbers) and simple ratios are unaffected — those come from `networkx`'s exact max-flow on
  the same graph, not the ML component.

---

## 10. Verifying your reproduction

After running the full pipeline, compare your fresh `results/*.json` files against the
copies already checked into this repo:

```bash
cd NAAF_Implementation
for f in results/*.json; do
  echo "=== $f ==="
  diff <(python3 -m json.tool "$f") <(python3 -m json.tool "results_reference/$(basename "$f")") \
    && echo "MATCH" || echo "DIFFERS (see above)"
done
```

(Copy the shipped `results/` folder to `results_reference/` first if you want a live
side-by-side diff while regenerating into a fresh `results/`.) Expect an exact match on
every integer/flow field, and matches to at least 3 decimal places on every AUC/probability
field, per the version-drift note above.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AssertionError` in Stage 1 | Wrong or corrupted `facebook_combined.txt` | Re-download from SNAP; must be exactly 4,039 nodes / 88,234 edges |
| `FileNotFoundError: ../data/honest_graph.edgelist` | Skipped the manual copy in Section 4 | `cp data/facebook_combined.txt data/honest_graph.edgelist` |
| `FileNotFoundError: ../results/gatv2_model.pt` (in Stage 6/7/8/9) | Ran a later stage before Stage 5 | Run `05_gatv2_layer.py` (and/or `05_gatv2_sparse.py`) first |
| `ModuleNotFoundError` on an `import_module("0N_...")` call | Running from the wrong directory | `cd src/` before running anything |
| `pip install torch_geometric` fails to resolve | Platform-specific wheel resolution issue | Install `torch` first from pytorch.org's CPU instructions, then `pip install torch_geometric` separately |
| Slightly different AUC values (3rd+ decimal) | Different torch/numpy/BLAS build | Expected — see version-drift note in Section 9; qualitative results and rounded headline numbers are unaffected |
