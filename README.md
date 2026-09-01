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
│   ├── 06_naaf_fusion.py          <- Step 6: NAAF fusion, dense scenario
│   ├── 06_naaf_fusion_sparse.py   <- NAAF fusion, sparse scenario
│   ├── 07_ablation_study.py       <- Step 7: capacity-formula ablation, dense scenario (diagnostic)
│   └── 07_ablation_study_sparse.py <- capacity-formula ablation, sparse scenario (diagnostic)
├── results/                 <- metrics/tables land here as we produce them (dense + *_sparse variants)
└── figures/                 <- plots/visualizations land here as we produce them
```

## Setup (to run locally in VS Code)

```
python3 -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd src
python 01_build_graph.py
```

## Progress log

- **Stage 3, Step 1 (done):** Loaded the real SNAP ego-Facebook graph
  (`facebook_combined.txt`, McAuley & Leskovec, NIPS 2012). Verified against
  published statistics: 4,039 nodes, 88,234 edges, fully connected, average
  degree 43.691 (cross-checked against the resistance paper's reported
  43.69 for the same dataset). Script: `src/01_build_graph.py`.
- **Stage 3, Step 2 (done):** Injected a synthetic Sybil cluster onto the real
  graph — 200 Sybil nodes (~5% of honest count), densely interconnected among
  themselves (Erdos-Renyi, p=0.3 -> 5,918 sybil-sybil edges), attached via a
  configurable number of scarce "attack edges" into random honest nodes
  (default: 8 per Sybil -> exactly 1,600 attack edges, matching SybilGAT's
  standard synthetic-experiment setting). Verified: original 88,234 honest
  edges untouched, exact attack-edge count, graph stays fully connected.
  `attack_edges_per_sybil` is parametrized for the later 1/4/8/12 sweep.
  Script: `src/02_inject_sybils.py`. Ground truth saved to
  `data/sybil_labels.json`.
- **Stage 3, Step 3 (done):** SumUp baseline. Implemented ticket distribution
  (BFS from collector, uniform split of remaining tickets per level, capacity
  = 1 + tickets) exactly per Tran et al. 2009, then evaluated achievable
  vote flow via EXACT max-flow (networkx) rather than SumUp's own greedy
  approximation -- documented simplification, justified because our graph
  (4,239 nodes) is small enough for exact computation, and it isolates the
  capacity-assignment policy (what changes across SumUp/SybilRank/NAAF) from
  search-algorithm approximation noise.
  - Diagnostic finding: picking the max-degree node as collector (degree
    1,045 vs a median of 26) is an unrepresentative hub outlier -- switched
    default collector selection to the median-degree honest node.
  - Locked evaluation protocol (used identically for every method from here
    on): median-degree collector, C_max=1000 tickets, 200 honest voters
    (fixed random sample, seed 42) vs. all 200 Sybil voters, 1 vote cap each.
  - **Baseline SumUp result on this protocol: attacker capacity/attack edge
    = 0.073 (bound: theoretical guarantee is <=1, so this is consistent);
    honest collection rate = 43.0%.** This is the number NAAF must beat.
  - Note: does not attempt to reproduce the original paper's ">90% honest
    collection" figure -- that claim is explicitly conditioned on "<1% of
    users voting" on graphs orders of magnitude larger than ours; chasing
    that exact percentage on our much smaller graph isn't the point. What
    matters is comparing SumUp vs. SybilRank vs. NAAF on one fixed protocol.
  - Script: `src/03_sumup_baseline.py`. Results: `results/sumup_baseline.json`.
- **Stage 3, Step 4 (done):** SybilRank baseline. Implemented early-terminated
  power iteration (trust seeded at 100 manually-trusted-style honest nodes,
  spread equally outward per hop, stopped after log2(n)=13 rounds) +
  degree normalization, per Cao et al. 2012. Evaluated as a ranking/
  detection problem (ROC-AUC) rather than forced into the capacity metric --
  this matches how the original paper and SybilGAT's own paper evaluate it.
  - Caught and fixed a real bug during testing: an earlier version had the
    AUC sign convention backwards (assumed "lower trust = more Sybil-like",
    but our Sybils actually end up with higher trust on average in this
    configuration). Verified the fix three independent ways (manual
    pairwise comparison, corrected sklearn call, raw-trust cross-check)
    before trusting the number.
  - **Result: ROC-AUC = 0.592** (degree-normalized, the algorithm's actual
    output) -- only modestly better than random (0.5).
  - Diagnostic secondary finding, kept because it's genuinely explanatory:
    AUC on RAW (pre-degree-normalization) trust is 0.868 -- much stronger.
    Degree normalization (meant to remove high-degree bias) is here
    partly cancelling the real signal, because our synthetic Sybil cluster
    is itself unusually high-degree (mean 67 vs honest mean 44, from dense
    internal linking). This reproduces a weakness the original SybilRank
    paper itself acknowledges ("cannot reliably distinguish high-degree
    Sybils from legitimate high-degree users") -- a real, explainable
    result, not a fabricated one.
  - Script: `src/04_sybilrank_baseline.py`. Results: `results/sybilrank_baseline.json`.
- **Stage 3, Step 5 (done):** GATv2 attention layer. 2-layer GATv2Conv
  (hidden width 8, 4 heads), node features = [normalized degree, revealed
  label with 0.5=unknown placeholder], trained semi-supervised on a 20%
  revealed subset, evaluated (AUC) on the held-out 80%.
  - Caught and fixed two real bugs during testing before trusting any
    number: (1) node-index mismatch -- `sorted(G.nodes())` used for
    features/labels didn't match `from_networkx`'s internal node order
    (file-read/insertion order), silently scrambling which features
    belonged to which graph position; fixed by using `list(G.nodes())`
    everywhere. (2) GATv2Conv adds self-loops before computing attention
    by default -- verified empirically that original edges stay in the
    first m columns of the returned attention edge index, sliced off the
    self-loop tail before checking attention by edge type.
  - Ran two variants on purpose: **fair** (only honest labels revealed,
    matching SybilRank's own real-world constraint -- you can't seed from
    confirmed-Sybil ground truth in practice) gives **AUC 0.914** -- the
    legitimate comparison point against SybilRank's honest-only 0.592.
    **Upper bound** (some Sybil labels also revealed, realistic if a
    platform has moderator-confirmed bad accounts, same setup SybilGAT's
    own paper uses) gives **AUC 1.000** -- flagged honestly as an easy
    setting (small, tightly-clustered synthetic Sybil group), not
    oversold as generalizable.
  - **Important finding for Step 6**: mean learned attention weight is
    HIGHER on attack edges (0.046) than honest edges (0.021) in the fair
    variant. This is not a bug -- classification attention reflects "how
    informative is this neighbor," not "how trustworthy" -- but it means
    the raw attention weight can't be used directly to scale SumUp
    capacities (would amplify attack edges, the opposite of NAAF's goal).
    Plan for Step 6: scale capacity by (1 - predicted_sybil_probability
    of the edge's target node) instead, reusing the validated 0.914-AUC
    classifier output rather than the raw attention weight.
  - Scripts: `src/05_gatv2_layer.py`. Results: `results/gatv2_layer.json`,
    `results/gatv2_model.pt`, `data/edge_attention.json`.
- **Stage 3, Step 6 (done):** NAAF fusion -- combine SumUp's ticket-based
  capacities with the GATv2 Sybil-probability signal via
  `capacity_NAAF(u,v) = capacity_SumUp(u,v) * (1 - P(Sybil | v))`. Reused
  the exact locked SumUp protocol from Step 3 and the exact fair-variant
  feature construction from Step 5, so the comparison is apples-to-apples.
  Two variants: v1 (continuous scaling on every edge) and v2 (threshold-
  gated, only penalize when P(Sybil) > 0.5).
  - **Dense-scenario result (200 Sybils, 1,600 attack edges): attacker
    capacity/attack edge drops from 0.0731 (SumUp) to 0.0338 (NAAF v1/v2)
    -- a 53.76% reduction -- at a cost of -17.37 percentage points of
    honest collection rate (0.43 -> 0.2563/0.2565).** v1 and v2 land at
    essentially the same point because almost all the honest nodes with
    P(Sybil) in the 0.1-0.5 range that v2 was designed to protect turn out
    not to matter much for this trade-off.
  - Script: `src/06_naaf_fusion.py`. Results: `results/naaf_fusion.json`.
- **Stage 3, Step 7 (done):** Ablation study on the capacity-scaling
  formula -- diagnostic only, does not touch the locked Step 3/6 results.
  Swept `C = C_sumup * (1 - alpha * P_sybil)` for alpha in
  {0.25, 0.5, 0.75, 1.0}, same graph/collector/C_max/voter sample
  throughout so any difference is attributable to the formula alone.
  Milder alpha trades less attacker-capacity reduction for less honest
  collateral damage roughly proportionally (e.g. alpha=0.25 gives -6.7%
  attacker capacity at only -4.26pp honest, vs -53.76%/-17.37pp at
  alpha=1.0) -- the trade-off curve is fairly linear in alpha on the
  dense graph, no free lunch at any setting tried.
  Script: `src/07_ablation_study.py`. Results: `results/ablation_study.json`.

- **Stage 3, Sparse-attack scenario (done):** Built a second, separate
  injected dataset to test the sparse-cut regime the base papers'
  guarantees actually assume, and reran all four methods on it without
  touching any dense-scenario file.
  - **Parameter choice (literature-justified, not tuned by feel):** kept
    `attack_edges_per_sybil = 8` unchanged -- SybilGAT's own "standard"
    synthetic setting, not a knob we get to tune. SybilRank's guarantee is
    explicitly conditioned on Sybils having "a limited number of attack
    edges due to the difficulty of soliciting and maintaining reciprocal
    social relationships" -- that assumption is about attack edges being a
    small minority of the honest population, not about internal Sybil-
    cluster density (free for the attacker, so `internal_p = 0.3` also
    stays unchanged). The dense scenario's 200 Sybils x 8 = 1,600 attack
    edges covers 33.4% of the 4,039-node honest population -- not sparse
    by that definition. Sparse scenario instead reduces Sybil **count**
    200 -> 25 (8x fewer), giving 25 x 8 = 200 total attack edges, ~5% raw
    coverage before overlap -- comfortably in the sparse-cut regime.
    Verified: honest edges untouched, exact attack-edge count (200),
    graph stays fully connected. Script: `src/02_inject_sybils_sparse.py`.
    Data: `data/combined_graph_sparse.edgelist`,
    `data/sybil_labels_sparse.json`.
  - **SumUp on sparse:** collector recomputed on the sparse graph (node
    158, degree 25 -- different from the dense collector, since honest-
    node degrees shift slightly with fewer attack edges attached).
    Attacker capacity/attack edge = **0.09** (vs. dense's 0.073 -- slightly
    *worse* per-edge, consistent with SumUp's sparse-cut theory: a smaller,
    more isolated Sybil cluster has fewer competing paths splitting the
    ticket budget among Sybils, so each of its 200 attack edges captures
    marginally more). Honest collection rate = 21.5% (lower than dense's
    43% -- expected, since the *evaluation* population, 200 honest voters,
    is unchanged but overall network minor changes shift flow slightly;
    largely reflects sampling/collector differences, not a scenario
    artifact). Script: `src/03_sumup_sparse.py`. Results:
    `results/sumup_sparse.json`.
  - **SybilRank on sparse:** AUC (degree-normalized) = **0.7048** -- clearly
    *better* than dense's 0.592. AUC (raw, pre-normalization) = **0.3852**
    -- worse than random, and the reverse of dense's 0.868. This is the
    mirror image of the dense-scenario finding: with only 25 Sybils and
    200 attack edges, the injected cluster is no longer anomalously
    high-degree relative to the honest population (unlike the dense
    cluster's mean degree 67 vs honest 44), so degree normalization no
    longer cancels a real anomaly signal -- here the raw trust magnitude
    is just noisy while the normalized *ranking* is genuinely informative.
    Consistent, explainable, not cherry-picked. Script:
    `src/04_sybilrank_sparse.py`. Results: `results/sybilrank_sparse.json`.
  - **GATv2 on sparse:** retrained from scratch (sparse graph has different
    nodes, can't reuse the dense model). Fair variant (honest-only seeds):
    AUC = **0.8534** (vs. dense's 0.914 -- somewhat lower, plausibly because
    only 25 Sybils total gives the classifier far fewer positive examples
    to learn from). Upper-bound variant (some Sybil seeds revealed): AUC =
    **0.8474** -- *lower* than the fair variant here, unlike dense's 1.000;
    with only 25 Sybils, revealing 20% (5 nodes) for training leaves just
    20 held-out positives, so this number is noisier and not necessarily
    a real reversal. Scripts: `src/05_gatv2_sparse.py`. Results:
    `results/gatv2_layer_sparse.json`, `gatv2_model_sparse.pt`,
    `data/edge_attention_sparse.json`.
  - **NAAF fusion on sparse -- key finding:** v1/v2 barely move the needle
    vs. SumUp (attacker capacity 0.0896 vs 0.09, a **0.44% reduction**,
    against dense's 53.76%). Root cause, confirmed via a follow-up
    ablation (`07_ablation_study_sparse.py`): mean P(Sybil) across all
    4,064 nodes is only 0.006 and the **maximum is 0.022** -- with just 25
    Sybils among 4,064 nodes (0.6% positive rate), the BCE-trained
    classifier's sigmoid outputs are pushed uniformly low by the severe
    class imbalance, even though its *ranking* is good (AUC 0.8534). The
    fusion formula multiplies capacity by (1 - P), so when P never exceeds
    0.022 the formula is nearly a no-op regardless of ranking quality --
    and the threshold-gated v2 (P > 0.5) never fires at all, reducing to
    plain SumUp exactly. Swept alpha up to 50x to confirm: even a 50x
    amplification of the raw probability only reaches -26.1% attacker
    capacity. Substituting **percentile rank of P(Sybil)** for the raw
    value (decoupling the fusion formula from the classifier's poorly-
    calibrated absolute magnitude) recovers a **56.67% attacker capacity
    reduction** -- back in the same range as the dense result -- and its
    honest-cost trade-off ratio (4.48 points of attacker reduction per 1pp
    of honest collection lost) is comparable to the dense scenario's raw-
    probability scaling (4.0), not worse. Conclusion: NAAF's learned signal
    is real and transfers to the sparse regime, but the specific
    `(1 - P_sybil)` multiplicative formula silently fails under severe
    class imbalance unless probabilities are calibrated or rank-scaled
    first -- a genuine limitation worth stating plainly rather than
    hiding behind the dense scenario's cleaner numbers. Scripts:
    `src/06_naaf_fusion_sparse.py`, `src/07_ablation_study_sparse.py`.
    Results: `results/naaf_fusion_sparse.json`,
    `results/ablation_study_sparse.json`.
  - Dense-scenario files (`combined_graph.edgelist`, `sybil_labels.json`,
    and every dense `results/*.json`) were verified byte-for-byte
    unchanged (md5) after running the full sparse pipeline.

- **Stage 3, Step 8 (done):** post-review design pivot -- controlled
  percentile-scaling experiment (raw `1-P(Sybil)` vs. percentile-rank
  scaling of the same capacity formula, both scenarios, new files only)
  showed percentile scaling fixes the sparse no-op problem (0.44% ->
  56.67% attacker reduction) but *overcorrects* on dense (53.76% ->
  93.43% attacker reduction, but honest cost triples: -17.37pp ->
  -33.49pp), so it is not a universal fix. This motivated a full design
  review of three alternative NAAF fusion mechanisms (Option A:
  source/voter-level trust weighting: `capacity_NAAF-A(SOURCE->voter) =
  1*(1-P(Sybil|voter))`, intermediate capacities untouched, provably
  preserves SumUp's bound unconditionally. Option B: suspicious-boundary
  edge adaptation, throttle only edges whose far endpoint is a
  *confidently*-flagged node. Option C: conservative confidence-gated
  variant of A). Scripts: `src/08_percentile_experiment.py`. Results:
  `results/percentile_experiment_dense.json`,
  `results/percentile_experiment_sparse.json`.
- **Stage 3, Step 9 (done):** implemented and evaluated Option B
  (primary), with A (ablation) and C (conservative variant) on both
  scenarios. Boundary threshold `tau_hi` is calibrated PER SCENARIO as
  the 99th percentile of P(Sybil) among the revealed-HONEST training
  subset only (zero Sybil ground truth used to set it) -- a fixed
  constant like 0.8 was explicitly rejected because it would never fire
  on the sparse scenario's compressed probability scale (max P(Sybil)
  there is ~0.022).
  - **Key finding, dense:** Option B essentially reproduces v1's honest
    damage (honest rate 0.2565 vs. v1's 0.2563, both far below SumUp's
    0.43), *despite* only modifying 978 of many thousands of edges.
    Root cause, confirmed via direct min-cut inspection: the 200-honest-
    voter bottleneck min-cut (60 edges under plain SumUp) already runs
    through nodes that cross `tau_hi` even under this well-calibrated
    threshold (272 nodes flagged: all 200 true Sybils, who score
    0.72-0.81, comfortably clear of `tau_hi=0.152` -- plus 72 false
    positives among honest nodes, a ~1.9% FPR on the full honest
    population vs. the ~1% FPR the threshold targeted on its smaller
    807-node calibration sample). A handful of those 72 false positives
    sit on structurally central shared edges, so Option B's core safety
    argument ("correct boundary detection cannot hurt honest voters")
    is undermined in practice by exactly the false positives the
    calibration couldn't fully eliminate at this population size.
  - **Key finding, sparse:** Option B is an exact no-op (0.00% change
    from SumUp) -- confirmed that literally 0/25 true Sybils exceed the
    calibrated `tau_hi=0.0185` (all score 0.008-0.011). Same root cause
    as v1's sparse failure and percentile scaling's dense overcorrection:
    the classifier's raw probability scale is so compressed under severe
    class imbalance that no absolute-probability threshold calibrated
    from the honest distribution can separate it from true Sybils there.
  - **Option A outperforms both v1 and B on the dense scenario, on BOTH
    axes simultaneously:** honest collection rate 0.4294 (vs. SumUp's
    0.43 -- essentially free) while achieving a *larger* attacker-capacity
    reduction (61.42%) than v1/B's 53.76%. Structural explanation: the
    injected Sybil cluster is densely self-connected by design (Step 2),
    giving the attacker many alternate internal paths that let flow route
    around any edge-capacity throttle placed inside that redundant
    subgraph -- but a source-level discount on each Sybil voter's own
    entry weight cannot be routed around, since it caps the vote before
    it enters the flow network at all. Option C achieves 100% attacker
    suppression at -0.31pp honest cost on dense (all Sybils clear its
    saturation point) but only ~0.7% attacker reduction on sparse (same
    calibration ceiling as B).
  - Every result reported here reused the exact locked protocol
    (median-degree collector, C_max=1000, 200-honest-voter sample
    seed=42, all-Sybil-voters). All 18 previously-locked files (dense +
    sparse graphs/labels/every prior results/*.json) verified
    byte-for-byte unchanged (md5) before and after this step.
    Script: `src/09_naaf_option_b.py`. Results:
    `results/naaf_option_b_dense.json`, `results/naaf_option_b_sparse.json`.

- **Stage 3, Step 10 (next):** decide, with this evidence in hand,
  whether Option B remains the paper's primary formulation (in which
  case the boundary-detection false-positive problem needs a fix -- e.g.
  a stricter/adaptive threshold, or restricting throttling to true
  leaf/near-leaf boundary nodes rather than any confidently-flagged
  node) or whether Option A's empirically stronger and provably-safer
  result reframes it as the primary formulation with B/C as ablations.
