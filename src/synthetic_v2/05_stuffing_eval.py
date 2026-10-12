"""
Synthetic v2, Step 5: vote-stuffing + classifier-evasion evaluation.

Claim under test (pre-registered in results/synthetic_v2/stuffing/
preregistration.json): at every evasion level f, NAAF accepts no more Sybil
votes than BOTH SumUp and classifier-only.

Evasion model (worst case, model-agnostic): a fraction f of the Sybil
voters "fool the classifier" -- their P(bot) is replaced by the score of a
uniformly random honest user. The classifier itself is trained normally.
Evader sets are nested across f (first k of one fixed per-seed shuffle), so
increasing f only ever adds evaders.

Everything else is the shared protocol (helpers imported from
src/real/05_naaf_mgtab.py): collector = median-degree TRAIN human,
voters = 200 sampled humans + ALL Sybils of the evaluated part,
tau = 0.995 quantile of P(bot) over val humans (fixed in advance, not tuned).
Attack-edge metric here = Sybil votes / the 50 attack edges into the Sybil
region (450 of 500 Sybils have no attack edge of their own, so the
per-voter count used elsewhere does not apply).
"""
import argparse
import importlib.util
import json
import random
import statistics
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("naaf", ROOT / "src" / "real" / "05_naaf_mgtab.py")
naaf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naaf)

DATA = ROOT / "data" / "synthetic_v2"
OUT = ROOT / "results" / "synthetic_v2" / "stuffing"
SEEDS = [0, 1, 2, 3, 4]
Q = 0.995
EVASION = [0.0, 0.25, 0.5, 0.75, 1.0]
C_MAXES = [1000, 5000]
CLASSIFIER = "gatv2_social_skip"


def run_seed(seed, eval_part):
    d = DATA / f"stuffing_seed{seed}"
    meta, split = json.load(open(d / "meta.json")), json.load(open(d / "split.json"))
    G = nx.read_edgelist(d / "graph.edgelist", nodetype=int)
    probs = np.load(ROOT / "results" / "synthetic_v2" / "classifier" / f"p_bot_stuffing_{CLASSIFIER}_seed{seed}.npy")
    p_bot = dict(zip(sorted(G.nodes()), probs.tolist()))
    bots = set(meta["bot_nodes"])
    humans = sorted(meta["human_nodes"])

    honest_voters = random.Random(seed).sample([v for v in split[eval_part] if v not in bots], naaf.N_HONEST_VOTERS)
    bot_voters = [v for v in split[eval_part] if v in bots]
    tau = float(np.quantile([p_bot[v] for v in split["val"] if v not in bots], Q))
    shuffled = bot_voters[:]
    random.Random(1000 + seed).shuffle(shuffled)
    rng_scores = random.Random(2000 + seed)
    stolen_score = {v: p_bot[rng_scores.choice(humans)] for v in shuffled}  # honest score each evader would copy

    collector = naaf.sumup_mgtab.pick_collector(G, [v for v in split["train"] if v not in bots])
    levels = naaf.sumup.bfs_levels(G, collector)
    out = {"seed": seed, "collector": collector, "tau": round(tau, 4), "n_bot_voters": len(bot_voters),
           "n_attack_edges_total": meta["n_attack_edges"], "by_c_max": {}}
    for c_max in C_MAXES:
        flow = naaf.sumup.build_flow_graph(G, levels, naaf.sumup.ticket_distribution(G, levels, collector, c_max))

        def collected(voters, weight):
            return naaf.votes_collected_weighted(flow, collector, {v: weight(v) for v in voters})

        honest = {
            "sumup": collected(honest_voters, lambda v: 1.0),
            "classifier_only": sum(p_bot[v] < tau for v in honest_voters),
            "naaf_gated_hard": collected(honest_voters, lambda v: 0.0 if p_bot[v] >= tau else 1.0),
            "naaf_a": collected(honest_voters, lambda v: 1.0 - p_bot[v]),
        }
        sumup_bot = collected(bot_voters, lambda v: 1.0)
        rows = {}
        for f in EVASION:
            evaders = set(shuffled[:round(f * len(shuffled))])
            p_eff = {v: (stolen_score[v] if v in evaders else p_bot[v]) for v in bot_voters}
            bot = {
                "sumup": sumup_bot,
                "classifier_only": sum(p_eff[v] < tau for v in bot_voters),
                "naaf_gated_hard": collected(bot_voters, lambda v: 0.0 if p_eff[v] >= tau else 1.0),
                "naaf_a": collected(bot_voters, lambda v: 1.0 - p_eff[v]),
            }
            rows[str(f)] = {m: {"bot_votes": round(bot[m], 3),
                                "bot_vote_acceptance_rate": round(bot[m] / len(bot_voters), 4),
                                "attacker_votes_per_attack_edge": round(bot[m] / meta["n_attack_edges"], 4),
                                "honest_collection_rate": round(honest[m] / naaf.N_HONEST_VOTERS, 4)}
                            for m in bot}
        out["by_c_max"][str(c_max)] = rows
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", choices=["val", "test"], default="val")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in SEEDS:
        per_seed.append(run_seed(seed, args.eval))
        print(f"seed {seed} done: {per_seed[-1]['n_bot_voters']} Sybil voters, tau {per_seed[-1]['tau']}", flush=True)

    summary = {}
    for c_max in map(str, C_MAXES):
        summary[c_max] = {}
        for f in map(str, EVASION):
            summary[c_max][f] = {}
            for m in per_seed[0]["by_c_max"][c_max][f]:
                vals = [s["by_c_max"][c_max][f][m] for s in per_seed]
                summary[c_max][f][m] = {k: {"mean": round(statistics.mean(v[k] for v in vals), 4),
                                            "std": round(statistics.stdev(v[k] for v in vals), 4)}
                                        for k in vals[0]}
            s = summary[c_max][f]
            naaf_bot = s["naaf_gated_hard"]["bot_votes"]["mean"]
            s["claim_naaf_le_both"] = bool(naaf_bot <= s["sumup"]["bot_votes"]["mean"] + 1e-9
                                           and naaf_bot <= s["classifier_only"]["bot_votes"]["mean"] + 1e-9)

    for c_max in map(str, C_MAXES):
        print(f"\n{args.eval}, C_max={c_max} (mean of 5 injections) -- Sybil votes accepted (%) / honest collected (%)")
        print(f"  {'evasion f':>9}{'SumUp':>16}{'Classifier':>16}{'NAAF gated':>16}{'NAAF-A':>16}   NAAF<=both?")
        for f in map(str, EVASION):
            s = summary[c_max][f]
            cells = "".join(f"{100 * s[m]['bot_vote_acceptance_rate']['mean']:>8.1f} /{100 * s[m]['honest_collection_rate']['mean']:>5.1f}"
                            for m in ("sumup", "classifier_only", "naaf_gated_hard", "naaf_a"))
            print(f"  {f:>9}{cells}   {s['claim_naaf_le_both']}")

    json.dump({"eval": args.eval, "classifier": CLASSIFIER, "q": Q, "evasion_fractions": EVASION,
               "C_max": C_MAXES, "summary": summary, "per_seed": per_seed},
              open(OUT / f"stuffing_{args.eval}.json", "w"), indent=2)
    print(f"\nSaved {OUT / f'stuffing_{args.eval}.json'}")
