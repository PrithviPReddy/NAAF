"""
Real-data Step 7: test-split metrics for the already-fixed MGTAB classifiers
(report only -- every model and setting was chosen on val before this ran;
nothing is selected here).

Metrics per classifier, mean +/- std over the 5 splits: ROC-AUC, F1 at 0.5,
and the share of test bots ranked above all but 1% of test humans
(TPR at 1% FPR) -- the regime NAAF's threshold operates in.
"""
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIERS = ["gatv2_social_skip_e600", "mlp_e600", "gatv2_social_e600", "gatv2_all"]
SEEDS = [0, 1, 2, 3, 4]

if __name__ == "__main__":
    meta = json.load(open(ROOT / "data" / "real" / "mgtab_meta.json"))
    nodes = sorted(meta["bot_nodes"] + meta["human_nodes"])
    pos = {v: i for i, v in enumerate(nodes)}
    bots = set(meta["bot_nodes"])

    out = {}
    for clf in CLASSIFIERS:
        rows = []
        for seed in SEEDS:
            p = np.load(ROOT / "results" / "real" / "classifier" / f"p_bot_{clf}_seed{seed}.npy")
            test = json.load(open(ROOT / "results" / "real" / "splits" / f"split_seed{seed}.json"))["test"]
            y = np.array([v in bots for v in test])
            pt = p[[pos[v] for v in test]]
            fpr, tpr, _ = roc_curve(y, pt)
            rows.append({"auc": roc_auc_score(y, pt), "f1": f1_score(y, pt > 0.5),
                         "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr))})
        out[clf] = {k: {"mean": round(statistics.mean(r[k] for r in rows), 4),
                        "std": round(statistics.stdev(r[k] for r in rows), 4)} for k in rows[0]}
        print(f"{clf:>24}: " + "  ".join(f"{k} {v['mean']:.4f} +/- {v['std']:.4f}" for k, v in out[clf].items()))

    path = ROOT / "results" / "real" / "classifier" / "classifier_test_metrics.json"
    json.dump({"note": "gatv2_all = 300 epochs (Step 5); the others = 600 epochs (Step 6a)", **out},
              open(path, "w"), indent=2)
    print(f"Saved {path}")
