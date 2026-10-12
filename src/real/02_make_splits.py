"""
Real-data Step 2: freeze 5 stratified train/val/test splits of the MGTAB
social-graph LCC (70/10/20, bot ratio preserved in every part).

Rules every later script follows (decided up front, so results can't be
tuned on the numbers we report):
  - train: the only labels any classifier may see; also where the trusted
    SumUp collector is picked from.
  - val:   all NAAF design/tuning decisions are compared here.
  - test:  the voters. Touched once per final method, never for tuning.

Splits are saved (node ids only, no MGTAB content) to results/real/splits/
so they're versioned with the results that use them.
"""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2, 3, 4]
OUT = ROOT / "results" / "real" / "splits"
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "data" / "real" / "mgtab_meta.json") as f:
    meta = json.load(f)
bot_set = set(meta["bot_nodes"])
nodes = sorted(meta["bot_nodes"] + meta["human_nodes"])
y = [1 if v in bot_set else 0 for v in nodes]

for seed in SEEDS:
    train, rest, _, y_rest = train_test_split(nodes, y, train_size=0.7, stratify=y, random_state=seed)
    val, test = train_test_split(rest, train_size=1 / 3, stratify=y_rest, random_state=seed)  # 10% / 20%
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert len(train) + len(val) + len(test) == len(nodes)
    split = {"seed": seed, "train": sorted(train), "val": sorted(val), "test": sorted(test)}
    counts = {k: (len(split[k]), sum(v in bot_set for v in split[k])) for k in ("train", "val", "test")}
    print(f"seed {seed}: " + "  ".join(f"{k}={n} ({b} bots)" for k, (n, b) in counts.items()))
    with open(OUT / f"split_seed{seed}.json", "w") as f:
        json.dump(split, f)
print(f"Saved {len(SEEDS)} splits to {OUT}")
