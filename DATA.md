# Getting the data

The datasets are not stored in this repository: they are too large for GitHub, and their licences do not allow redistribution. Each one can be obtained from its authors and placed in the folder below.

| Dataset | Used for | How to get it | Put it in |
|---|---|---|---|
| SNAP ego-Facebook | Facebook experiments (wide cut, narrow cut, vote stuffing) | Already included: `data/facebook_combined.txt` (from [SNAP](https://snap.stanford.edu/data/ego-Facebook.html)) | `data/` |
| MGTAB | Main real-data result | Download `MGTAB.zip` via the Google Drive link on the [MGTAB GitHub page](https://github.com/GraphDetec/MGTAB) and unzip it. Licence: CC BY-NC-ND 4.0. | `data/real/MGTAB/` (files such as `edge_index.pt`, `labels_bot.pt`, `features.pt`) |
| TwiBot-22 | Not used in any reported experiment | Access is granted by the authors on request; see the [TwiBot-22 repository](https://github.com/LuoUndergradXJTU/TwiBot-22). About 110 GB. | not applicable |

Generated data does not need downloading:

- `data/real/mgtab_social.edgelist` and `data/real/mgtab_meta.json`: run `python src/real/01_load_mgtab.py` after placing MGTAB.
- `data/synthetic_v2/`: run `python src/synthetic_v2/01_make_scenarios.py` and `python src/synthetic_v2/04_make_stuffing.py`.

Splits, classifier predictions and all result files are already committed under `results/`, so the tables can be checked without retraining. `case_study_3_evidence/` holds a verified re-run of each paper table.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```
