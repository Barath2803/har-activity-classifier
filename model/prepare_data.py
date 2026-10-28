"""Build train/test CSVs from the raw UCI HAR Dataset folder.

Usage: python model/prepare_data.py "<path to 'UCI HAR Dataset'>"
Writes data/train_data.csv.gz and test_data.csv (561 features + Activity label).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
raw = Path(sys.argv[1])

names = pd.read_csv(raw / "features.txt", sep=r"\s+", header=None, names=["i", "name"])["name"]
# several feature names repeat in the raw file; suffix them so columns are unique
seen = {}
cols = []
for n in names:
    seen[n] = seen.get(n, 0) + 1
    cols.append(n if seen[n] == 1 else f"{n}_{seen[n]}")

labels = pd.read_csv(raw / "activity_labels.txt", sep=r"\s+", header=None, index_col=0)[1]


def load(split):
    X = pd.read_csv(raw / split / f"X_{split}.txt", sep=r"\s+", header=None, names=cols)
    y = pd.read_csv(raw / split / f"y_{split}.txt", header=None)[0].map(labels)
    X["Activity"] = y
    return X


(ROOT / "data").mkdir(exist_ok=True)
load("train").to_csv(ROOT / "data" / "train_data.csv.gz", index=False, float_format="%.6g")
load("test").to_csv(ROOT / "test_data.csv", index=False, float_format="%.6g")
print("done")
