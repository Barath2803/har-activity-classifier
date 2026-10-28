"""Train the five classifiers on UCI HAR, save them and the test-set metrics.

Usage (from the project root): python model/train_models.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics import evaluate

ROOT = Path(__file__).resolve().parent.parent
TARGET = "Activity"

train = pd.read_csv(ROOT / "data" / "train_data.csv.gz")
test = pd.read_csv(ROOT / "test_data.csv")
X_train, y_train = train.drop(columns=TARGET), train[TARGET]
X_test, y_test = test.drop(columns=TARGET), test[TARGET]

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True, random_state=42)),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, ROOT / "model" / f"{name.lower().replace(' ', '_')}.joblib", compress=3)
    m = evaluate(model, X_test, y_test)
    rows.append({"ML Model Name": name, **m})
    print(name, {k: round(v, 4) for k, v in m.items()})

pd.DataFrame(rows).to_csv(ROOT / "model" / "metrics.csv", index=False)
json.dump(sorted(y_train.unique().tolist()), open(ROOT / "model" / "classes.json", "w"))
