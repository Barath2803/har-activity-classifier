"""Shared metric helper used by training and by the Streamlit app."""
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def evaluate(model, X, y):
    """Accuracy, AUC (one-vs-rest), Precision, Recall, F1 (weighted) and MCC."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=model.classes_),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
