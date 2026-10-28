"""Streamlit demo: Human Activity Recognition (UCI HAR) with five classifiers."""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
from metrics import evaluate  # noqa: E402

TARGET = "Activity"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Activity Recognition Classifiers", page_icon="🏃", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(ROOT / "model" / MODEL_FILES[name])


st.title("🏃 Human Activity Recognition from Smartphone Sensors")
st.caption("UCI HAR dataset · 561 sensor features · 6 activities · five classic classifiers compared")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv",
                                help="Same 561 feature columns as test_data.csv, plus the 'Activity' label column.")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    st.markdown("Don't have a file? Use `test_data.csv` from the GitHub repo.")

tab_one, tab_all = st.tabs(["Selected model", "Compare all models"])

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to evaluate the models.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"The file needs a '{TARGET}' column with the true labels.")
    st.stop()

X, y = df.drop(columns=TARGET), df[TARGET]
st.sidebar.success(f"Loaded {len(df):,} rows × {X.shape[1]} features")

with tab_one:
    model = load_model(model_name)
    missing = [c for c in model.feature_names_in_ if c not in X.columns]
    if missing:
        st.error(f"{len(missing)} expected feature columns are missing, e.g. {missing[:3]}")
        st.stop()
    X_m = X[list(model.feature_names_in_)]

    st.subheader(f"{model_name} — evaluation metrics")
    scores = evaluate(model, X_m, y)
    cols = st.columns(len(scores))
    for col, (k, v) in zip(cols, scores.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    pred = model.predict(X_m)
    labels = list(model.classes_)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y, pred, labels=labels), annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=40, ha="right")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T
        st.dataframe(report.round(3), width="stretch")

with tab_all:
    st.subheader("All five models on the uploaded data")
    if st.button("Run all models"):
        rows = []
        for name in MODEL_FILES:
            m = load_model(name)
            rows.append({"ML Model Name": name, **evaluate(m, X[list(m.feature_names_in_)], y)})
        table = pd.DataFrame(rows).set_index("ML Model Name")
        st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                     width="stretch")
        st.bar_chart(table[["Accuracy", "F1", "MCC"]])
