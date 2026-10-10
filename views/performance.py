import matplotlib.pyplot as plt

from content import answers
from src.metrics import roc_points
from src.ui import get_bundle, inject_css, render_note

import streamlit as st


def render_performance() -> None:
    inject_css()
    st.title("Model performance")
    st.markdown(
        "The clinic uses a random forest. Curves and the 0.5 confusion sentences below "
        "are from cross-validated training probabilities — the same view as the exam."
    )

    bundle = get_bundle()
    roc = roc_points(bundle["y_train"], bundle["y_prob_cv"])

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor("#F7F1E8")
    ax.set_facecolor("#F7F1E8")
    ax.plot(roc["fpr"], roc["tpr"], color="#7A1F2B", linewidth=2.4, label="random forest")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#1C1917", linewidth=1, label="random guessing")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title(f"ROC curve, AUC = {round(roc['auc'], 3)}")
    ax.grid(color="#E4D8C4")
    ax.legend(frameon=False)
    st.pyplot(fig, width="content")
    plt.close(fig)

    render_note("In plain words", answers.AUC_PLAIN)
    render_note("At the default 0.5 threshold", answers.CONFUSION_AT_HALF)
    render_note("Logistic regression, before the search", answers.LOGISTIC_CV)
