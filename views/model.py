import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

from content import answers
from src.data import NUMBER_COLUMNS, TEXT_COLUMNS
from src.model import CHOSEN_THRESHOLD
from src.ui import get_bundle, inject_css, render_note

import streamlit as st

CREAM = "#F7F1E8"
CARD = "#FFFDF8"
LINE = "#E4D8C4"
INK = "#1C1917"
MUTED = "#6B635A"
BURGUNDY = "#7A1F2B"
TEAL = "#2F5D62"

PIPELINE_DOT = f"""
digraph {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box style="rounded,filled" fillcolor="{CARD}" color="{LINE}"
        fontname="Helvetica" fontsize=11 fontcolor="{INK}" margin="0.18,0.1"];
  edge [color="{MUTED}" arrowsize=0.7];

  intake [label="Patient intake\\n13 tests from the clinic"];
  drop [label="Never used\\npatient_id, followup_booked" fillcolor="#F8ECEC" color="#E0C4C7"];
  num [label="8 number columns\\nfill gaps: median\\nthen scale"];
  txt [label="5 text columns\\nfill gaps: most common\\nthen one-hot"];
  forest [label="Random forest\\n200 trees\\nmin_samples_leaf = 5\\nmax_depth = None"
          fillcolor="{BURGUNDY}" color="{BURGUNDY}" fontcolor="white"];
  prob [label="Probability of\\nheart disease"];
  cut [label="At or above {CHOSEN_THRESHOLD}?" shape=diamond fillcolor="{CARD}"];
  send [label="Send for an angiogram" fillcolor="#EEF4F4" color="#C9D9DA"];
  home [label="Send home"];

  intake -> num; intake -> txt;
  intake -> drop [style=dashed arrowhead=tee];
  num -> forest; txt -> forest;
  forest -> prob -> cut;
  cut -> send [label=" yes" fontsize=10 fontcolor="{MUTED}"];
  cut -> home [label=" no" fontsize=10 fontcolor="{MUTED}"];
}}
"""


def readable_names(raw_names) -> list[str]:
    """'num__age' -> 'age', 'cat__chest_pain_asymptomatic' -> 'chest_pain = asymptomatic'."""
    out = []
    for name in raw_names:
        _, col = name.split("__", 1)
        for text_col in TEXT_COLUMNS:
            if col.startswith(text_col + "_"):
                col = f"{text_col} = {col[len(text_col) + 1:]}"
                break
        out.append(col)
    return out


@st.cache_data
def _importances() -> pd.DataFrame:
    chosen = get_bundle()["chosen"]
    names = readable_names(chosen.named_steps["prep"].get_feature_names_out())
    imp = chosen.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _style_axes(fig, ax) -> None:
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _importance_chart(table: pd.DataFrame, top: int = 10) -> None:
    shown = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _style_axes(fig, ax)
    ax.barh(shown["feature"], shown["importance"], color=BURGUNDY, height=0.62)
    ax.xaxis.grid(True, color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=INK, labelsize=10)
    ax.set_xlabel("share of the forest's splitting power", color=MUTED, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _plain_split(line: str, scaler) -> str:
    """Undo scaling/one-hot in a split label: 'age <= 1.579' -> 'age <= 69.5'."""
    if " <= " not in line:
        return line
    feature, cut = line.rsplit(" <= ", 1)
    if " = " in feature:  # one-hot column: <= 0.5 means the patient does NOT have this value
        col, value = feature.split(" = ", 1)
        return f"{col} is not {value}"
    if feature in NUMBER_COLUMNS:
        i = NUMBER_COLUMNS.index(feature)
        real = float(cut) * scaler.scale_[i] + scaler.mean_[i]
        return f"{feature} <= {real:.1f}"
    return line


def _tree_figure(depth: int) -> None:
    chosen = get_bundle()["chosen"]
    tree = chosen.named_steps["model"].estimators_[0]
    names = readable_names(chosen.named_steps["prep"].get_feature_names_out())
    width = {1: 7, 2: 11, 3: 18}[depth]
    fig, ax = plt.subplots(figsize=(width, 2.4 + 1.6 * depth))
    fig.patch.set_facecolor(CARD)
    artists = plot_tree(
        tree,
        max_depth=depth,
        feature_names=names,
        class_names=["no heart disease", "heart disease"],
        impurity=False,
        proportion=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    scaler = chosen.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    # plot_tree colors nodes from matplotlib's default cycle; restyle to the clinic palette.
    for art in artists:
        lines = art.get_text().split("\n")
        lines[0] = _plain_split(lines[0], scaler)
        art.set_text("\n".join(lines))
        box = art.get_bbox_patch()
        if box is not None:
            box.set_facecolor(CREAM)
            box.set_edgecolor(LINE)
        art.set_color(INK)
        if art.arrow_patch is not None:
            art.arrow_patch.set_color(MUTED)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_model() -> None:
    inject_css()
    st.title("The model")
    st.markdown(
        "The clinic's final model is a random forest inside a scikit-learn pipeline. "
        "Everything a new patient's data goes through, from intake to the decision:"
    )
    st.graphviz_chart(PIPELINE_DOT, width="stretch")

    c1, c2, c3 = st.columns(3)
    for col, lbl, val, cls in [
        (c1, "Logistic regression, best ROC AUC", "0.923", " teal"),
        (c2, "Random forest, best ROC AUC", "0.940", ""),
        (c3, "Decision threshold", f"{CHOSEN_THRESHOLD}", ""),
    ]:
        col.markdown(
            f'<div class="metric-card{cls}"><div class="lbl">{lbl}</div>'
            f'<div class="val">{val}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    render_note("Why the random forest", answers.MODEL_CHOICE)

    st.subheader("What the forest pays attention to")
    left, right = st.columns((1.3, 1), gap="large")
    table = _importances()
    with left:
        _importance_chart(table)
    with right:
        st.markdown(
            "Each bar is how much a feature helped the 200 trees separate patients with "
            "and without heart disease, averaged over the forest. Text columns are split "
            "into one bar per value after one-hot encoding."
        )
        with st.expander("All features as a table"):
            st.dataframe(
                table.assign(importance=table["importance"].round(3)),
                hide_index=True,
                width="stretch",
            )

    st.subheader("One of the 200 trees")
    st.markdown(
        "This is the first tree in the forest, drawn from its top. Each box asks one yes/no "
        "question (left branch = True = yes); `value` is the share of patients in that box "
        "(from this tree's own sample of the training data) without and with heart disease. "
        "The full tree keeps splitting as long as every leaf still holds at least 5 patients; "
        "the forest averages 200 trees like this one."
    )
    depth = st.segmented_control(
        "Levels to show", options=[1, 2, 3], default=2, key="tree_depth"
    )
    _tree_figure(depth or 2)

    render_note("A forest that grew too far", answers.FOREST_OVERFIT)
    render_note("How missing values are filled", answers.PIPELINE_IMPUTATION)
