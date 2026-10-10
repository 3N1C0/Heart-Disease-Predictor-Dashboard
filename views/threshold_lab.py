from content import answers
from src.metrics import delta_vs_half, recall_by_sex, threshold_metrics
from src.model import CHOSEN_THRESHOLD, PRESET_THRESHOLDS
from src.ui import get_bundle, inject_css, render_note

import streamlit as st


def render_threshold_lab() -> None:
    inject_css()
    st.title("Threshold lab")
    st.markdown(
        "These counts use **cross-validated probabilities on the 1500 training patients**. "
        "If the probability is at or above the threshold, the patient is sent for an angiogram."
    )
    st.markdown(
        f'<span class="chosen-chip">Your choice: {CHOSEN_THRESHOLD}</span>',
        unsafe_allow_html=True,
    )

    if "threshold" not in st.session_state:
        st.session_state.threshold = CHOSEN_THRESHOLD

    cols = st.columns(len(PRESET_THRESHOLDS) + 1)
    for i, t in enumerate(PRESET_THRESHOLDS):
        label = f"{t:.1f}"
        if t == CHOSEN_THRESHOLD:
            label = f"{t:.1f} *"
        if cols[i].button(label, width="stretch"):
            st.session_state.threshold = t
            st.rerun()
    if cols[-1].button("Reset to 0.2", width="stretch"):
        st.session_state.threshold = CHOSEN_THRESHOLD
        st.rerun()

    t = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        step=0.01,
        key="threshold",
    )

    bundle = get_bundle()
    y_true = bundle["y_train"]
    y_prob = bundle["y_prob_cv"]
    m = threshold_metrics(y_true, y_prob, t)
    half = threshold_metrics(y_true, y_prob, 0.5)
    d = delta_vs_half(m, half)
    y_pred = (y_prob >= t).astype(int)
    sex = recall_by_sex(y_true, y_pred, bundle["X_train"]["sex"])

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Sent for angiogram", m["n_sent"], False),
        (c2, "Heart disease sent home", m["missed"], True),
        (c3, "Precision", f"{m['precision']:.3f}", False),
        (c4, "Recall", f"{m['recall']:.3f}", False),
    ]
    for col, lbl, val, warn in cards:
        extra = " warn" if warn else ""
        col.markdown(
            f'<div class="metric-card{extra}"><div class="lbl">{lbl}</div>'
            f'<div class="val">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### What this threshold does to patients")
    st.markdown(
        f"""
<div class="conf-grid">
  <div class="conf-card"><div class="n">{m['tn']}</div><p>did not have heart disease and were sent home.</p></div>
  <div class="conf-card send"><div class="n">{m['fp']}</div><p>did not have heart disease and were sent for an angiogram.</p></div>
  <div class="conf-card miss"><div class="n">{m['fn']}</div><p>did have heart disease and were sent home.</p></div>
  <div class="conf-card"><div class="n">{m['tp']}</div><p>did have heart disease and were sent for an angiogram.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

    w1, w2 = st.columns(2)
    w1.markdown(
        f'<div class="metric-card teal"><div class="lbl">Recall, women</div>'
        f'<div class="val">{sex["recall_women"]:.3f}</div></div>',
        unsafe_allow_html=True,
    )
    w2.markdown(
        f'<div class="metric-card teal"><div class="lbl">Recall, men</div>'
        f'<div class="val">{sex["recall_men"]:.3f}</div></div>',
        unsafe_allow_html=True,
    )

    if abs(t - CHOSEN_THRESHOLD) < 1e-9:
        st.info(
            f"Compared with 0.5: {d['more_sent']} more patients sent, "
            f"{d['fewer_missed']} fewer with heart disease sent home."
        )
    else:
        sign_sent = "+" if d["more_sent"] >= 0 else ""
        st.caption(
            f"Vs threshold 0.5 on this same cohort: {sign_sent}{d['more_sent']} sent, "
            f"{d['fewer_missed']} fewer missed (negative means more missed)."
        )

    render_note("Why this cutoff was chosen", answers.THRESHOLD_CHOICE)
    render_note("Women and men at 0.2", answers.RECALL_BY_SEX)
