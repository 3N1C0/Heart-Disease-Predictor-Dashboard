import pandas as pd

from src.data import FEATURE_COLUMNS
from src.metrics import recall_by_sex
from src.model import CHOSEN_THRESHOLD
from src.predict import (
    FIELDS,
    NEVER_FEATURES,
    clean_patients,
    missing_fields,
    predict_patients,
    template_frame,
)
from src.ui import get_bundle, inject_css

import streamlit as st

NOT_KNOWN = "Not known"


def _verdict_card(prob: float, sent: bool) -> None:
    pct = prob * 100
    cut = CHOSEN_THRESHOLD * 100
    word = "Send for an angiogram" if sent else "Send home"
    extra = " send" if sent else ""
    st.markdown(
        f"""
<div class="verdict{extra}">
  <div class="lbl">Model's recommendation at the {CHOSEN_THRESHOLD} cutoff</div>
  <div class="word">{word}</div>
  <p>The model gives this patient a <strong>{pct:.0f}%</strong> chance that an angiogram would show heart disease.</p>
  <div class="gauge" role="img" aria-label="Probability {pct:.0f} percent, cutoff {cut:.0f} percent">
    <div class="fill" style="width:{pct:.1f}%"></div>
    <div class="cut" style="left:{cut:.1f}%"><span>cutoff {cut:.0f}%</span></div>
  </div>
  <div class="gauge-scale"><span>0%</span><span>100%</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _sex_caution(sex_value: str | None) -> None:
    if sex_value != "female":
        return
    bundle = get_bundle()
    y_pred = (bundle["y_prob_test"] >= CHOSEN_THRESHOLD).astype(int)
    r = recall_by_sex(bundle["y_test"], y_pred, bundle["X_test"]["sex"])
    st.warning(
        f"This patient is a woman. On the holdout patients the model caught "
        f"{r['recall_women']:.0%} of women with heart disease, compared with "
        f"{r['recall_men']:.0%} of men. Treat a 'send home' result with extra care."
    )


def _form_tab() -> None:
    st.caption(
        "The form starts filled in with a typical patient from the clinic file. "
        "Fields marked optional can be left as not known; the model fills them in "
        "the same way it did during training."
    )
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**About the patient**")
            age = st.number_input(FIELDS["age"]["label"], 18, 100, 55, step=1)
            sex = st.selectbox(FIELDS["sex"]["label"], FIELDS["sex"]["options"], index=1)
            chest_pain = st.selectbox(FIELDS["chest_pain"]["label"], FIELDS["chest_pain"]["options"])
            exercise_angina = st.radio(
                FIELDS["exercise_angina"]["label"], ["No", "Yes"], horizontal=True
            )
        with c2:
            st.markdown("**At rest**")
            resting_bp = st.number_input(FIELDS["resting_bp"]["label"], 60, 250, 130, step=1)
            cholesterol = st.number_input(
                FIELDS["cholesterol"]["label"] + " (optional)",
                min_value=80,
                max_value=700,
                value=None,
                step=1,
                placeholder=NOT_KNOWN,
            )
            fasting = st.radio(
                FIELDS["fasting_sugar_high"]["label"], ["No", "Yes"], horizontal=True
            )
            rest_ecg = st.selectbox(FIELDS["rest_ecg"]["label"], FIELDS["rest_ecg"]["options"])
        with c3:
            st.markdown("**Exercise test and imaging**")
            max_hr = st.number_input(FIELDS["max_heart_rate"]["label"], 50, 230, 152, step=1)
            st_dep = st.number_input(
                FIELDS["st_depression"]["label"], 0.0, 10.0, 0.8, step=0.1, format="%.1f"
            )
            st_slope = st.selectbox(FIELDS["st_slope"]["label"], FIELDS["st_slope"]["options"])
            vessels = st.selectbox(
                FIELDS["vessels_colored"]["label"] + " (optional)", [0, 1, 2, 3, NOT_KNOWN]
            )
            thal = st.selectbox(
                FIELDS["thal"]["label"] + " (optional)", FIELDS["thal"]["options"] + [NOT_KNOWN]
            )
        submitted = st.form_submit_button("Get the model's recommendation", type="primary")

    if not submitted:
        return

    row = {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol if cholesterol is not None else "",
        "fasting_sugar_high": fasting,
        "rest_ecg": rest_ecg,
        "max_heart_rate": max_hr,
        "exercise_angina": exercise_angina,
        "st_depression": st_dep,
        "st_slope": st_slope,
        "vessels_colored": "" if vessels == NOT_KNOWN else vessels,
        "thal": "" if thal == NOT_KNOWN else thal,
    }
    clean, errors = clean_patients(pd.DataFrame([row]))
    if errors:
        for e in errors:
            st.error(e)
        return
    result = predict_patients(get_bundle()["chosen"], clean, CHOSEN_THRESHOLD).iloc[0]
    _verdict_card(float(result["probability"]), result["decision"] != "Send home")
    filled = missing_fields(clean)[0]
    if filled:
        st.caption("Filled in by the model because they were not known: " + ", ".join(filled))
    _sex_caution(sex)


def _upload_tab() -> None:
    st.markdown(
        "Upload a CSV with one row per patient. It needs the 13 intake columns below; "
        "a `patient_id` column is optional and is only used to label the results. "
        "Leave a cell blank (or write `?`) if a value is not known."
    )
    st.download_button(
        "Download a template CSV",
        template_frame().to_csv(index=False).encode("utf-8"),
        file_name="patient_template.csv",
        mime="text/csv",
    )
    with st.expander("Columns and allowed values"):
        rows = []
        for col in FEATURE_COLUMNS:
            spec = FIELDS[col]
            if spec["kind"] == "choice":
                allowed = ", ".join(spec["options"])
            elif spec["kind"] == "flag":
                allowed = "0 or 1 (yes/no also works)"
            else:
                allowed = f"{spec['min']} to {spec['max']}"
            rows.append({"column": col, "meaning": spec["label"], "allowed": allowed})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    upload = st.file_uploader("Patient file", type=["csv"])
    if upload is None:
        return
    try:
        raw = pd.read_csv(upload, dtype=str, keep_default_na=False)
    except Exception as e:  # malformed CSVs raise several pandas error types
        st.error(f"Could not read that file as a CSV: {e}")
        return

    ignored = [c for c in NEVER_FEATURES if c in raw.columns and c != "patient_id"]
    if ignored:
        st.info(
            "Ignored " + ", ".join(ignored) + ": the model never uses these columns "
            "(see Analysis notes on leakage)."
        )

    clean, errors = clean_patients(raw)
    if errors:
        st.error(f"Fix {len(errors)} problem(s) in the file, then upload it again.")
        st.code("\n".join(errors[:50]) + ("\n..." if len(errors) > 50 else ""), language=None)
        return

    result = predict_patients(get_bundle()["chosen"], clean, CHOSEN_THRESHOLD)
    out = pd.DataFrame(index=clean.index)
    out["patient"] = raw["patient_id"] if "patient_id" in raw.columns else [f"Row {i + 1}" for i in clean.index]
    out["sex"] = clean["sex"]
    out["probability"] = result["probability"].round(3)
    out["decision"] = result["decision"]
    out["filled in"] = [", ".join(f) for f in missing_fields(clean)]

    n_sent = int((out["decision"] != "Send home").sum())
    m1, m2, m3 = st.columns(3)
    for col, lbl, val in [
        (m1, "Patients scored", len(out)),
        (m2, "Send for an angiogram", n_sent),
        (m3, "Send home", len(out) - n_sent),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="lbl">{lbl}</div><div class="val">{val}</div></div>',
            unsafe_allow_html=True,
        )
    st.dataframe(
        out,
        hide_index=True,
        width="stretch",
        column_config={
            "probability": st.column_config.ProgressColumn(
                "probability", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    if (out["sex"] == "female").any():
        _sex_caution("female")
    st.download_button(
        "Download results",
        out.to_csv(index=False).encode("utf-8"),
        file_name="patient_predictions.csv",
        mime="text/csv",
    )


def render_patient() -> None:
    inject_css()
    st.title("Try a patient")
    st.markdown(
        f"Enter one patient's intake results, or upload a file of patients, and the clinic's "
        f"random forest will recommend whether to send them for an angiogram. A patient is "
        f"sent when the model's probability is at or above **{CHOSEN_THRESHOLD}**."
    )
    st.markdown(
        '<p class="caption-quiet">For teaching only. The model was trained on 1500 synthetic '
        "records and must not be used for real medical decisions. Uploaded files are scored in "
        "memory and are not saved.</p>",
        unsafe_allow_html=True,
    )
    form_tab, upload_tab = st.tabs(["Fill in a form", "Upload a file"])
    with form_tab:
        _form_tab()
    with upload_tab:
        _upload_tab()
