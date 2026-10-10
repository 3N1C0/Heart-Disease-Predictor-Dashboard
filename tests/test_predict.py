"""Patient upload checks and scoring for the Try a patient page."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_heart
from src.model import CHOSEN_THRESHOLD, fit_and_cv_proba
from src.predict import clean_patients, missing_fields, predict_patients, template_frame


def _as_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Mimic the page: every cell read as text, blanks as ''."""
    return df.astype(str).replace({"nan": ""})


def test_template_is_valid():
    clean, errors = clean_patients(_as_upload(template_frame()))
    assert errors == []
    assert missing_fields(clean) == [[], ["cholesterol", "vessels_colored", "thal"]]


def test_reports_missing_columns():
    _, errors = clean_patients(template_frame().drop(columns=["thal", "age"]))
    assert errors == ["Missing columns: age, thal"]


def test_reports_bad_values_with_row_numbers():
    bad = template_frame().astype(str)
    bad.loc[0, "chest_pain"] = "sharp"
    bad.loc[1, "age"] = "400"
    bad.loc[1, "exercise_angina"] = "maybe"
    _, errors = clean_patients(_as_upload(bad))
    assert len(errors) == 3
    assert any(e.startswith("Row 1, chest_pain") for e in errors)
    assert any(e.startswith("Row 2, age") for e in errors)
    assert any(e.startswith("Row 2, exercise_angina") for e in errors)


def test_accepts_yes_no_and_case_differences():
    df = template_frame().astype(str)
    df.loc[0, "fasting_sugar_high"] = "Yes"
    df.loc[0, "sex"] = "Male"
    clean, errors = clean_patients(_as_upload(df))
    assert errors == []
    assert clean.loc[0, "fasting_sugar_high"] == 1
    assert clean.loc[0, "sex"] == "male"


def test_upload_scores_match_model_on_clinic_rows():
    bundle = fit_and_cv_proba()
    heart = load_heart()
    rows = heart.loc[bundle["X_test"].index[:25]]
    clean, errors = clean_patients(_as_upload(rows))
    assert errors == []
    got = predict_patients(bundle["chosen"], clean, CHOSEN_THRESHOLD)
    expected = bundle["y_prob_test"][:25]
    assert (abs(got["probability"].to_numpy() - expected) < 1e-12).all()
    assert ((got["decision"] != "Send home") == (expected >= CHOSEN_THRESHOLD)).all()
