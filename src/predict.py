"""Check patient intake values and score them with the chosen forest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS

# Plain-language labels and allowed values for each model feature.
# Number ranges are wide on purpose: they reject typos, not unusual patients.
FIELDS = {
    "age": {"label": "Age (years)", "kind": "number", "min": 18, "max": 100},
    "sex": {"label": "Sex", "kind": "choice", "options": ["female", "male"]},
    "chest_pain": {
        "label": "Chest pain type",
        "kind": "choice",
        "options": ["asymptomatic", "atypical angina", "non-anginal pain", "typical angina"],
    },
    "resting_bp": {"label": "Resting blood pressure (mm Hg)", "kind": "number", "min": 60, "max": 250},
    "cholesterol": {"label": "Cholesterol (mg/dl)", "kind": "number", "min": 80, "max": 700},
    "fasting_sugar_high": {"label": "Fasting blood sugar above 120 mg/dl", "kind": "flag"},
    "rest_ecg": {
        "label": "Resting ECG",
        "kind": "choice",
        "options": ["normal", "ST-T abnormality", "enlarged ventricle"],
    },
    "max_heart_rate": {"label": "Max heart rate on exercise test", "kind": "number", "min": 50, "max": 230},
    "exercise_angina": {"label": "Chest pain during exercise", "kind": "flag"},
    "st_depression": {"label": "ST depression on exercise test", "kind": "number", "min": 0, "max": 10},
    "st_slope": {"label": "Slope of the ST segment", "kind": "choice", "options": ["up", "flat", "down"]},
    "vessels_colored": {
        "label": "Major vessels colored by fluoroscopy",
        "kind": "number",
        "min": 0,
        "max": 3,
    },
    "thal": {
        "label": "Thallium stress test",
        "kind": "choice",
        "options": ["normal", "fixed defect", "reversible defect"],
    },
}

# Columns that must never reach the model (exam 2d/2e): one leaks the answer,
# the other carries no information about the heart.
NEVER_FEATURES = ["patient_id", "followup_booked", "heart_disease"]

_YES = {"1", "1.0", "yes", "y", "true"}
_NO = {"0", "0.0", "no", "n", "false"}


def _clean_flag(value):
    text = str(value).strip().lower()
    if text in _YES:
        return 1
    if text in _NO:
        return 0
    raise ValueError("use 0/1 or yes/no")


def clean_patients(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (rows ready for the model, error messages).

    Blank or "?" cells are kept as missing; the pipeline fills them the same way
    it did in training. Any error means the whole upload should be fixed first.
    """
    raw = raw.rename(columns=lambda c: str(c).strip())
    missing_cols = [c for c in FEATURE_COLUMNS if c not in raw.columns]
    if missing_cols:
        return pd.DataFrame(columns=FEATURE_COLUMNS), [
            "Missing columns: " + ", ".join(missing_cols)
        ]

    clean = pd.DataFrame(index=raw.index)
    errors: list[str] = []
    for col in FEATURE_COLUMNS:
        spec = FIELDS[col]
        values = raw[col].replace({"?": np.nan, "": np.nan})
        out = []
        for i, v in values.items():
            row = i + 1
            if pd.isna(v) or (isinstance(v, str) and not v.strip()):
                out.append(np.nan)
                continue
            if spec["kind"] == "choice":
                text = str(v).strip()
                match = {o.lower(): o for o in spec["options"]}.get(text.lower())
                if match is None:
                    errors.append(
                        f"Row {row}, {col}: '{text}' is not one of " + ", ".join(spec["options"])
                    )
                out.append(match)
            elif spec["kind"] == "flag":
                try:
                    out.append(_clean_flag(v))
                except ValueError as e:
                    errors.append(f"Row {row}, {col}: '{v}' - {e}")
                    out.append(np.nan)
            else:
                try:
                    num = float(v)
                except (TypeError, ValueError):
                    errors.append(f"Row {row}, {col}: '{v}' is not a number")
                    out.append(np.nan)
                    continue
                if not spec["min"] <= num <= spec["max"]:
                    errors.append(
                        f"Row {row}, {col}: {v} is outside {spec['min']}-{spec['max']}"
                    )
                out.append(num)
        clean[col] = out

    for col in FEATURE_COLUMNS:
        if FIELDS[col]["kind"] != "choice":
            clean[col] = pd.to_numeric(clean[col])
        else:
            clean[col] = clean[col].astype(object)
    if clean.empty:
        errors.append("The file has no patient rows.")
    return clean[FEATURE_COLUMNS], errors


def missing_fields(clean: pd.DataFrame) -> list[list[str]]:
    """Per row, the features the model had to fill in."""
    return [[c for c in FEATURE_COLUMNS if pd.isna(row[c])] for _, row in clean.iterrows()]


def predict_patients(model, clean: pd.DataFrame, threshold: float) -> pd.DataFrame:
    prob = model.predict_proba(clean[FEATURE_COLUMNS])[:, 1]
    sent = prob >= threshold
    return pd.DataFrame(
        {
            "probability": prob,
            "decision": np.where(sent, "Send for an angiogram", "Send home"),
        },
        index=clean.index,
    )


def template_frame() -> pd.DataFrame:
    """Two example rows showing the expected columns and spellings."""
    rows = [
        {
            "patient_id": "P-001", "age": 63, "sex": "male", "chest_pain": "asymptomatic",
            "resting_bp": 145, "cholesterol": 233, "fasting_sugar_high": 1,
            "rest_ecg": "enlarged ventricle", "max_heart_rate": 150, "exercise_angina": 0,
            "st_depression": 2.3, "st_slope": "down", "vessels_colored": 0, "thal": "fixed defect",
        },
        {
            "patient_id": "P-002", "age": 41, "sex": "female", "chest_pain": "atypical angina",
            "resting_bp": 130, "cholesterol": "", "fasting_sugar_high": 0,
            "rest_ecg": "normal", "max_heart_rate": 172, "exercise_angina": 0,
            "st_depression": 1.4, "st_slope": "up", "vessels_colored": "", "thal": "",
        },
    ]
    return pd.DataFrame(rows, columns=["patient_id"] + FEATURE_COLUMNS)
