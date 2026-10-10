"""Load the clinic CSV and apply the exam train/test split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "heart_clinic.csv"

NUMBER_COLUMNS = [
    "age",
    "resting_bp",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
    "vessels_colored",
    "fasting_sugar_high",
    "exercise_angina",
]
TEXT_COLUMNS = ["chest_pain", "rest_ecg", "st_slope", "thal", "sex"]
FEATURE_COLUMNS = NUMBER_COLUMNS + TEXT_COLUMNS


def load_heart(path: Path | None = None) -> pd.DataFrame:
    # The file marks missing values with "?" (same as Exam1.ipynb cell 6).
    return pd.read_csv(path or CSV_PATH, na_values="?")


def make_xy(heart: pd.DataFrame):
    y = heart["heart_disease"]
    X = heart[FEATURE_COLUMNS]
    return X, y


def split_exam(X, y, test_size: float = 0.25, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
