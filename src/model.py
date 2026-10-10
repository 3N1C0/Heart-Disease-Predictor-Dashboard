"""Chosen random-forest pipeline from the exam (no grid search at startup)."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import NUMBER_COLUMNS, TEXT_COLUMNS, load_heart, make_xy, split_exam

CHOSEN_THRESHOLD = 0.2
PRESET_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def exam_folds(random_state: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def build_preprocessing() -> ColumnTransformer:
    number_path = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    text_path = Pipeline(
        [
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("dummy", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", number_path, NUMBER_COLUMNS),
            ("cat", text_path, TEXT_COLUMNS),
        ]
    )


def build_chosen_forest() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessing()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    max_depth=None,
                    min_samples_leaf=5,
                ),
            ),
        ]
    )


def fit_and_cv_proba():
    heart = load_heart()
    X, y = make_xy(heart)
    X_train, X_test, y_train, y_test = split_exam(X, y)
    chosen = build_chosen_forest()
    folds = exam_folds()
    y_prob_cv = cross_val_predict(
        chosen,
        X_train,
        y_train,
        cv=folds,
        method="predict_proba",
    )[:, 1]
    chosen.fit(X_train, y_train)
    y_prob_test = chosen.predict_proba(X_test)[:, 1]
    return {
        "chosen": chosen,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob_cv": y_prob_cv,
        "y_prob_test": y_prob_test,
    }
