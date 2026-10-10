"""Match Exam1.ipynb printed threshold metrics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import recall_by_sex, threshold_metrics
from src.model import CHOSEN_THRESHOLD, fit_and_cv_proba


def _bundle():
    return fit_and_cv_proba()


def test_cv_threshold_0_5_matches_notebook():
    bundle = _bundle()
    m = threshold_metrics(bundle["y_train"], bundle["y_prob_cv"], 0.5)
    assert m["tn"] == 1023
    assert m["fp"] == 70
    assert m["fn"] == 105
    assert m["tp"] == 302
    assert m["n_sent"] == 372
    assert m["missed"] == 105
    assert round(m["precision"], 3) == 0.812
    assert round(m["recall"], 3) == 0.742


def test_cv_threshold_0_2_matches_notebook():
    bundle = _bundle()
    m = threshold_metrics(bundle["y_train"], bundle["y_prob_cv"], CHOSEN_THRESHOLD)
    assert m["n_sent"] == 615
    assert m["missed"] == 21
    assert round(m["precision"], 3) == 0.628
    assert round(m["recall"], 3) == 0.948


def test_cv_recall_by_sex_at_0_2_matches_notebook():
    bundle = _bundle()
    y_pred = (bundle["y_prob_cv"] >= CHOSEN_THRESHOLD).astype(int)
    sex = bundle["X_train"]["sex"]
    r = recall_by_sex(bundle["y_train"], y_pred, sex)
    assert round(r["recall_women"], 3) == 0.885
    assert round(r["recall_men"], 3) == 0.96


def test_holdout_at_0_2_matches_notebook():
    bundle = _bundle()
    m = threshold_metrics(bundle["y_test"], bundle["y_prob_test"], CHOSEN_THRESHOLD)
    assert m["n_sent"] == 196
    assert m["missed"] == 10
    assert m["tn"] == 294
    assert m["fp"] == 70
    assert m["fn"] == 10
    assert m["tp"] == 126
    y_pred = (bundle["y_prob_test"] >= CHOSEN_THRESHOLD).astype(int)
    r = recall_by_sex(bundle["y_test"], y_pred, bundle["X_test"]["sex"])
    assert r["n_women_disease"] == 19
    assert round(r["recall_women"], 3) == 0.789
    assert round(r["recall_men"], 3) == 0.949
