# Valley Heart Clinic: Heart Disease Predictor

A Streamlit dashboard for the random forest model built in `Exam1.ipynb`. The model uses a patient's intake tests to recommend whether they should be sent for an angiogram or sent home.

> **For teaching only.** The 2000 patient records are synthetic, patterned on the public Cleveland Clinic heart-disease data. Do not use this model for real medical decisions.

## Quick start

Requires Python 3.10 or newer (tested on 3.11).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The model trains once when the app starts, which takes a few seconds, and is cached after that.

## Pages

| Page | What it shows |
|---|---|
| **Home** | Heart scan hero image, the note to the director, and the one-time holdout check at the 0.2 threshold |
| **Try a patient** | Enter one patient in a form, or upload a CSV of patients, and get the model's recommendation and probability |
| **The model** | Pipeline diagram, one of the forest's 200 trees, feature importance, and logistic regression vs random forest |
| **Threshold lab** | Change the decision threshold with preset buttons or a slider and see who gets sent, who is missed, and recall for women vs men |
| **Model performance** | ROC curve and the confusion counts at the default 0.5 threshold |
| **Analysis notes** | The remaining written answers from the exam, grouped by topic |

## The model

| | |
|---|---|
| Algorithm | Random forest, 200 trees, `min_samples_leaf=5`, `max_depth=None` |
| Features | 13 intake columns: 8 numeric (median-filled, then scaled) and 5 text (filled with the most common value, then one-hot encoded) |
| Never used as features | `patient_id` (no information about the heart) and `followup_booked` (leaks the answer) |
| Training data | 1500 patients (75% stratified split, `random_state=42`) |
| Cross-validated ROC AUC | 0.940, vs 0.923 for logistic regression |
| Decision threshold | 0.2: a patient is sent for an angiogram when the probability is 0.2 or higher |
| Holdout (500 patients) | 196 sent, 10 with heart disease missed; recall 0.789 for women vs 0.949 for men |

The model is less reliable for women. The Try a patient page shows a warning whenever the patient is female.

## Uploading patients

The **Try a patient → Upload a file** tab accepts a CSV with one row per patient. Download the template from that tab to see the format.

| Column | Allowed values |
|---|---|
| `age` | 18–100 |
| `sex` | `female`, `male` |
| `chest_pain` | `asymptomatic`, `atypical angina`, `non-anginal pain`, `typical angina` |
| `resting_bp` | 60–250 (mm Hg) |
| `cholesterol` | 80–700 (mg/dl) |
| `fasting_sugar_high` | `0`/`1` (or `yes`/`no`) |
| `rest_ecg` | `normal`, `ST-T abnormality`, `enlarged ventricle` |
| `max_heart_rate` | 50–230 |
| `exercise_angina` | `0`/`1` (or `yes`/`no`) |
| `st_depression` | 0–10 |
| `st_slope` | `up`, `flat`, `down` |
| `vessels_colored` | 0–3 |
| `thal` | `normal`, `fixed defect`, `reversible defect` |

- `patient_id` is optional and is only used to label the results.
- Leave a cell blank, or write `?`, if a value is not known. The model fills it in the same way it did during training.
- Any `followup_booked` or `heart_disease` column is ignored.
- Values outside these ranges or not in these lists are reported by row and column, and the file is not scored until they are fixed.
- Uploaded files are processed in memory and are never saved.

## Project layout

```
app.py                  Entry point: page config and navigation
views/
  pages.py              Defines every page once so pages can link to each other
  home.py               Landing page
  patient.py            Try a patient (form and CSV upload)
  model.py              The model (diagram, tree, feature importance)
  threshold_lab.py      Threshold lab
  performance.py        Model performance
  notes.py              Analysis notes
src/
  data.py               Loads the CSV ("?" = missing) and makes the exam's train/test split
  model.py              Builds the chosen pipeline and computes cross-validated probabilities
  metrics.py            Confusion counts, precision/recall, recall by sex, ROC points
  predict.py            Validates patient input and scores it
  ui.py                 Shared helpers: CSS, cached model, answer cards
content/
  answers.py            Written answers, copied word for word from Exam1.ipynb
assets/                 Heart scan image and stylesheet
data/heart_clinic.csv   The 2000 patient records
Exam1.ipynb             The original exam notebook
tests/                  pytest suite
.streamlit/config.toml  Theme colors
```

## Tests

```bash
python -m pytest -q
```

- `test_metrics.py` checks that the app reproduces the numbers printed in the notebook, both cross-validated and on the holdout.
- `test_answers_verbatim.py` checks that every string in `content/answers.py` matches an answer cell in `Exam1.ipynb` word for word, and that no written answer is missing.
- `test_predict.py` covers upload validation and checks that uploaded rows score the same as the model's own predictions.

## The written answers

All analysis writing in the dashboard comes from `content/answers.py`. It is copied exactly from the answer cells in `Exam1.ipynb` and must not be edited; the verbatim test fails if it is. Exam cells 6b and 7a were left blank in the notebook, so they are not shown. Page titles, captions and the home-page subtitle are dashboard text and can be edited freely.

## Credits

Heart SPECT illustration by Patrick J. Lynch, CC BY 2.5.
