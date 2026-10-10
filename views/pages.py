"""Every page in the app, defined once so pages can link to each other."""

import streamlit as st

from views.home import render_home
from views.model import render_model
from views.notes import render_notes
from views.patient import render_patient
from views.performance import render_performance
from views.threshold_lab import render_threshold_lab

home = st.Page(render_home, title="Home", url_path="home", default=True)
patient = st.Page(render_patient, title="Try a patient", url_path="patient")
model = st.Page(render_model, title="The model", url_path="model")
lab = st.Page(render_threshold_lab, title="Threshold lab", url_path="threshold_lab")
perf = st.Page(render_performance, title="Model performance", url_path="performance")
notes = st.Page(render_notes, title="Analysis notes", url_path="notes")

ALL = [home, patient, model, lab, perf, notes]
