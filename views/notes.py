from content import answers
from src.ui import inject_css, render_note

import streamlit as st


def render_notes() -> None:
    inject_css()
    st.title("Analysis notes")
    st.markdown("Writing from the exam, grouped by topic. Wording is unchanged.")

    render_note("What is in the file", answers.DATA_SIZE)
    render_note("A model that always says no", answers.LAZY_BASELINE)
    render_note("Columns that cannot be features", answers.DROPPED_COLUMNS)
    render_note("Why follow-up booking leaks the answer", answers.FOLLOWUP_LEAKAGE)
    render_note("Chest pain does not mean the same thing for men and women", answers.CHEST_PAIN_BY_SEX)
    render_note("Holdout compared with cross validation", answers.TEST_VS_CV)
    render_note("Note to the director", answers.DIRECTOR_LETTER)
