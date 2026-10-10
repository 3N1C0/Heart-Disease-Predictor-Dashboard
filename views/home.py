from content import answers
from src.metrics import recall_by_sex, threshold_metrics
from src.model import CHOSEN_THRESHOLD
from src.ui import get_bundle, image_data_uri, inject_css, render_note

import streamlit as st


def render_home() -> None:
    inject_css()
    uri = image_data_uri("heart_scan.jpg")
    st.markdown(
        f"""
<div class="hero">
  <img src="{uri}" alt="Illustrative heart scan" />
  <div class="hero-overlay"></div>
  <div class="hero-copy">
    <div class="hero-kicker">Decision support</div>
    <h1 class="hero-title">Valley Heart Clinic</h1>
    <p class="hero-sub">A model that helps decide who is sent for an angiogram, and who is sent home — built on the clinic’s own intake tests, not the hospital result.</p>
  </div>
</div>
<p class="caption-quiet">Heart SPECT illustration by Patrick J. Lynch, CC BY 2.5. The 2000 patient records are synthetic, patterned on public Cleveland Clinic heart-disease data. This dashboard is not for real medical decisions.</p>
""",
        unsafe_allow_html=True,
    )

    bundle = get_bundle()
    holdout = threshold_metrics(
        bundle["y_test"], bundle["y_prob_test"], CHOSEN_THRESHOLD
    )
    y_pred = (bundle["y_prob_test"] >= CHOSEN_THRESHOLD).astype(int)
    sex = recall_by_sex(bundle["y_test"], y_pred, bundle["X_test"]["sex"])

    left, right = st.columns((1.4, 1), gap="large")
    with left:
        st.subheader("Note to the director")
        st.markdown(f'<div class="letter">{answers.DIRECTOR_LETTER}</div>', unsafe_allow_html=True)
        st.caption("Shown exactly as written.")
    with right:
        st.markdown(
            f"""
<div class="holdout">
  <h3>Holdout check at 0.2</h3>
  <p class="muted">Scored one time on 500 patients the model had not seen. Threshold shopping does not use these numbers.</p>
  <p><strong>{holdout['n_sent']}</strong> sent for an angiogram<br/>
  <strong>{holdout['missed']}</strong> with heart disease sent home<br/>
  Women with heart disease in this set: <strong>{sex['n_women_disease']}</strong></p>
</div>
""",
            unsafe_allow_html=True,
        )

    from views import pages  # imported here: views.pages imports this module

    st.markdown("##### Explore the model")
    l1, l2, l3 = st.columns(3)
    for col, page, blurb in [
        (l1, pages.patient, "Get a recommendation for one patient, or upload a file of patients."),
        (l2, pages.model, "See the pipeline, one of the 200 trees, and what the forest pays attention to."),
        (l3, pages.lab, "Try other cutoffs on the training cohort."),
    ]:
        with col:
            st.page_link(page, label=f"{page.title} →")
            st.caption(blurb)
