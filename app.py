"""Valley Heart Clinic dashboard."""

import streamlit as st

st.set_page_config(
    page_title="Valley Heart Clinic",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from views import pages  # noqa: E402  (pages must be built after set_page_config)

pg = st.navigation({"Valley Heart Clinic": pages.ALL})
pg.run()
