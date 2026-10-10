"""Shared Streamlit helpers."""

from __future__ import annotations

import base64
import re
from pathlib import Path

import streamlit as st

from src.model import fit_and_cv_proba

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def inject_css() -> None:
    css = (ASSETS / "style.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def image_data_uri(name: str) -> str:
    path = ASSETS / name
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{b64}"


@st.cache_resource
def get_bundle():
    return fit_and_cv_proba()


def render_note(label: str, text: str) -> None:
    """Show one piece of exam writing inside a card. `text` is passed through untouched."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    with st.container(key=f"note-{slug}"):
        st.markdown(f'<div class="note-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(text)
