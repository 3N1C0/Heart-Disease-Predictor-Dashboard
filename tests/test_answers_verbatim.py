"""Every string in content/answers.py must be word-for-word from Exam1.ipynb.

The notebook italicises each answer paragraph (*...*) and puts "**Your answer:**"
above it; only that wrapper and trailing spaces are ignored. Any other edit fails.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content import answers

NAMES = [n for n in dir(answers) if n.isupper()]


def _paragraphs(text: str) -> list[str]:
    text = text.replace("**Your answer:**", "")
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = "\n".join(line.rstrip() for line in p.strip().splitlines())
        if not p:
            continue
        if p.startswith("*") and p.endswith("*") and not p.startswith("**"):
            p = p[1:-1].strip()
        out.append(p)
    return out


def _notebook_answers() -> list[list[str]]:
    nb = json.loads((ROOT / "Exam1.ipynb").read_text())
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and src.strip().startswith("**Your answer:**"):
            cells.append(_paragraphs(src))
    return cells


NOTEBOOK = _notebook_answers()


@pytest.mark.parametrize("name", NAMES)
def test_answer_is_verbatim(name):
    assert _paragraphs(getattr(answers, name)) in NOTEBOOK, (
        f"answers.{name} no longer matches any answer cell in Exam1.ipynb"
    )


def test_every_written_answer_is_included():
    included = [_paragraphs(getattr(answers, n)) for n in NAMES]
    for cell in NOTEBOOK:
        if cell:
            assert cell in included
