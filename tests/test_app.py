"""Smoke test for the Streamlit dashboard, run through Streamlit's test harness."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_dashboard_loads_and_reports_missing_inputs():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception

    at.button[0].click().run()
    assert not at.exception
    messages = [e.value for e in at.error]
    # CI has no checkpoint under outputs/; a local run may have one, in which case the upload check fires.
    assert any("No checkpoint" in m or "Upload both" in m for m in messages), messages
