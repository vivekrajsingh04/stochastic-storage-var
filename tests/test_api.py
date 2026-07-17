"""API tests: bring-your-own-fleet analyzer + auto-refresh quarter logic."""
import io
import os
import sys
from datetime import date

import pytest

pytest.importorskip("httpx", reason="httpx needed for FastAPI TestClient")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "StorageIQ_Dashboard"))
import main as dashboard  # noqa: E402
from auto_refresh import published_quarters, quarter_end  # noqa: E402

client = TestClient(dashboard.app)


def _csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def test_analyze_happy_path(fleet_df):
    r = client.post("/api/analyze", files={"file": ("fleet.csv", _csv_bytes(fleet_df), "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kpis"]["days"] == len(fleet_df)
    assert d["kpis"]["drives"] > 0
    assert len(d["forecast"]["afr_pct"]) == 30
    assert all(v >= 0 for v in d["forecast"]["afr_pct"])
    assert d["risk"]["var95_failures"] <= d["risk"]["cvar95_failures"]


def test_analyze_missing_columns(fleet_df):
    bad = fleet_df.rename(columns={"Failures": "Deaths"})
    r = client.post("/api/analyze", files={"file": ("fleet.csv", _csv_bytes(bad), "text/csv")})
    assert r.status_code == 422
    assert "Failures" in r.json()["detail"]


def test_analyze_not_a_csv():
    r = client.post("/api/analyze", files={"file": ("x.csv", b"\x00\x01\x02 not csv", "text/csv")})
    assert r.status_code in (400, 422)


def test_analyze_too_few_rows(fleet_df):
    r = client.post("/api/analyze",
                    files={"file": ("fleet.csv", _csv_bytes(fleet_df.head(10)), "text/csv")})
    assert r.status_code == 422
    assert "60" in r.json()["detail"]


def test_analyze_rejects_oversize():
    r = client.post("/api/analyze",
                    files={"file": ("big.csv", b"a" * (dashboard.MAX_UPLOAD_BYTES + 1), "text/csv")})
    assert r.status_code == 413


# ── auto-refresh quarter logic ───────────────────────────────────────
def test_published_quarters_respects_lag():
    # Feb 1 2026: Q4 2025 ended 32 days ago (< 40-day lag) → newest must be Q3 2025
    qs = published_quarters(date(2026, 2, 1))
    assert qs[0] == (3, 2025)
    # Mar 1 2026: Q4 2025 ended 60 days ago → now newest
    assert published_quarters(date(2026, 3, 1))[0] == (4, 2025)
    # newest-first, contiguous
    assert qs[1] == (2, 2025) and qs[2] == (1, 2025)


def test_quarter_end():
    assert quarter_end(1, 2025) == date(2025, 3, 31)
    assert quarter_end(4, 2024) == date(2024, 12, 31)
