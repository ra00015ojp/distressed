"""
Distressed Debt Entry Signal Dashboard v3 — CORRECTED
=======================================================
Fixes applied:
  ✅ OAS unit fix: FRED returns % → multiplied ×100 → displayed in bps
  ✅ Signal 3: Fed Funds replaced by Yield Curve 10Y-2Y (FRED T10Y2Y)
       Green flag: spread < 2% AND steepening (stress peak indicator)

Automation:
  - Signal 2: Spread-based default rate from FRED CCC/HY
  - Signal 5: SEC 8-K filing scraping + P/NAV metric tracking

Requirements:
    pip install streamlit pandas requests yfinance plotly python-dateutil numpy beautifulsoup4 feedparser

Run:
    streamlit run distressed_dashboard_v3_corrected.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Distressed Credit Dashboard v3 (Corrected)",
    page_icon="📡",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS & THRESHOLDS
# ─────────────────────────────────────────────
# FRED series
HY_OAS_SERIES        = "BAMLH0A0HYM2"   # Returns %, multiplied ×100 → bps
CCC_OAS_SERIES       = "BAMLH0A0HYC"    # Returns %, multiplied ×100 → bps
YIELD_CURVE_SERIES   = "T10Y2Y"         # 10Y minus 2Y Treasury spread (%)
                                         # REPLACES: FED_RATE_SERIES = "DFF"

# BDC tickers and NAV
BDC_NAV = {
    "ARCC": 19.24,
    "FSK":  23.81,
    "OBDC": 15.33,
}

# Target managers for SEC gate event scraping
BDC_MANAGERS = {
    "Ares Capital": {
        "cik": "0001564590",
        "tickers": ["ARCC"],
        "fund_names": ["Ares Capital Corporation"],
    },
    "Apollo": {
        "cik": "0001668700",
        "tickers": ["APO"],
        "fund_names": ["Apollo Strategic Growth Capital", "Apollo Tactical Income"],
    },
    "Blue Owl": {
        "cik": "0001565280",
        "tickers": ["OBDC"],
        "fund_names": ["Blue Owl Capital Corporation"],
    },
    "Blackstone": {
        "cik": "0001393110",
        "tickers": ["BX"],
        "fund_names": ["Blackstone Credit"],
    },
}

# Gate event keywords
GATE_KEYWORDS = [
    "repurchase offer",
    "redemption cap",
    "5% threshold",
    "exceeded aggregate offering",
    "liquidity constraint",
    "gating",
    "gated",
    "suspension of redemptions",
    "pro-rata",
    "tender offer",
    "share repurchase",
]

# Signal thresholds
# OAS thresholds in bps (FRED % × 100)
OAS_THRESHOLDS = {
    "green": 500,   # 500 bps = 5.00%
    "amber": 350,   # 350 bps = 3.50%
}
# Current market context note:
# HY OAS at 270 bps (2.70%) → correctly flagged RED (below amber 350 bps)
# Previously appeared as "3 bps" due to missing ×100 conversion

YIELD_CURVE_GREEN_THRESHOLD = 2.0       # 10Y-2Y < 2% = green flag zone
DEFAULT_RATE_GREEN           = 5.5
BDC_DISCOUNT_GREEN           = 15.0
GATE_EVENTS_GREEN            = 3

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    fred_key = st.text_input(
        "FRED API Key",
        type="password",
        help="Free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        placeholder="Paste your FRED API key",
    )

    st.markdown("---")
    st.subheader("Signal 2: Default Rate")

    override_defaults = st.checkbox(
        "Override automated default rate?",
        value=False,
    )
    if override_defaults:
        manual_default_rate = st.number_input(
            "Manual Default Rate (%)",
            min_value=0.0, max_value=20.0, value=3.2, step=0.1,
        )
    else:
        manual_default_rate = None

    st.markdown("---")
    st.subheader("Signal 5: Gate Events")

    override_gates = st.checkbox(
        "Override automated gate count?",
        value=False,
    )
    if override_gates:
        manual_gate_count = st.number_input(
            "Manual YTD Gate Events",
            min_value=0, max_value=20, value=2, step=1,
        )
    else:
        manual_gate_count = None

    st.markdown("---")
    st.subheader("Signal 6: Maturity Wall")

    maturity_wall_firing = st.checkbox(
        "2027 maturity wall restructurings announced?",
        value=False,
    )

    st.markdown("---")
    st.caption("Last refresh: " + datetime.now().strftime("%d %b %Y  %H:%M UTC"))


# ─────────────────────────────────────────────
# DATA FETCHING — FRED
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fred(series_id: str, api_key: str, days: int = 365) -> pd.DataFrame:
    """Fetch FRED time series. Values returned as published (% for OAS/yields)."""
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}"
        f"&observation_start={start}&observation_end={end}"
        f"&file_type=json&sort_order=asc"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()["observations"]
        df = pd.DataFrame(data)[["date", "value"]].copy()
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "value"])


# ─────────────────────────────────────────────
# SIGNAL 2 — DEFAULT RATE AUTOMATION
# ─────────────────────────────────────────────
def compute_default_rate_from_spreads(
    hy_oas_df: pd.DataFrame, ccc_oas_df: pd.DataFrame
) -> tuple:
    """
    Estimate default rate from CCC/HY spread differential.
    Note: inputs are raw FRED % values (pre-bps conversion).
    Spread diff remains in % — coefficient adjusted accordingly.
    """
    if hy_oas_df.empty or ccc_oas_df.empty:
        return None, None, 0.0

    merged = pd.merge(
        hy_oas_df[["date", "value"]].rename(columns={"value": "hy_oas"}),
        ccc_oas_df[["date", "value"]].rename(columns={"value": "ccc_oas"}),
        on="date", how="inner"
    )
    if merged.empty:
        return None, None, 0.0

    merged = merged.sort_values("date")
    merged["spread_diff"] = merged["ccc_oas"] - merged["hy_oas"]  # in %

    current_spread_diff = merged["spread_diff"].iloc[-1]
    # Coefficient 0.3 (was 0.003 pre-fix) — spread_diff now in % not bps
    estimated_default_rate = max(0.5 + (current_spread_diff * 0.3), 0.1)

    if len(merged) >= 28:
        recent_4w  = merged.tail(20)["spread_diff"].mean()
        older_12w  = (
            merged.iloc[-60:-28]["spread_diff"].mean()
            if len(merged) >= 60
            else merged["spread_diff"].mean()
        )
        is_accelerating = recent_4w > older_12w
    else:
        is_accelerating = None

    confidence = min(100.0, 80.0 + (len(merged) / 365 * 20))

    return estimated_default_rate, is_accelerating, confidence


# ─────────────────────────────────────────────
# SIGNAL 5 — GATE EVENT AUTOMATION (SEC EDGAR)
# ─────────────────────────────────────────────
@st.cache_data(ttl=7200)
def fetch_sec_8k_filings(cik: str, days: int = 365) -> pd.DataFrame:
    """Fetch 8-K filings from SEC EDGAR for a given CIK."""
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days)

    url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}&type=8-K"
        f"&dateb=&owner=exclude&count=100&output=json"
    )

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()

        if "filings" not in data or "files" not in data["filings"]:
            return pd.DataFrame(columns=["date", "accession", "url"])

        results = []
        for filing in data["filings"]["files"]:
            filing_date = datetime.strptime(filing["filingDate"], "%Y-%m-%d")
            if filing_date < start_date:
                continue
            results.append({
                "date":      filing_date,
                "accession": filing["accessionNumber"],
                "url": (
                    f"https://www.sec.gov/cgi-bin/viewer"
                    f"?action=view&cik={cik}"
                    f"&accession_number={filing['accessionNumber']}&xbrl_type=v"
                ),
            })
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame(columns=["date", "accession", "url"])


def extract_gate_keywords_from_filing(accession: str, cik: str) -> bool:
    """Scan 8-K text for gate keywords. Returns True if found."""
    try:
        doc_url = (
            f"https://www.sec.gov/cgi-bin/viewer"
            f"?action=view&cik={cik}&accession_number={accession}"
        )
        r = requests.get(doc_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return False
        text = r.text.lower()
        return any(kw.lower() in text for kw in GATE_KEYWORDS)
    except Exception:
        return False


def detect_new_gate_events(bdc_managers: dict, days: int = 365) -> tuple:
    """
    Layer 1: SEC 8-K keyword scan.
    Returns (gate_count, details_list, confidence).
    """
    gate_events = []

    for manager_name, manager_data in bdc_managers.items():
        cik      = manager_data["cik"]
        filings  = fetch_sec_8k_filings(cik, days=days)

        for _, filing in filings.iterrows():
            if extract_gate_keywords_from_filing(filing["accession"], cik):
                gate_events.append({
                    "date":    filing["date"],
                    "manager": manager_name,
                    "source":  "SEC 8-K",
                    "url":     filing["url"],
                })

    unique_gates = {}
    for event in gate_events:
        key = (event["date"].date(), event["manager"])
        if key not in unique_gates:
            unique_gates[key] = event

    gate_list  = list(unique_gates.values())
    gate_count = len(gate_list)
    confidence = min(100, 70 + (gate_count * 5))

    return gate_count, gate_list, confidence


# ─────────────────────────────────────────────
# BDC PRICE & NAV
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_bdc_prices(tickers: list) -> dict:
    result = {}
    for t in tickers:
        try:
            tk   = yf.Ticker(t)
            hist = tk.history(period="1y")[["Close"]].reset_index()
            hist.columns = ["date", "value"]
            hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
            result[t] = {
                "history": hist.sort_values("date"),
                "latest":  float(hist["value"].iloc[-1]) if not hist.empty else None,
            }
        except Exception:
            result[t] = {"history": pd.DataFrame(), "latest": None}
    return result


def compute_bdc_discount(bdc_prices: dict, bdc_nav: dict) -> float:
    discounts = []
    for ticker, nav in bdc_nav.items():
        price = bdc_prices.get(ticker, {}).get("latest")
        if price and nav:
            discounts.append((nav - price) / nav * 100)
    return sum(discounts) / len(discounts) if discounts else 0.0


# ─────────────────────────────────────────────
# SIGNAL EVALUATION
# ─────────────────────────────────────────────
def oas_status(oas_bps: float):
    """
    oas_bps: already converted to bps (FRED % × 100).
    Example: FRED 2.70% → 270 bps → correctly RED below 350 bps amber.
    """
    if oas_bps >= OAS_THRESHOLDS["green"]:
        return "🟢", "GREEN", f"{oas_bps:.0f} bps — Strong entry signal"
    elif oas_bps >= OAS_THRESHOLDS["amber"]:
        return "🟡", "AMBER", f"{oas_bps:.0f} bps — Begin accumulating"
    else:
        return "🔴", "RED",   f"{oas_bps:.0f} bps — Too tight, complacent"


def default_status(rate: float, accelerating, confidence: float):
    conf_label = (
        "🔒 HIGH" if confidence >= 80
        else ("🟡 MED" if confidence >= 60 else "⚠️  LOW")
    )
    if rate >= DEFAULT_RATE_GREEN and not accelerating:
        return "🟢", "GREEN", f"{rate:.1f}% — Plateau reached ({conf_label})"
    elif rate >= DEFAULT_RATE_GREEN and accelerating:
        return "🟡", "AMBER", f"{rate:.1f}% — Rising toward trigger ({conf_label})"
    elif rate >= 3.0 and accelerating:
        return "🟡", "AMBER", f"{rate:.1f}% — Accelerating ({conf_label})"
    else:
        return "🔴", "RED",   f"{rate:.1f}% — Still low/early ({conf_label})"


def yield_curve_status(yc_df: pd.DataFrame):
    """
    REPLACES fed_status().
    Signal 3: 10Y-2Y Treasury spread.
    Green flag: spread < 2% AND steepening (resolving inversion = stress peak).
    FRED T10Y2Y returns % directly — no unit conversion needed.
    """
    if yc_df.empty:
        return "⚪", "UNKNOWN", "No data — enter FRED key"
    current    = yc_df["value"].iloc[-1]
    recent_60  = yc_df.tail(60)["value"]
    steepening = recent_60.iloc[-1] > recent_60.iloc[0]
    trend      = "📈 steepening" if steepening else "📉 flattening"
    label      = f"10Y-2Y: {current:.2f}% ({trend})"

    if current < YIELD_CURVE_GREEN_THRESHOLD and steepening:
        return "🟢", "GREEN", f"{label} — Stress peak signal ✅"
    elif current < YIELD_CURVE_GREEN_THRESHOLD:
        return "🟡", "AMBER", f"{label} — Sub-2% but not steepening yet"
    else:
        return "🔴", "RED",   f"{label} — Above 2%, no distress signal"


def bdc_status(avg_discount: float):
    if avg_discount >= BDC_DISCOUNT_GREEN:
        return "🟢", "GREEN", f"{avg_discount:.1f}% avg discount — Stress priced"
    elif avg_discount >= 8:
        return "🟡", "AMBER", f"{avg_discount:.1f}% avg discount — Widening"
    elif avg_discount >= 0:
        return "🔴", "RED",   f"{avg_discount:.1f}% avg discount — Near par"
    else:
        return "🔴", "RED",   "Trading at premium"


def gate_status(events: int, confidence: float):
    conf_label = f" ({confidence:.0f}% confidence)" if confidence < 100 else ""
    if events >= GATE_EVENTS_GREEN:
        return "🟢", "GREEN", f"{events} events — Capitulation approaching{conf_label}"
    elif events >= 2:
        return "🟡", "AMBER", f"{events} events — Stress building{conf_label}"
    else:
        return "🔴", "RED",   f"{events} events — Too few to signal{conf_label}"


def maturity_status(firing: bool):
    if firing:
        return "🟢", "GREEN", "2027 restructurings announced — Wall biting"
    else:
        return "🔴", "RED",   "No major 2027 restructurings yet"


def overall_recommendation(green_count: int) -> tuple:
    if green_count >= 5:
        return (
            "🚀 DEPLOY CAPITAL", "green",
            "5–6 signals green. Allocate planned position in full across 1–2 tranches.",
        )
    elif green_count >= 4:
        return (
            "📈 ACCUMULATE", "orange",
            "4 signals green. Begin meaningful accumulation — 50–60% of planned allocation now.",
        )
    elif green_count >= 2:
        return (
            "🟡 WATCH & WAIT", "yellow",
            "2–3 signals green. Deploy first small tranche (£2–4k) if OAS crosses 500 bps.",
        )
    else:
        return (
            "🔴 STAY IN CASH", "red",
            "Fewer than 2 signals green. Park capital in T-bills. Revisit in 60 days.",
        )


# ─────────────────────────────────────────────
# CHARTING
# ─────────────────────────────────────────────
def spark_chart(df: pd.DataFrame, title: str, color: str,
                threshold_lines: list = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"],
        mode="lines", name=title,
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
    ))
    if threshold_lines:
        for level, label, tcolor in threshold_lines:
            fig.add_hline(
                y=level, line_dash="dash", line_color=tcolor, opacity=0.7,
                annotation_text=label, annotation_position="right",
            )
    fig.update_layout(
        height=220, margin=dict(l=0, r=60, t=30, b=0),
        title=dict(text=title, font=dict(size=13)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#ffffff"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2a2d35"),
        showlegend=False,
    )
    return fig


def bdc_chart(bdc_prices: dict, bdc_nav: dict) -> go.Figure:
    fig    = go.Figure()
    colors = ["#4fc3f7", "#aed581", "#ffb74d"]
    for (ticker, nav), color in zip(bdc_nav.items(), colors):
        hist = bdc_prices.get(ticker, {}).get("history", pd.DataFrame())
        if hist.empty:
            continue
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["value"],
            mode="lines", name=f"{ticker} price",
            line=dict(color=color, width=2),
        ))
        fig.add_hline(
            y=nav, line_dash="dot", line_color=color, opacity=0.5,
            annotation_text=f"{ticker} NAV ${nav}",
            annotation_position="right",
        )
    fig.update_layout(
        height=220, margin=dict(l=0, r=90, t=30, b=0),
        title=dict(text="BDC Prices vs NAV (dashed = NAV)", font=dict(size=13)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#ffffff"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2a2d35"),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def spread_chart(hy_df: pd.DataFrame, ccc_df: pd.DataFrame) -> go.Figure:
    """
    Chart uses raw FRED % values (not bps) for visual continuity.
    Y-axis labelled in % with bps equivalents in annotations.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hy_df["date"], y=hy_df["value"],
        mode="lines", name="HY OAS (%)",
        line=dict(color="#4fc3f7", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ccc_df["date"], y=ccc_df["value"],
        mode="lines", name="CCC OAS (%)",
        line=dict(color="#ef5350", width=2),
    ))
    fig.add_hline(
        y=5.0, line_dash="dash", line_color="#4caf50", opacity=0.7,
        annotation_text="5.0% = 500 bps 🟢 Entry", annotation_position="right",
    )
    fig.add_hline(
        y=3.5, line_dash="dash", line_color="#ff9800", opacity=0.7,
        annotation_text="3.5% = 350 bps 🟡 Accumulate", annotation_position="right",
    )
    fig.update_layout(
        height=220, margin=dict(l=0, r=120, t=30, b=0),
        title=dict(text="Credit Spread Divergence — CCC vs HY (FRED %, thresholds in bps)", font=dict(size=13)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#ffffff"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2a2d35", title="Spread (%)"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.title("📡 Distressed Credit Entry Dashboard v3 — Corrected")
st.caption(
    "Fixes: OAS unit (% → bps ×100) | Signal 3: Yield Curve 10Y-2Y replaces Fed Funds. "
    "Built April 2026 · Corrected Sept 2026."
)
st.markdown("---")

# ── Fetch data ──────────────────────────────
if not fred_key:
    st.warning(
        "⚠️  Enter your free FRED API key in the sidebar. "
        "Get one at **https://fred.stlouisfed.org/docs/api/api_key.html**"
    )
    oas_df = pd.DataFrame(columns=["date", "value"])
    ccc_df = pd.DataFrame(columns=["date", "value"])
    yc_df  = pd.DataFrame(columns=["date", "value"])
else:
    with st.spinner("Fetching FRED data…"):
        oas_df = fetch_fred(HY_OAS_SERIES,      fred_key, days=730)
        ccc_df = fetch_fred(CCC_OAS_SERIES,     fred_key, days=730)
        yc_df  = fetch_fred(YIELD_CURVE_SERIES, fred_key, days=730)

with st.spinner("Fetching BDC prices…"):
    bdc_prices = fetch_bdc_prices(list(BDC_NAV.keys()))

# ── Signal 2: Default Rate ──────────────────
if override_defaults and manual_default_rate is not None:
    default_rate        = manual_default_rate
    default_accelerating = False
    default_confidence  = 100.0
    default_source      = "🟢 Manual override"
else:
    default_rate, default_accelerating, default_confidence = compute_default_rate_from_spreads(
        oas_df, ccc_df
    )
    if default_rate is None:
        default_rate        = 0.0
        default_accelerating = False
        default_confidence  = 0.0
    default_source = f"🤖 Automated (CCC/HY spread, {default_confidence:.0f}% conf)"

# ── Signal 5: Gate Events ───────────────────
if override_gates and manual_gate_count is not None:
    gate_events    = manual_gate_count
    gate_confidence = 100.0
    gate_details   = []
    gate_source    = "🟢 Manual override"
else:
    with st.spinner("Scanning SEC EDGAR for gate events…"):
        gate_events, gate_details, gate_confidence = detect_new_gate_events(
            BDC_MANAGERS, days=365
        )
    gate_source = f"🤖 Automated (SEC 8-K scan, {gate_confidence:.0f}% conf)"

# ── Current values ──────────────────────────
# OAS: FRED returns % → multiply ×100 for bps comparison against thresholds
current_oas_bps = (
    float(oas_df["value"].iloc[-1]) * 100
    if not oas_df.empty else None
)
avg_bdc_disc = compute_bdc_discount(bdc_prices, BDC_NAV)

# ── Evaluate signals ─────────────────────────
s1_icon, s1_state, s1_detail = (
    oas_status(current_oas_bps)
    if current_oas_bps
    else ("⚪", "NO DATA", "Enter FRED key")
)
s2_icon, s2_state, s2_detail = default_status(default_rate, default_accelerating, default_confidence)
s3_icon, s3_state, s3_detail = yield_curve_status(yc_df)      # ← replaced fed_status
s4_icon, s4_state, s4_detail = bdc_status(avg_bdc_disc)
s5_icon, s5_state, s5_detail = gate_status(gate_events, gate_confidence)
s6_icon, s6_state, s6_detail = maturity_status(maturity_wall_firing)

green_count = sum(
    1 for state in [s1_state, s2_state, s3_state, s4_state, s5_state, s6_state]
    if state == "GREEN"
)

rec_label, rec_color, rec_text = overall_recommendation(green_count)

# ─────────────────────────────────────────────
# OVERALL RECOMMENDATION
# ─────────────────────────────────────────────
color_map = {
    "green": "#1b5e20", "orange": "#e65100",
    "yellow": "#f9a825", "red": "#b71c1c",
}
bg = color_map.get(rec_color, "#1a1a2e")

st.markdown(
    f"""
    <div style="background:{bg};padding:24px 28px;border-radius:12px;margin-bottom:8px;">
        <h2 style="margin:0;color:#ffffff;font-size:2rem;">{rec_label}</h2>
        <p style="margin:8px 0 0;color:#ffffffcc;font-size:1.05rem;">{rec_text}</p>
        <p style="margin:8px 0 0;color:#ffffff99;font-size:0.9rem;">
            Signals active: <strong style="color:#fff">{green_count} / 6 GREEN</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ─────────────────────────────────────────────
# SIX-SIGNAL CHECKLIST
# ─────────────────────────────────────────────
st.subheader("🚦 Six-Signal Checklist")

signals = [
    ("HY OAS Spread",            s1_icon, s1_state, s1_detail, "Cross and hold above 500 bps"),
    ("Default Rate Trajectory",  s2_icon, s2_state, s2_detail, "Rate stabilises at 5–7% plateau"),
    ("Yield Curve 10Y-2Y",       s3_icon, s3_state, s3_detail, "Spread <2% + steepening trend"),   # ← updated
    ("BDC Discount to NAV",      s4_icon, s4_state, s4_detail, "Sector average exceeds 15%"),
    ("Redemption Gate Events",   s5_icon, s5_state, s5_detail, "Third or fourth top-tier event"),
    ("2027 Maturity Wall",       s6_icon, s6_state, s6_detail, "Major restructurings announced"),
]

col_labels = st.columns([2, 0.6, 1, 3, 3])
col_labels[0].markdown("**Signal**")
col_labels[1].markdown("**Status**")
col_labels[2].markdown("**State**")
col_labels[3].markdown("**Current Reading**")
col_labels[4].markdown("**Green Light Trigger**")

for name, icon, state, detail, trigger in signals:
    cols = st.columns([2, 0.6, 1, 3, 3])
    cols[0].markdown(f"**{name}**")
    cols[1].markdown(icon)
    state_color = {"GREEN": "green", "AMBER": "orange", "RED": "red"}.get(state, "grey")
    cols[2].markdown(
        f"<span style='color:{state_color};font-weight:bold'>{state}</span>",
        unsafe_allow_html=True,
    )
    cols[3].markdown(detail)
    cols[4].markdown(f"<span style='color:#888'>{trigger}</span>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# AUTOMATION DETAILS — SIGNAL 2 & 5
# ─────────────────────────────────────────────
col_auto1, col_auto2 = st.columns(2)

with col_auto1:
    with st.expander("🤖 Signal 2: Default Rate Automation"):
        st.markdown(f"**Source:** {default_source}")
        st.markdown(f"**Computed Rate:** {default_rate:.2f}%")
        st.markdown(f"**Acceleration:** {'📈 Accelerating' if default_accelerating else '📉 Stabilising'}")
        st.markdown(f"**Confidence:** {default_confidence:.0f}%")
        st.caption("Estimates from CCC/HY spread ratio. Validate monthly vs Moody's releases.")

with col_auto2:
    with st.expander("🤖 Signal 5: Gate Event Automation"):
        st.markdown(f"**Source:** {gate_source}")
        st.markdown(f"**Detected Events:** {gate_events}")
        st.markdown(f"**Confidence:** {gate_confidence:.0f}%")
        if gate_details:
            st.markdown("**Recent Gate Events Detected:**")
            for event in gate_details[-5:]:
                st.markdown(
                    f"- **{event['manager']}** ({event['date'].strftime('%Y-%m-%d')}) "
                    f"— [SEC Filing]({event['url']})"
                )
        else:
            st.markdown("No gate events detected in SEC filings (past 365 days)")
        st.caption(
            "Scans SEC 8-K filings for: 'redemption cap', 'gating', 'suspension of redemptions' etc."
        )

st.markdown("---")

# ─────────────────────────────────────────────
# UNIT NOTE
# ─────────────────────────────────────────────
with st.expander("ℹ️ OAS Unit Clarification"):
    st.markdown("""
    **Why FRED shows 2.70 but dashboard displays 270 bps:**

    | Source | Raw value | Unit | Conversion | Dashboard display |
    |--------|-----------|------|------------|-------------------|
    | FRED `BAMLH0A0HYM2` | 2.70 | % | × 100 | **270 bps** |
    | Green threshold | 5.00 | % | × 100 | **500 bps** |
    | Amber threshold | 3.50 | % | × 100 | **350 bps** |

    Current HY OAS of ~270 bps correctly triggers 🔴 RED (below amber 350 bps).
    The original code omitted ×100, displaying 2.70 as "2 bps" — now fixed.
    """)

st.markdown("---")

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
st.subheader("📈 Live Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if not oas_df.empty:
        # Chart in % (FRED native) with bps annotations
        fig_oas = spark_chart(
            oas_df, "HY OAS Spread (FRED % — 270 bps = 2.70%)", "rgb(79,195,247)",
            threshold_lines=[
                (5.0, "🟢 5.0% = 500 bps Entry",     "#4caf50"),
                (3.5, "🟡 3.5% = 350 bps Accumulate", "#ff9800"),
            ],
        )
        st.plotly_chart(fig_oas, use_container_width=True)
    else:
        st.info("OAS chart requires FRED API key.")

with chart_col2:
    if not yc_df.empty:
        fig_yc = spark_chart(
            yc_df, "Yield Curve 10Y-2Y Spread (%)", "rgb(174,213,129)",
            threshold_lines=[
                (2.0, "🟢 <2% Green flag zone",  "#4caf50"),
                (0.0, "Inversion line",           "#ef5350"),
            ],
        )
        st.plotly_chart(fig_yc, use_container_width=True)
    else:
        st.info("Yield curve chart requires FRED API key.")

if not oas_df.empty and not ccc_df.empty:
    st.plotly_chart(spread_chart(oas_df, ccc_df), use_container_width=True)

bdc_has_data = any(not v["history"].empty for v in bdc_prices.values())
if bdc_has_data:
    st.plotly_chart(bdc_chart(bdc_prices, BDC_NAV), use_container_width=True)
else:
    st.warning("BDC price data unavailable.")

st.markdown("---")

# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
st.subheader("💼 Position Sizing Guide (£100k spare cash)")

sizing_data = {
    "Scenario":  ["0–1 green", "2–3 green", "4 green", "5–6 green"],
    "Signals":   ["Stay out",  "Watch",     "Accumulate", "Deploy"],
    "FALN %":    ["0%",        "2–5%",      "8–12%",      "12–15%"],
    "FALN £":    ["£0",        "£2–5k",     "£8–12k",     "£12–15k"],
    "Remainder": [
        "100% T-bills",
        "95–98% T-bills",
        "T-bills; keep £20–30k dry powder",
        "T-bills; review at 6 months",
    ],
}
df_sizing = pd.DataFrame(sizing_data)

def highlight_current(row):
    green_ranges = {
        "0–1 green": green_count <= 1,
        "2–3 green": 2 <= green_count <= 3,
        "4 green":   green_count == 4,
        "5–6 green": green_count >= 5,
    }
    if green_ranges.get(row["Scenario"], False):
        return ["background-color: #1b3a2a; font-weight: bold"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_sizing.style.apply(highlight_current, axis=1),
    use_container_width=True, hide_index=True,
)

st.markdown("---")

# ─────────────────────────────────────────────
# RAW DATA
# ─────────────────────────────────────────────
with st.expander("📋 Raw Data & Validation"):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["HY OAS", "CCC OAS", "Yield Curve", "Default Rate", "Gate Events", "BDC Prices"]
    )
    with tab1:
        st.caption("Raw FRED % values. Multiply ×100 for bps.")
        st.dataframe(oas_df.tail(30), use_container_width=True, hide_index=True)
    with tab2:
        st.caption("Raw FRED % values. Multiply ×100 for bps.")
        st.dataframe(ccc_df.tail(30), use_container_width=True, hide_index=True)
    with tab3:
        st.caption("10Y-2Y spread in %. Positive = normal curve. Negative = inverted.")
        st.dataframe(yc_df.tail(30), use_container_width=True, hide_index=True)
    with tab4:
        inputs_df = pd.DataFrame({
            "Parameter": ["Default Rate (%)", "Accelerating", "Confidence (%)", "Source"],
            "Value": [
                f"{default_rate:.2f}%",
                "Yes" if default_accelerating else "No",
                f"{default_confidence:.0f}%",
                default_source,
            ],
        })
        st.dataframe(inputs_df, use_container_width=True, hide_index=True)
    with tab5:
        if gate_details:
            gate_df = pd.DataFrame(gate_details).sort_values("date", ascending=False)
            st.dataframe(gate_df[["date", "manager", "source"]], use_container_width=True, hide_index=True)
        else:
            st.markdown("No gate events detected in the past 365 days.")
    with tab6:
        rows = []
        for ticker, nav in BDC_NAV.items():
            price = bdc_prices.get(ticker, {}).get("latest")
            disc  = (nav - price) / nav * 100 if price else None
            rows.append({
                "Ticker":        ticker,
                "NAV/share ($)": nav,
                "Price ($)":     round(price, 2) if price else "N/A",
                "Discount (%)":  f"{disc:.1f}%" if disc else "N/A",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "**Disclaimer:** Educational purposes only. Not regulated financial advice. "
    "Data: FRED, SEC EDGAR, Yahoo Finance | Corrected Sept 2026"
)
