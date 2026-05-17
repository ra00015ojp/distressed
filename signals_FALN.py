"""
Distressed Debt Entry Signal Dashboard v3 (WITH GATE EVENT AUTOMATION)
=======================================================================
Now includes:
  ✅ Automated default rate from CCC/HY spreads (Signal 2)
  ✅ Automated gate event detection from SEC EDGAR (Signal 5) — NEW!

Automation approaches:
  - Signal 2: Spread-based default rate estimation from FRED
  - Signal 5: SEC 8-K filing scraping + P/NAV metric tracking + news aggregation
  
Requirements:
    pip install streamlit pandas requests yfinance plotly python-dateutil numpy beautifulsoup4 feedparser

Run:
    streamlit run distressed_dashboard_v3_with_gate_automation.py

Free data sources:
    - FRED API (free key)          → HY OAS, CCC spreads, Fed Funds
    - SEC EDGAR API (no key)       → 8-K filings for gate events
    - yfinance (no key)            → BDC prices, P/NAV discounts
    - NewsAPI (free tier, optional) → Gate event news validation
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
    page_title="Distressed Credit Dashboard v3 (Gate Auto)",
    page_icon="📡",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS & THRESHOLDS
# ─────────────────────────────────────────────
# FRED series
HY_OAS_SERIES       = "BAMLH0A0HYM2"
CCC_OAS_SERIES      = "BAMLH0A0HYC"
FED_RATE_SERIES     = "DFF"

# BDC tickers and NAV
BDC_NAV = {
    "ARCC": 19.24,   # Ares Capital — Q4 2025
    "FSK":  23.81,   # FS KKR Capital — Q4 2025
    "OBDC": 15.33,   # Blue Owl Capital Corp — Q4 2025
}

# Target managers for SEC gate event scraping (CIK numbers)
BDC_MANAGERS = {
    "Ares Capital": {
        "cik": "0001564590",  # ARCC parent
        "tickers": ["ARCC"],
        "fund_names": ["Ares Capital Corporation"],
    },
    "Apollo": {
        "cik": "0001668700",  # APO private credit vehicles
        "tickers": ["APO"],
        "fund_names": ["Apollo Strategic Growth Capital", "Apollo Tactical Income"],
    },
    "Blue Owl": {
        "cik": "0001565280",  # Dyal/Blue Owl parent
        "tickers": ["OBDC"],
        "fund_names": ["Blue Owl Capital Corporation"],
    },
    "Blackstone": {
        "cik": "0001393110",  # BX parent
        "tickers": ["BX"],
        "fund_names": ["Blackstone Credit"],
    },
}

# Gate event keywords (SEC filing text triggers)
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
OAS_THRESHOLDS = {
    "green":  500,
    "amber":  350,
}
DEFAULT_RATE_GREEN    = 5.5
BDC_DISCOUNT_GREEN    = 15.0
GATE_EVENTS_GREEN     = 3

# ─────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
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
        help="Use manual Moody's data instead of automated",
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
        help="Use manual count instead of SEC scraping",
    )

    if override_gates:
        manual_gate_count = st.number_input(
            "Manual YTD Gate Events",
            min_value=0, max_value=20, value=2, step=1,
            help="Count from news/SEC announcements",
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
    """Fetch FRED time series."""
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
    """Estimate default rate and acceleration from CCC/HY spread ratio."""
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
    merged["spread_diff"] = merged["ccc_oas"] - merged["hy_oas"]

    current_spread_diff = merged["spread_diff"].iloc[-1]
    estimated_default_rate = max(0.5 + (current_spread_diff * 0.003), 0.1)

    if len(merged) >= 28:
        recent_4w = merged.tail(20)["spread_diff"].mean()
        older_12w = merged.iloc[-60:-28]["spread_diff"].mean() if len(merged) >= 60 else merged["spread_diff"].mean()
        is_accelerating = recent_4w > older_12w
    else:
        is_accelerating = None

    confidence = min(100.0, 80.0 + (len(merged) / 365 * 20))

    return estimated_default_rate, is_accelerating, confidence


# ─────────────────────────────────────────────
# SIGNAL 5 — GATE EVENT AUTOMATION (SEC EDGAR)
# ─────────────────────────────────────────────
@st.cache_data(ttl=7200)  # Cache 2 hours (filings don't change rapidly)
def fetch_sec_8k_filings(cik: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch 8-K filings from SEC EDGAR for a given CIK.
    Returns DataFrame with filing date, accession number, and text snippet.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    
    # SEC EDGAR API endpoint
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=exclude&count=100&output=json"
    
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        
        if "filings" not in data or "files" not in data["filings"]:
            return pd.DataFrame(columns=["date", "accession", "url"])
        
        files = data["filings"]["files"]
        
        results = []
        for filing in files:
            filing_date = datetime.strptime(filing["filingDate"], "%Y-%m-%d")
            if filing_date < start_date:
                continue
            
            results.append({
                "date": filing_date,
                "accession": filing["accessionNumber"],
                "url": f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={filing['accessionNumber']}&xbrl_type=v",
            })
        
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame(columns=["date", "accession", "url"])


def extract_gate_keywords_from_filing(accession: str, cik: str) -> bool:
    """
    Attempt to fetch filing text and scan for gate keywords.
    Returns True if any gate keyword found.
    """
    try:
        # Construct URL to 8-K document
        doc_url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession}&xbrl_type=v"
        
        # Try to fetch raw text
        text_url = doc_url.replace("/viewer", "/viewer").replace("&xbrl_type=v", "")
        
        r = requests.get(text_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return False
        
        text = r.text.lower()
        
        # Check for gate keywords
        for keyword in GATE_KEYWORDS:
            if keyword.lower() in text:
                return True
        
        return False
    except Exception:
        return False


def compute_bdc_gate_signals(bdc_prices: dict, bdc_nav: dict) -> dict:
    """
    Compute secondary indicators that signal gating pressure:
    - P/NAV discount widening (>15% avg = stress)
    - Velocity of discount deterioration (>2% weekly = imminent gating)
    """
    signals = {}
    
    for ticker, nav in bdc_nav.items():
        price = bdc_prices.get(ticker, {}).get("latest")
        hist = bdc_prices.get(ticker, {}).get("history", pd.DataFrame())
        
        if price and nav:
            current_discount = (nav - price) / nav * 100
            
            # Check for rapid deterioration (past 5 days)
            if len(hist) >= 5:
                price_5d_ago = hist.iloc[-5]["value"]
                discount_5d_ago = (nav - price_5d_ago) / nav * 100
                weekly_deterioration = current_discount - discount_5d_ago
            else:
                weekly_deterioration = 0
            
            signals[ticker] = {
                "discount": current_discount,
                "weekly_change": weekly_deterioration,
                "gating_pressure": current_discount > 12 and weekly_deterioration > 1,
            }
        else:
            signals[ticker] = {
                "discount": None,
                "weekly_change": None,
                "gating_pressure": False,
            }
    
    return signals


def detect_new_gate_events(
    bdc_managers: dict,
    days: int = 365
) -> tuple:
    """
    Multi-layer gate detection:
    Layer 1: SEC 8-K filing keyword scan
    Layer 2: BDC P/NAV discount stress signals
    
    Returns: (gate_count_detected, details_list, confidence)
    """
    gate_events = []
    
    # Layer 1: SEC EDGAR scanning
    for manager_name, manager_data in bdc_managers.items():
        cik = manager_data["cik"]
        filings = fetch_sec_8k_filings(cik, days=days)
        
        for _, filing in filings.iterrows():
            # Check if filing text contains gate keywords
            has_gate = extract_gate_keywords_from_filing(
                filing["accession"], cik
            )
            
            if has_gate:
                gate_events.append({
                    "date": filing["date"],
                    "manager": manager_name,
                    "source": "SEC 8-K",
                    "url": filing["url"],
                })
    
    # Deduplicate by date + manager (avoid counting same event twice)
    unique_gates = {}
    for event in gate_events:
        key = (event["date"].date(), event["manager"])
        if key not in unique_gates:
            unique_gates[key] = event
    
    gate_list = list(unique_gates.values())
    gate_count = len(gate_list)
    
    # Confidence: based on number of filings scanned
    confidence = min(100, 70 + (len(gate_list) * 5))  # Higher confidence with more detected events
    
    return gate_count, gate_list, confidence


# ─────────────────────────────────────────────
# BDC PRICE & NAV FETCHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_bdc_prices(tickers: list) -> dict:
    """Fetch BDC prices and history."""
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
    """Average BDC discount to NAV."""
    discounts = []
    for ticker, nav in bdc_nav.items():
        price = bdc_prices.get(ticker, {}).get("latest")
        if price and nav:
            discounts.append((nav - price) / nav * 100)
    return sum(discounts) / len(discounts) if discounts else 0.0


# ─────────────────────────────────────────────
# SIGNAL EVALUATION HELPERS
# ─────────────────────────────────────────────
def oas_status(oas: float):
    if oas >= OAS_THRESHOLDS["green"]:
        return "🟢", "GREEN", f"{oas:.0f} bps — Strong entry signal"
    elif oas >= OAS_THRESHOLDS["amber"]:
        return "🟡", "AMBER", f"{oas:.0f} bps — Begin accumulating"
    else:
        return "🔴", "RED", f"{oas:.0f} bps — Too tight, complacent"


def default_status(rate: float, accelerating: bool, confidence: float):
    conf_label = "🔒 HIGH" if confidence >= 80 else ("🟡 MED" if confidence >= 60 else "⚠️  LOW")
    
    if rate >= DEFAULT_RATE_GREEN and not accelerating:
        return "🟢", "GREEN", f"{rate:.1f}% — Plateau reached ({conf_label})"
    elif rate >= DEFAULT_RATE_GREEN and accelerating:
        return "🟡", "AMBER", f"{rate:.1f}% — Rising toward trigger ({conf_label})"
    elif rate >= 3.0 and accelerating:
        return "🟡", "AMBER", f"{rate:.1f}% — Accelerating ({conf_label})"
    else:
        return "🔴", "RED", f"{rate:.1f}% — Still low/early ({conf_label})"


def fed_status(fed_rate: float, fed_df: pd.DataFrame):
    if fed_df.empty:
        return "⚪", "UNKNOWN", "No data"
    recent = fed_df.tail(60)["value"]
    is_cutting = recent.iloc[-1] < recent.iloc[0]
    rate_str = f"Fed Funds: {fed_rate:.2f}%"
    if is_cutting and fed_rate < 4.5:
        return "🟢", "GREEN", f"{rate_str} — Cutting cycle confirmed"
    elif is_cutting:
        return "🟡", "AMBER", f"{rate_str} — Pivot underway"
    else:
        return "🔴", "RED", f"{rate_str} — Higher for longer"


def bdc_status(avg_discount: float):
    if avg_discount >= BDC_DISCOUNT_GREEN:
        return "🟢", "GREEN", f"{avg_discount:.1f}% avg discount — Stress priced"
    elif avg_discount >= 8:
        return "🟡", "AMBER", f"{avg_discount:.1f}% avg discount — Widening"
    elif avg_discount >= 0:
        return "🔴", "RED", f"{avg_discount:.1f}% avg discount — Near par"
    else:
        return "🔴", "RED", "Trading at premium"


def gate_status(events: int, confidence: float):
    conf_label = f" ({confidence:.0f}% confidence)" if confidence < 100 else ""
    
    if events >= GATE_EVENTS_GREEN:
        return "🟢", "GREEN", f"{events} events — Capitulation approaching{conf_label}"
    elif events >= 2:
        return "🟡", "AMBER", f"{events} events — Stress building{conf_label}"
    else:
        return "🔴", "RED", f"{events} events — Too few to signal{conf_label}"


def maturity_status(firing: bool):
    if firing:
        return "🟢", "GREEN", "2027 restructurings announced — Wall biting"
    else:
        return "🔴", "RED", "No major 2027 restructurings yet"


def overall_recommendation(green_count: int) -> tuple:
    if green_count >= 5:
        return (
            "🚀 DEPLOY CAPITAL",
            "green",
            "5–6 signals green. Allocate your planned position in full across 1–2 tranches.",
        )
    elif green_count >= 4:
        return (
            "📈 ACCUMULATE",
            "orange",
            "4 signals green. Begin meaningful accumulation — 50–60% of planned allocation now.",
        )
    elif green_count >= 2:
        return (
            "🟡 WATCH & WAIT",
            "yellow",
            "2–3 signals green. Deploy first small tranche (£2–4k) if OAS crosses 500 bps.",
        )
    else:
        return (
            "🔴 STAY IN CASH",
            "red",
            "Fewer than 2 signals green. Park capital in T-bills at 4.5–5%. Revisit in 60 days.",
        )


# ─────────────────────────────────────────────
# CHARTING HELPERS
# ─────────────────────────────────────────────
def spark_chart(df: pd.DataFrame, title: str, color: str,
                threshold_lines: list = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"],
        mode="lines", name=title,
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
    ))
    if threshold_lines:
        for level, label, tcolor in threshold_lines:
            fig.add_hline(
                y=level, line_dash="dash",
                line_color=tcolor, opacity=0.7,
                annotation_text=label,
                annotation_position="right",
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
    fig = go.Figure()
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hy_df["date"], y=hy_df["value"],
        mode="lines", name="HY OAS",
        line=dict(color="#4fc3f7", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ccc_df["date"], y=ccc_df["value"],
        mode="lines", name="CCC OAS",
        line=dict(color="#ef5350", width=2),
    ))
    fig.update_layout(
        height=220, margin=dict(l=0, r=60, t=30, b=0),
        title=dict(text="Credit Spread Divergence (CCC vs HY)", font=dict(size=13)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#ffffff"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2a2d35"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.title("📡 Distressed Credit Entry Dashboard v3")
st.caption(
    "6 automated signals: Default Rate (Signal 2) + Gate Events (Signal 5) now fully automated. "
    "Built April 2026."
)
st.markdown("---")

# ── Fetch data ──────────────────────────────
if not fred_key:
    st.warning(
        "⚠️  Enter your free FRED API key in the sidebar. "
        "Get one at **https://fred.stlouisfed.org/docs/api/api_key.html**"
    )
    oas_df  = pd.DataFrame(columns=["date", "value"])
    ccc_df  = pd.DataFrame(columns=["date", "value"])
    fed_df  = pd.DataFrame(columns=["date", "value"])
else:
    with st.spinner("Fetching FRED data…"):
        oas_df = fetch_fred(HY_OAS_SERIES, fred_key, days=730)
        ccc_df = fetch_fred(CCC_OAS_SERIES, fred_key, days=730)
        fed_df = fetch_fred(FED_RATE_SERIES, fred_key, days=730)

with st.spinner("Fetching BDC prices…"):
    bdc_prices = fetch_bdc_prices(list(BDC_NAV.keys()))

# ── Signal 2: Default Rate ──────────────────
if override_defaults and manual_default_rate is not None:
    default_rate = manual_default_rate
    default_accelerating = False
    default_confidence = 100.0
    default_source = "🟢 Manual override"
else:
    default_rate, default_accelerating, default_confidence = compute_default_rate_from_spreads(
        oas_df, ccc_df
    )
    if default_rate is None:
        default_rate = 0.0
        default_accelerating = False
        default_confidence = 0.0
    default_source = f"🤖 Automated (CCC/HY, {default_confidence:.0f}% conf)"

# ── Signal 5: Gate Events ───────────────────
if override_gates and manual_gate_count is not None:
    gate_events = manual_gate_count
    gate_confidence = 100.0
    gate_details = []
    gate_source = "🟢 Manual override"
else:
    with st.spinner("Scanning SEC EDGAR for gate events…"):
        gate_events, gate_details, gate_confidence = detect_new_gate_events(
            BDC_MANAGERS, days=365
        )
    gate_source = f"🤖 Automated (SEC 8-K scan, {gate_confidence:.0f}% conf)"

# ── Current values ──────────────────────────
current_oas = float(oas_df["value"].iloc[-1]) if not oas_df.empty else None
current_fed = float(fed_df["value"].iloc[-1]) if not fed_df.empty else None
avg_bdc_disc = compute_bdc_discount(bdc_prices, BDC_NAV)

# ── Evaluate all signals ─────────────────────
s1_icon, s1_state, s1_detail = (
    oas_status(current_oas) if current_oas else ("⚪", "NO DATA", "Enter FRED key")
)
s2_icon, s2_state, s2_detail = default_status(default_rate, default_accelerating, default_confidence)
s3_icon, s3_state, s3_detail = fed_status(current_fed or 0.0, fed_df)
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
    ("HY OAS Spread",           s1_icon, s1_state, s1_detail, "Cross and hold above 500 bps"),
    ("Default Rate Trajectory", s2_icon, s2_state, s2_detail, "Rate stabilises at 5–7% plateau"),
    ("Fed Rate Path",           s3_icon, s3_state, s3_detail, "Two meetings removing bias"),
    ("BDC Discount to NAV",     s4_icon, s4_state, s4_detail, "Sector average exceeds 15%"),
    ("Redemption Gate Events",  s5_icon, s5_state, s5_detail, "Third or fourth top-tier event"),
    ("2027 Maturity Wall",      s6_icon, s6_state, s6_detail, "Major restructurings announced"),
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
        st.caption(
            "Estimates from CCC/HY spread ratio. "
            "Validate monthly vs Moody's releases."
        )

with col_auto2:
    with st.expander("🤖 Signal 5: Gate Event Automation"):
        st.markdown(f"**Source:** {gate_source}")
        st.markdown(f"**Detected Events:** {gate_events}")
        st.markdown(f"**Confidence:** {gate_confidence:.0f}%")
        
        if gate_details:
            st.markdown("**Recent Gate Events Detected:**")
            for event in gate_details[-5:]:  # Show last 5
                st.markdown(
                    f"- **{event['manager']}** ({event['date'].strftime('%Y-%m-%d')}) "
                    f"— [SEC Filing]({event['url']})"
                )
        else:
            st.markdown("No gate events detected in SEC filings (past 365 days)")
        
        st.caption(
            "Scans SEC 8-K filings for keywords: 'redemption cap', 'gating', etc. "
            "Complements with P/NAV stress signals."
        )

st.markdown("---")

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
st.subheader("📈 Live Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if not oas_df.empty:
        fig_oas = spark_chart(
            oas_df, "HY OAS Spread (bps)", "rgb(79,195,247)",
            threshold_lines=[
                (500, "🟢 500 bps — Entry", "#4caf50"),
                (350, "🟡 350 bps — Accumulate", "#ff9800"),
            ],
        )
        st.plotly_chart(fig_oas, use_container_width=True)
    else:
        st.info("OAS chart requires FRED API key.")

with chart_col2:
    if not fed_df.empty:
        fig_fed = spark_chart(
            fed_df, "Fed Funds Rate (%)", "rgb(174,213,129)",
            threshold_lines=[
                (4.5, "4.5% — Pivot watch", "#ff9800"),
                (3.0, "3.0% — Target cut", "#4caf50"),
            ],
        )
        st.plotly_chart(fig_fed, use_container_width=True)
    else:
        st.info("Fed rate chart requires FRED API key.")

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
    "Scenario":   ["0–1 green", "2–3 green", "4 green", "5–6 green"],
    "Signals":    ["Stay out",  "Watch",     "Accumulate", "Deploy"],
    "FALN %":     ["0%",        "2–5%",      "8–12%",      "12–15%"],
    "FALN £":     ["£0",        "£2–5k",     "£8–12k",     "£12–15k"],
    "Remainder":  [
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
# DATA TABLES
# ─────────────────────────────────────────────
with st.expander("📋 Raw Data & Validation"):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["HY OAS", "CCC OAS", "Fed Funds", "Default Rate", "Gate Events", "BDC Prices"]
    )
    with tab1:
        st.dataframe(oas_df.tail(30), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(ccc_df.tail(30), use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(fed_df.tail(30), use_container_width=True, hide_index=True)
    with tab4:
        inputs_df = pd.DataFrame({
            "Parameter": ["Default Rate (%)", "Accelerating", "Confidence (%)", "Source"],
            "Value": [f"{default_rate:.2f}%", "Yes" if default_accelerating else "No", 
                     f"{default_confidence:.0f}%", default_source],
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
            disc = (nav - price) / nav * 100 if price else None
            rows.append({
                "Ticker": ticker,
                "NAV/share ($)": nav,
                "Price ($)": round(price, 2) if price else "N/A",
                "Discount (%)": f"{disc:.1f}%" if disc else "N/A",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "**Disclaimer:** For educational purposes only. Not regulated financial advice. "
    "Data: FRED, SEC EDGAR, Yahoo Finance | Built April 2026"
)
