"""
Distressed Debt Entry Signal Dashboard
======================================
Tracks 6 key signals to determine optimal entry timing
for fallen angel / distressed credit investments (e.g. FALN ETF).

Requirements:
    pip install streamlit pandas requests yfinance plotly python-dateutil

Run:
    streamlit run distressed_dashboard.py

Free data sources used:
    - FRED API (free key at fred.stlouisfed.org) → HY OAS spread, Fed Funds Rate
    - yfinance (no key needed)                  → BDC prices
    - Manual inputs                             → Default rate, gate events
"""

import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Distressed Credit Entry Dashboard",
    page_icon="📡",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS & THRESHOLDS
# ─────────────────────────────────────────────
HY_OAS_SERIES    = "BAMLH0A0HYM2"   # ICE BofA US HY OAS (bps)
FED_RATE_SERIES  = "DFF"             # Daily Fed Funds Effective Rate

# BDC tickers and their last known NAV per share (update quarterly from earnings)
BDC_NAV = {
    "ARCC": 19.24,   # Ares Capital — Q4 2025 NAV/share
    "FSK":  23.81,   # FS KKR Capital — Q4 2025 NAV/share
    "OBDC": 15.33,   # Blue Owl Capital Corp — Q4 2025 NAV/share
}

# Signal thresholds (from the dashboard logic in the analysis)
OAS_THRESHOLDS = {
    "green":  500,   # bps — deploy meaningful capital
    "amber":  350,   # bps — begin slow accumulation
}
DEFAULT_RATE_GREEN    = 5.5   # % — stabilising plateau
BDC_DISCOUNT_GREEN    = 15.0  # % average discount to NAV
GATE_EVENTS_GREEN     = 3     # number of top-tier gating events

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
        placeholder="d0d2c8e46964b4dd9fafc65fe9141aa8",
    )

    st.markdown("---")
    st.subheader("Manual Inputs (Monthly)")

    default_rate = st.number_input(
        "Trailing 12m Speculative-Grade Default Rate (%)",
        min_value=0.0, max_value=20.0, value=3.2, step=0.1,
        help="Update monthly from Moody's or S&P Global press releases (free PDF)",
    )
    default_accelerating = st.checkbox(
        "Default rate still accelerating?", value=True,
        help="Uncheck when Moody's/S&P guidance signals stabilisation",
    )

    gate_events = st.number_input(
        "Major BDC/Private Credit Gate Events (YTD)",
        min_value=0, max_value=20, value=2, step=1,
        help="Count Ares, Apollo, Blue Owl, Blackstone gating announcements",
    )

    maturity_wall_firing = st.checkbox(
        "2027 maturity wall restructurings announced?", value=False,
        help="Tick when major leveraged loan/HY bond 2027 maturities begin restructuring",
    )

    st.markdown("---")
    st.caption("Data refreshes on each page load. Last refresh:")
    st.caption(datetime.now().strftime("%d %b %Y  %H:%M UTC"))


# ─────────────────────────────────────────────
# DATA FETCHING FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)  # cache 1 hour
def fetch_fred(series_id: str, api_key: str, days: int = 365) -> pd.DataFrame:
    """Pull a FRED time series as a DataFrame."""
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
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame(columns=["date", "value"])


@st.cache_data(ttl=3600)
def fetch_bdc_prices(tickers: list) -> dict:
    """Pull latest price and 1-year history for BDC tickers via yfinance."""
    result = {}
    for t in tickers:
        try:
            tk   = yf.Ticker(t)
            hist = tk.history(period="1y")[["Close"]].reset_index()
            hist.columns = ["date", "value"]
            hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
            result[t] = {
                "history": hist,
                "latest":  float(hist["value"].iloc[-1]) if not hist.empty else None,
            }
        except Exception:
            result[t] = {"history": pd.DataFrame(), "latest": None}
    return result


def compute_bdc_discount(bdc_prices: dict, bdc_nav: dict) -> float:
    """Return average % discount of BDC prices to their stated NAV."""
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
        return "🟡", "AMBER", f"{oas:.0f} bps — Begin accumulating slowly"
    else:
        return "🔴", "RED", f"{oas:.0f} bps — Too tight, market complacent"


def default_status(rate: float, accelerating: bool):
    if rate >= DEFAULT_RATE_GREEN and not accelerating:
        return "🟢", "GREEN", f"{rate:.1f}% — Plateau reached, cycle maturing"
    elif rate >= DEFAULT_RATE_GREEN and accelerating:
        return "🟡", "AMBER", f"{rate:.1f}% — Rising but approaching trigger level"
    else:
        return "🔴", "RED", f"{rate:.1f}% — Still accelerating, cycle early"


def fed_status(fed_rate: float, fed_df: pd.DataFrame):
    """Approximate Fed pivot signal from rate trend."""
    if fed_df.empty:
        return "⚪", "UNKNOWN", "No data — check FRED API key"
    recent = fed_df.tail(60)["value"]
    is_cutting = recent.iloc[-1] < recent.iloc[0]
    rate_str = f"Fed Funds: {fed_rate:.2f}%"
    if is_cutting and fed_rate < 4.5:
        return "🟢", "GREEN", f"{rate_str} — Cutting cycle confirmed"
    elif is_cutting:
        return "🟡", "AMBER", f"{rate_str} — Pivot underway, watch meeting language"
    else:
        return "🔴", "RED", f"{rate_str} — Higher for longer still in force"


def bdc_status(avg_discount: float):
    if avg_discount >= BDC_DISCOUNT_GREEN:
        return "🟢", "GREEN", f"{avg_discount:.1f}% avg discount — Stress fully priced"
    elif avg_discount >= 8:
        return "🟡", "AMBER", f"{avg_discount:.1f}% avg discount — Widening, watch closely"
    elif avg_discount >= 0:
        return "🔴", "RED", f"{avg_discount:.1f}% avg discount — Still near par"
    else:
        return "🔴", "RED", f"Trading at premium — No distress signal"


def gate_status(events: int):
    if events >= GATE_EVENTS_GREEN:
        return "🟢", "GREEN", f"{events} events — Capitulation signal approaching"
    elif events >= 2:
        return "🟡", "AMBER", f"{events} events — Stress building, not at maximum"
    else:
        return "🔴", "RED", f"{events} events — Too few to signal cycle bottom"


def maturity_status(firing: bool):
    if firing:
        return "🟢", "GREEN", "2027 restructurings announced — Wall biting"
    else:
        return "🔴", "RED", "No major 2027 restructurings yet — Cycle early"


def overall_recommendation(green_count: int) -> tuple:
    if green_count >= 5:
        return (
            "🚀 DEPLOY CAPITAL",
            "green",
            "5–6 signals green. This is the entry window the analysis described. "
            "Allocate your planned position in full across 1–2 tranches.",
        )
    elif green_count >= 4:
        return (
            "📈 ACCUMULATE",
            "orange",
            "4 signals green. Begin meaningful accumulation — 50–60% of planned "
            "allocation now, remainder on next confirmation signal.",
        )
    elif green_count >= 2:
        return (
            "🟡 WATCH & WAIT",
            "yellow",
            "2–3 signals green. Maintain dry powder in short-duration T-bills. "
            "Deploy first small tranche (£2–4k) only if OAS crosses 500 bps.",
        )
    else:
        return (
            "🔴 STAY IN CASH",
            "red",
            "Fewer than 2 signals green. Cycle is early-stage. Park capital in "
            "money market / T-bills at 4.5–5%. Revisit in 60 days.",
        )


# ─────────────────────────────────────────────
# CHART HELPERS
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


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.title("📡 Distressed Credit Entry Signal Dashboard")
st.caption(
    "Tracks 6 key market signals to determine optimal entry timing "
    "for fallen angel / distressed credit investments (e.g. FALN ETF). "
    "Based on the investor framework developed in analysis — April 2026."
)
st.markdown("---")

# ── Fetch data ──────────────────────────────
if not fred_key:
    st.warning(
        "⚠️  Enter your free FRED API key in the sidebar to load live OAS and Fed rate data. "
        "Get one in 30 seconds at **https://fred.stlouisfed.org/docs/api/api_key.html**"
    )
    oas_df  = pd.DataFrame(columns=["date", "value"])
    fed_df  = pd.DataFrame(columns=["date", "value"])
else:
    with st.spinner("Fetching FRED data…"):
        oas_df = fetch_fred(HY_OAS_SERIES, fred_key, days=730)
        fed_df = fetch_fred(FED_RATE_SERIES, fred_key, days=730)

with st.spinner("Fetching BDC prices from Yahoo Finance…"):
    bdc_prices = fetch_bdc_prices(list(BDC_NAV.keys()))

# ── Compute current values ───────────────────
current_oas = float(oas_df["value"].iloc[-1]) if not oas_df.empty else None
current_fed = float(fed_df["value"].iloc[-1]) if not fed_df.empty else None
avg_bdc_disc = compute_bdc_discount(bdc_prices, BDC_NAV)

# ── Evaluate signals ─────────────────────────
if current_oas:
    s1_icon, s1_state, s1_detail = oas_status(current_oas)
else:
    s1_icon, s1_state, s1_detail = "⚪", "NO DATA", "Enter FRED API key"

s2_icon, s2_state, s2_detail = default_status(default_rate, default_accelerating)
s3_icon, s3_state, s3_detail = fed_status(current_fed or 0.0, fed_df)
s4_icon, s4_state, s4_detail = bdc_status(avg_bdc_disc)
s5_icon, s5_state, s5_detail = gate_status(gate_events)
s6_icon, s6_state, s6_detail = maturity_status(maturity_wall_firing)

green_count = sum(
    1 for state in [s1_state, s2_state, s3_state, s4_state, s5_state, s6_state]
    if state == "GREEN"
)

rec_label, rec_color, rec_text = overall_recommendation(green_count)

# ─────────────────────────────────────────────
# SECTION 1 — OVERALL RECOMMENDATION
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
# SECTION 2 — SIGNAL CHECKLIST
# ─────────────────────────────────────────────
st.subheader("🚦 Six-Signal Checklist")

signals = [
    ("HY OAS Spread",           s1_icon, s1_state, s1_detail,
     "Cross and hold above 500 bps"),
    ("Default Rate Trajectory", s2_icon, s2_state, s2_detail,
     "Rate stabilises at 5–7% plateau"),
    ("Fed Rate Path",           s3_icon, s3_state, s3_detail,
     "Two meetings removing tightening bias"),
    ("BDC Discount to NAV",     s4_icon, s4_state, s4_detail,
     "Sector average discount exceeds 15%"),
    ("Redemption Gate Events",  s5_icon, s5_state, s5_detail,
     "Third or fourth top-tier gating event"),
    ("2027 Maturity Wall",      s6_icon, s6_state, s6_detail,
     "Major 2027 maturity restructurings announced"),
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
    state_color = {"GREEN": "green", "AMBER": "orange",
                   "RED": "red"}.get(state, "grey")
    cols[2].markdown(
        f"<span style='color:{state_color};font-weight:bold'>{state}</span>",
        unsafe_allow_html=True,
    )
    cols[3].markdown(detail)
    cols[4].markdown(f"<span style='color:#888'>{trigger}</span>",
                     unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 3 — CHARTS
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
                (270, "Historical avg (pre-2025)", "#e53935"),
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
                (4.5, "4.5% — Pivot watch zone", "#ff9800"),
                (3.0, "3.0% — Target cut level", "#4caf50"),
            ],
        )
        st.plotly_chart(fig_fed, use_container_width=True)
    else:
        st.info("Fed rate chart requires FRED API key.")

# BDC chart full width
bdc_has_data = any(
    not v["history"].empty for v in bdc_prices.values()
)
if bdc_has_data:
    st.plotly_chart(
        bdc_chart(bdc_prices, BDC_NAV),
        use_container_width=True,
    )
else:
    st.warning("BDC price data unavailable — check internet connection.")

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 4 — POSITION SIZING GUIDE
# ─────────────────────────────────────────────
st.subheader("💼 Position Sizing Guide (based on £100k spare cash)")

sizing_data = {
    "Scenario":   ["0–1 green", "2–3 green", "4 green", "5–6 green"],
    "Signals":    ["Stay out",  "Watch",     "Accumulate", "Deploy"],
    "FALN %":     ["0%",        "2–5%",      "8–12%",      "12–15%"],
    "FALN £":     ["£0",        "£2–5k",     "£8–12k",     "£12–15k"],
    "Remainder":  [
        "100% T-bills / money market (4.5–5%)",
        "95–98% T-bills",
        "Residual in T-bills; keep £20–30k dry powder",
        "Residual in T-bills; review at 6 months",
    ],
}
df_sizing = pd.DataFrame(sizing_data)
# Highlight current row
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
st.caption(
    f"▶ Currently highlighted row reflects {green_count}/6 signals green. "
    "Your situation: spare cash, meaningful existing credit exposure, "
    "sub-2-year horizon — apply a further 30–40% haircut to FALN allocation."
)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 5 — DATA TABLE & FOOTER
# ─────────────────────────────────────────────
with st.expander("📋 Raw Data Tables"):
    tab1, tab2, tab3 = st.tabs(["HY OAS", "Fed Funds Rate", "BDC Prices"])
    with tab1:
        st.dataframe(oas_df.tail(30).sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(fed_df.tail(30).sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)
    with tab3:
        rows = []
        for ticker, nav in BDC_NAV.items():
            price  = bdc_prices.get(ticker, {}).get("latest")
            disc   = (nav - price) / nav * 100 if price else None
            rows.append({
                "Ticker":        ticker,
                "NAV/share ($)": nav,
                "Price ($)":     round(price, 2) if price else "N/A",
                "Discount (%)":  f"{disc:.1f}%" if disc is not None else "N/A",
                "NAV Source":    "Q4 2025 earnings — update quarterly",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "⚠ NAV values are manually set from latest quarterly earnings. "
            "Update BDC_NAV dict in source code each quarter."
        )

st.markdown("---")
st.caption(
    "**Disclaimer:** This dashboard is for informational and educational purposes only. "
    "It does not constitute regulated financial advice. All investment decisions carry risk "
    "including possible loss of principal. Consult a qualified financial adviser before investing. "
    "| Data: FRED (Federal Reserve Bank of St. Louis), Yahoo Finance | "
    f"Built April 2026"
)
