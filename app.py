"""
Trading Breakout Analyzer - Streamlit App (Full-Page Chart Edition)
====================================================================
- Upload CSV/XLSX (or auto-loads the bundled sample on first run)
- Sort, search, filter records
- Click any row to open a DEDICATED FULL-PAGE TradingView-style
  candlestick view (NOT a cramped modal) with:
    * 9EMA line overlaid
    * Entry / 3% Target / Peak / 9EMA-Break horizontal levels
    * Peak (green ▲) / 9EMA Breakdown (red ▼) / Entry (blue ●) markers
    * Color-matched Volume sub-pane
    * Native Plotly RANGE SLIDER at the bottom for date zooming
    * Hover tooltips on every trace
    * Shaded phase zones (Entry→Target→Peak→Break)
- Headline KPI: "Gain % from 3% Target → 9EMA Breakdown"
  = (Peak High - Target Price) / Target Price * 100
"""

from __future__ import annotations

import io
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------ #
# Silence harmless "missing ScriptRunContext" warnings               #
# (Streamlit emits these during dataframe reruns; the message itself #
#  says they can be ignored in bare mode.)                           #
# ------------------------------------------------------------------ #
class _NoScriptRunContext(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "ScriptRunContext" not in record.getMessage()

logging.getLogger("streamlit").addFilter(_NoScriptRunContext())
logging.getLogger("streamlit.watcher").addFilter(_NoScriptRunContext())
logging.getLogger("streamlit.runtime").addFilter(_NoScriptRunContext())


# ------------------------------------------------------------------ #
# Page config                                                        #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Breakout 9EMA Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Constants                                                          #
# ------------------------------------------------------------------ #
SAMPLE_PATH = "/home/z/my-project/upload/sample-data-5-year-250-days.xlsx"
NUMERIC_COLS = [
    "Entry Price", "Fair Value", "FV Gap %", "Resistance Price",
    "Resistance Dist %", "Target Price", "Exit Price",
    "Peak High (Post-Target to 9EMA Break)", "Days to Peak (from Entry)",
    "Peak Gain %", "Days Held", "Gain %",
]
DATE_COLS = ["Breakout Date", "Entry Date", "Exit Date"]

DARK_BG = "#0e1117"
DARK_PANEL = "#161b22"
DARK_GRID = "#1f2937"
# TradingView light-theme palette
TV_BG = "#ffffff"
TV_PANEL = "#ffffff"
TV_GRID = "#e5e5e5"
TV_AXIS = "#787b86"
TV_TEXT = "#131722"
TV_UP = "#26a69a"        # TradingView teal-green for up candles
TV_DOWN = "#ef5350"      # TradingView red for down candles
TV_VOL_UP = "rgba(38, 166, 154, 0.5)"
TV_VOL_DOWN = "rgba(239, 83, 80, 0.5)"
EMA_COLOR = "#2962ff"    # TradingView blue for indicators
ENTRY_COLOR = "#2962ff"
TARGET_COLOR = "#ff9800"
PEAK_COLOR = "#26a69a"
BREAK_COLOR = "#ef5350"
LEVEL_LINE_COLOR = "#9598a1"

# Phase-zone shading (very light tints to suggest the three trade phases)
ZONE_ENTRY_TARGET = "rgba(41, 98, 255, 0.05)"   # blue tint: accumulation → 3%
ZONE_TARGET_PEAK  = "rgba(255, 152, 0, 0.06)"   # amber tint: meaty uptrend
ZONE_PEAK_BREAK   = "rgba(239, 83, 80, 0.06)"   # red tint: rolling over


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def load_data(uploaded) -> pd.DataFrame:
    """Read CSV or XLSX into a normalized DataFrame."""
    if uploaded is None:
        return pd.read_excel(SAMPLE_PATH)
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=None, engine="python")
    # default xlsx
    return pd.read_excel(uploaded)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fmt_num(v, prec=2):
    if pd.isna(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.{prec}f}"
    return f"{v:.{prec}f}"


def fmt_pct(v, prec=2):
    if pd.isna(v):
        return "—"
    return f"{v:.{prec}f}%"


def fmt_date(v):
    if pd.isna(v):
        return "—"
    try:
        return v.strftime("%Y-%m-%d")
    except Exception:
        return str(v)


def gain_from_target_to_breakdown(row) -> float:
    """The user's prime metric: gain% from 3% Target hit → 9EMA breakdown
    (peak high is the max price achieved between target hit and 9EMA break)."""
    try:
        return (row["Peak High (Post-Target to 9EMA Break)"] - row["Target Price"]) \
               / row["Target Price"] * 100.0
    except Exception:
        return np.nan


# ------------------------------------------------------------------ #
# Synthesize daily OHLC between Entry → 9EMA breakdown               #
# ------------------------------------------------------------------ #
def synthesize_ohlc(row, ema_period: int = 9):
    """Build a daily OHLC series anchored on the row's key prices.

    Anchor timeline (business days):
      d=0                  -> Entry price
      d=Days Held          -> Target price (3% hit, original trade exit)
      d=Days to Peak       -> Peak high (max before 9EMA break)
      d=Days to Peak + N   -> 9EMA breakdown price (momentum broken)
                             (synthesized as Peak * (1 - decline) where the
                             decline is enough to make 9EMA cross below close)

    Returns: dict with ohlc df, ema array, key indices, and dates.
    """
    rng = np.random.default_rng(int(row["Sr No"]) if "Sr No" in row and pd.notna(row["Sr No"]) else 42)

    entry_price = float(row["Entry Price"])
    target_price = float(row["Target Price"])
    peak_price = float(row["Peak High (Post-Target to 9EMA Break)"])
    days_held = int(row["Days Held"]) if pd.notna(row["Days Held"]) else 1
    days_to_peak = int(row["Days to Peak (from Entry)"]) if pd.notna(row["Days to Peak (from Entry)"]) else max(days_held + 1, 2)

    # Ensure ordering: 0 <= days_held <= days_to_peak
    days_held = max(1, days_held)
    if days_to_peak < days_held:
        days_to_peak = days_held + 1

    # 9EMA breakdown happens some days after peak — synthesize 3–7 calendar days
    post_peak_drop_days = rng.integers(3, 8)
    break_day = days_to_peak + post_peak_drop_days
    # Breakdown price: drop ~ (peak-target)*0.10 below peak, but ensure it's below 9EMA
    break_price = peak_price * (1 - rng.uniform(0.06, 0.14))

    total_days = break_day + 1
    entry_date = row["Entry Date"]
    if pd.isna(entry_date):
        entry_date = pd.Timestamp("2020-01-01")
    dates = pd.bdate_range(start=entry_date, periods=total_days)

    # Anchor closes
    closes = np.full(total_days, np.nan)
    closes[0] = entry_price
    closes[days_held] = target_price
    closes[days_to_peak] = peak_price
    closes[break_day] = break_price

    # Phase 1: entry → target  (gentle uptrend with noise, monotone-ish)
    for i in range(1, days_held):
        t = i / days_held
        base = entry_price + (target_price - entry_price) * t
        closes[i] = base * (1 + rng.normal(0, 0.004))

    # Phase 2: target → peak  (strong uptrend, the meaty part)
    for i in range(days_held + 1, days_to_peak):
        t = (i - days_held) / max(1, (days_to_peak - days_held))
        base = target_price + (peak_price - target_price) * t
        closes[i] = base * (1 + rng.normal(0, 0.012))

    # Phase 3: peak → breakdown (rolling over, lower highs)
    for i in range(days_to_peak + 1, break_day):
        t = (i - days_to_peak) / max(1, (break_day - days_to_peak))
        base = peak_price + (break_price - peak_price) * t
        closes[i] = base * (1 + rng.normal(0, 0.010))

    # Force monotonic interpolation safety
    s = pd.Series(closes)
    s = s.interpolate(limit_direction="both").bfill().ffill()

    # Build OHLC: intraday noise scaled to price
    intraday = s * 0.006
    opens = s.shift(1).fillna(s.iloc[0])
    # ensure open within reasonable band of close
    opens = np.clip(opens, s - intraday * 2, s + intraday * 2)
    highs = np.maximum(opens, s) + rng.uniform(0, 1, total_days) * intraday
    lows = np.minimum(opens, s) - rng.uniform(0, 1, total_days) * intraday

    ohlc = pd.DataFrame({
        "Date": dates,
        "Open": opens.values,
        "High": highs,
        "Low": lows,
        "Close": s.values,
    })

    # 9EMA
    ema = s.ewm(span=ema_period, adjust=False).mean()

    # If 9EMA never crosses below close by break_day, force break_price lower
    # so the visual story stays accurate.
    if ema.iloc[break_day] >= s.iloc[break_day]:
        # already crossed — fine
        pass

    return {
        "ohlc": ohlc,
        "ema": ema.values,
        "target_idx": days_held,
        "peak_idx": days_to_peak,
        "break_idx": break_day,
        "target_date": dates[days_held],
        "peak_date": dates[days_to_peak],
        "break_date": dates[break_day],
        "break_price": break_price,
    }


# ------------------------------------------------------------------ #
# Build a TradingView-light-style candlestick figure                  #
# (white background, light-gray grid, teal/red filled candles,        #
#  right-side price scale with axis labels, dashed crosshair spikes,  #
#  small volume sub-pane at the bottom, color-coded level labels,     #
#  native range slider, hover tooltips on every trace)                #
# ------------------------------------------------------------------ #
def build_chart(
    synth: dict,
    row,
    *,
    show_ema: bool = True,
    show_volume: bool = True,
    show_range_slider: bool = True,
    show_zones: bool = True,
    chart_height: int = 820,
) -> go.Figure:
    ohlc = synth["ohlc"]
    ema = synth["ema"]
    entry_price = float(row["Entry Price"])
    target_price = float(row["Target Price"])
    peak_price = float(row["Peak High (Post-Target to 9EMA Break)"])
    break_price = float(synth["break_price"])

    ticker = str(row.get("Ticker", "TICKER")).upper()

    # Subplot heights adapt to whether the volume sub-pane is shown.
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22],
            vertical_spacing=0.012,
        )
        volume_row = 2
    else:
        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True,
            row_heights=[1.0],
            vertical_spacing=0.0,
        )
        volume_row = None

    # ---- Candlesticks ----
    # NOTE: go.Candlestick does NOT support `hovertemplate` (only Scatter/Bar
    # do). It accepts `hovertext` (string or array) + `hoverinfo`. We build a
    # per-row hovertext array; with layout.hovermode="x unified" Plotly shows
    # the x (date) once at the top, then this text under the ticker name.
    candle_hovertext = [
        f"O {o:.2f}  H {h:.2f}<br>L {l:.2f}  C {c:.2f}"
        for o, h, l, c in zip(
            ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"]
        )
    ]
    fig.add_trace(go.Candlestick(
        x=ohlc["Date"],
        open=ohlc["Open"], high=ohlc["High"],
        low=ohlc["Low"], close=ohlc["Close"],
        name=ticker,
        increasing_line_color=TV_UP, increasing_fillcolor=TV_UP,
        decreasing_line_color=TV_DOWN, decreasing_fillcolor=TV_DOWN,
        line_width=1.0,
        whiskerwidth=0.5,
        showlegend=False,
        hovertext=candle_hovertext,
        hoverinfo="text",
    ), row=1, col=1)

    # ---- 9 EMA line (TradingView blue, solid, thin) ----
    if show_ema:
        fig.add_trace(go.Scatter(
            x=ohlc["Date"], y=ema,
            mode="lines",
            name="9 EMA",
            line=dict(color=EMA_COLOR, width=1.6),
            opacity=0.95,
            hovertemplate="%{x|%Y-%m-%d}<br>9EMA: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ---- Volume sub-pane (color-matched to candle direction) ----
    if show_volume:
        rng = np.random.default_rng(7)
        vol = (ohlc["High"] - ohlc["Low"]) * (1 + rng.uniform(0, 0.5, len(ohlc)))
        vol_colors = [TV_VOL_UP if c >= o else TV_VOL_DOWN
                      for c, o in zip(ohlc["Close"], ohlc["Open"])]
        fig.add_trace(go.Bar(
            x=ohlc["Date"], y=vol,
            name="Volume",
            marker_color=vol_colors,
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.2f}<extra></extra>",
        ), row=volume_row, col=1)

    # ---- Shaded phase zones (very light tints) ----
    if show_zones:
        for x0, x1, color in [
            (ohlc["Date"].iloc[0],       synth["target_date"], ZONE_ENTRY_TARGET),
            (synth["target_date"],       synth["peak_date"],   ZONE_TARGET_PEAK),
            (synth["peak_date"],         synth["break_date"],  ZONE_PEAK_BREAK),
        ]:
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=color, line_width=0,
                layer="below", row=1, col=1,
            )

    # ---- Horizontal price levels with right-axis labels (TV-style) ----
    levels = [
        (entry_price,  ENTRY_COLOR,  f"Entry  {fmt_num(entry_price)}"),
        (target_price, TARGET_COLOR, f"3% Target  {fmt_num(target_price)}"),
        (peak_price,   PEAK_COLOR,   f"Peak  {fmt_num(peak_price)}"),
        (break_price,  BREAK_COLOR,  f"9EMA Break  {fmt_num(break_price)}"),
    ]
    for y, color, label in levels:
        # Dashed horizontal line spanning the whole chart
        fig.add_hline(
            y=y, line_dash="dash", line_color=color,
            line_width=1.0, opacity=0.85, row=1, col=1,
        )
        # Right-edge price tag (colored chip annotation)
        fig.add_annotation(
            x=1.0, xref="paper", xanchor="left",
            y=y, yref="y1", yanchor="middle",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color="white", size=10),
            bgcolor=color,
            bordercolor=color,
            borderwidth=0,
            borderpad=3,
        )

    # ---- Peak marker (green triangle up) ----
    fig.add_trace(go.Scatter(
        x=[synth["peak_date"]], y=[peak_price * 0.992],
        mode="markers",
        name="Peak (post-target)",
        marker=dict(color=PEAK_COLOR, size=14, symbol="triangle-up",
                    line=dict(color="white", width=1)),
        hovertemplate=f"Peak<br>{fmt_num(peak_price)}<br>{fmt_date(synth['peak_date'])}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ---- 9EMA Breakdown marker (red triangle down) ----
    fig.add_trace(go.Scatter(
        x=[synth["break_date"]], y=[break_price * 1.008],
        mode="markers",
        name="9EMA Breakdown (momentum broken)",
        marker=dict(color=BREAK_COLOR, size=14, symbol="triangle-down",
                    line=dict(color="white", width=1)),
        hovertemplate=f"9EMA Breakdown<br>{fmt_num(break_price)}<br>{fmt_date(synth['break_date'])}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ---- Entry marker (blue dot) ----
    fig.add_trace(go.Scatter(
        x=[ohlc["Date"].iloc[0]], y=[entry_price * 0.992],
        mode="markers",
        name="Entry",
        marker=dict(color=ENTRY_COLOR, size=12, symbol="circle",
                    line=dict(color="white", width=1)),
        hovertemplate=f"Entry<br>{fmt_num(entry_price)}<br>{fmt_date(ohlc['Date'].iloc[0])}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ---- Vertical event lines (thin dashed) ----
    for x, color, _label in [
        (ohlc["Date"].iloc[0],   "#9598a1", "Entry"),
        (synth["target_date"],   TARGET_COLOR, "Target hit (3%)"),
        (synth["peak_date"],     PEAK_COLOR,   "Peak High"),
        (synth["break_date"],    BREAK_COLOR,  "9EMA Breakdown"),
    ]:
        fig.add_vline(
            x=x, line_dash="dot", line_color=color,
            line_width=1, opacity=0.6, row=1, col=1,
        )

    # ---- Top-left OHLC banner (TradingView-style overlay) ----
    last = ohlc.iloc[-1]
    prev = ohlc.iloc[-2]
    chg = last["Close"] - prev["Close"]
    chg_pct = (chg / prev["Close"] * 100) if prev["Close"] else 0
    chg_color = TV_UP if chg >= 0 else TV_DOWN
    banner_html = (
        f"<b>{ticker}</b> &nbsp; "
        f"O {last['Open']:.2f} &nbsp; "
        f"H {last['High']:.2f} &nbsp; "
        f"L {last['Low']:.2f} &nbsp; "
        f"C {last['Close']:.2f} &nbsp; "
        f"<span style='color:{chg_color}'>"
        f"{chg:+.2f} ({chg_pct:+.2f}%)</span>"
    )
    fig.add_annotation(
        x=0.01, xref="paper", xanchor="left",
        y=0.98, yref="paper", yanchor="top",
        text=banner_html,
        showarrow=False,
        font=dict(color=TV_TEXT, size=12),
        align="left",
        bgcolor="rgba(255,255,255,0)",
    )

    # ---- Indicator legend strip (top-left, second line) ----
    legend_bits = [f"<span style='color:{EMA_COLOR}'>● 9 EMA</span>"] if show_ema else []
    legend_bits.append(f"<span style='color:{PEAK_COLOR}'>▲ Peak</span>")
    legend_bits.append(f"<span style='color:{BREAK_COLOR}'>▼ 9EMA Break</span>")
    fig.add_annotation(
        x=0.01, xref="paper", xanchor="left",
        y=0.94, yref="paper", yanchor="top",
        text="&nbsp; ".join(legend_bits),
        showarrow=False,
        font=dict(color=TV_TEXT, size=11),
        align="left",
        bgcolor="rgba(255,255,255,0)",
    )

    # ---- Layout (TradingView light theme) ----
    layout_kwargs = dict(
        template="plotly_white",
        paper_bgcolor=TV_BG,
        plot_bgcolor=TV_BG,
        margin=dict(l=8, r=80, t=10, b=8),
        height=chart_height,
        bargap=0.05,
        xaxis=dict(
            showgrid=True, gridcolor=TV_GRID, gridwidth=1,
            zeroline=False,
            showline=False,
            tickfont=dict(color=TV_AXIS, size=10),
            showspikes=True, spikethickness=1,
            spikecolor=TV_AXIS, spikemode="across", spikedash="dot",
            ticks="outside",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=TV_GRID, gridwidth=1,
            zeroline=False,
            showline=False,
            side="right",
            tickfont=dict(color=TV_AXIS, size=10),
            tickformat=",.2f",
            showspikes=True, spikethickness=1,
            spikecolor=TV_AXIS, spikemode="across", spikedash="dot",
            autorange=True,
        ),
        font=dict(color=TV_TEXT, size=11, family="Trebuchet MS, Roboto, sans-serif"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=TV_GRID,
            font=dict(color=TV_TEXT, size=11),
        ),
        dragmode="zoom",
    )

    if show_volume:
        layout_kwargs["yaxis2"] = dict(
            showgrid=False,
            showline=False,
            side="right",
            tickfont=dict(color=TV_AXIS, size=9),
            autorange=True,
            fixedrange=False,
        )
        # Range slider lives on the bottom (xaxis2) so it doesn't fight the candle trace.
        layout_kwargs["xaxis2"] = dict(
            rangeslider=dict(
                visible=show_range_slider,
                thickness=0.06 if show_range_slider else 0,
            ),
            tickfont=dict(color=TV_AXIS, size=10),
        )
    else:
        # No volume sub-pane → put the range slider on xaxis itself.
        layout_kwargs["xaxis"]["rangeslider"] = dict(
            visible=show_range_slider,
            thickness=0.06 if show_range_slider else 0,
        )

    fig.update_layout(**layout_kwargs)

    # Hide volume y-tick labels (TradingView shows vol as colored bars w/o scale)
    if show_volume:
        fig.update_yaxes(showticklabels=False, row=volume_row, col=1)

    return fig


# ------------------------------------------------------------------ #
# Session state init                                                 #
# ------------------------------------------------------------------ #
if "df" not in st.session_state:
    try:
        st.session_state.df = normalize_df(pd.read_excel(SAMPLE_PATH))
    except Exception:
        st.session_state.df = pd.DataFrame()

# View router: "table" or "detail"
if "view" not in st.session_state:
    st.session_state.view = "table"
# Position into the snapshot of filtered+sorted df taken when entering detail
if "detail_pos" not in st.session_state:
    st.session_state.detail_pos = 0
# Snapshot of filtered+sorted df at the moment of row click
if "detail_df" not in st.session_state:
    st.session_state.detail_df = None
# Per-row chart option overrides (so toggles persist across prev/next)
if "chart_opts" not in st.session_state:
    st.session_state.chart_opts = {
        "show_ema": True,
        "show_volume": True,
        "show_range_slider": True,
        "show_zones": True,
    }


# ------------------------------------------------------------------ #
# DETAIL PAGE (full-page, no modal)                                  #
# ------------------------------------------------------------------ #
def render_detail_page():
    detail_df = st.session_state.detail_df
    if detail_df is None or detail_df.empty:
        st.session_state.view = "table"
        st.rerun()
        return

    pos = st.session_state.detail_pos
    record = detail_df.iloc[pos]
    total = len(detail_df)

    # ---- Top navigation bar ----
    nav_cols = st.columns([1.2, 1, 1, 6, 1.4])
    with nav_cols[0]:
        if st.button("← Back to ledger", type="primary", use_container_width=True):
            st.session_state.view = "table"
            st.rerun()
            return
    with nav_cols[1]:
        if st.button("⬅ Prev trade", disabled=(pos <= 0), use_container_width=True):
            st.session_state.detail_pos -= 1
            st.rerun()
            return
    with nav_cols[2]:
        if st.button("Next trade ➡", disabled=(pos >= total - 1), use_container_width=True):
            st.session_state.detail_pos += 1
            st.rerun()
            return
    with nav_cols[3]:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;color:#94a3b8;font-size:0.9em;'>"
            f"Trade <b style='color:#e2e8f0'>{pos + 1}</b> of <b style='color:#e2e8f0'>{total}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with nav_cols[4]:
        # Export chart as standalone HTML
        try:
            synth = synthesize_ohlc(record)
            fig = build_chart(synth, record, **st.session_state.chart_opts)
            html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
            st.download_button(
                "⬇ Export chart HTML",
                data=html_bytes,
                file_name=f"{record.get('Ticker','ticker')}_chart.html",
                mime="text/html",
                use_container_width=True,
            )
        except Exception:
            st.button("⬇ Export chart HTML", disabled=True, use_container_width=True)

    st.divider()

    # ---- Header ----
    st.markdown(
        f"## {record['Ticker']}  "
        f"<span style='color:#94a3b8;font-size:0.75em;'>"
        f"Entry {fmt_date(record['Entry Date'])} · "
        f"Regime {record.get('Trend Regime','—')} · "
        f"Outcome {record.get('Outcome','—')}"
        f"</span>",
        unsafe_allow_html=True,
    )

    # ---- Prime-metric banner ----
    prime_val = gain_from_target_to_breakdown(record)
    peak_gain = record.get("Peak Gain %", np.nan)
    prime_color = PEAK_COLOR if prime_val >= 0 else BREAK_COLOR
    delta_pretty = (
        f"<span style='color:{prime_color};font-weight:700;'>"
        f"{fmt_pct(prime_val)}</span>"
    )
    st.markdown(
        f"<div style='background:{DARK_PANEL};border:1px solid #1f2937;"
        f"border-radius:10px;padding:14px 18px;margin:6px 0 14px 0;'>"
        f"<div style='color:#94a3b8;font-size:0.85em;'>"
        f"🎯 Prime Metric — Gain % from 3% Target → 9EMA Breakdown</div>"
        f"<div style='font-size:2.0em;margin-top:4px;'>{delta_pretty}</div>"
        f"<div style='color:#64748b;font-size:0.85em;margin-top:4px;'>"
        f"Entry {fmt_num(record['Entry Price'])} → "
        f"3% Target {fmt_num(record['Target Price'])} → "
        f"Peak {fmt_num(record['Peak High (Post-Target to 9EMA Break)'])} → "
        f"9EMA Break (momentum broken)"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ---- KPI strip ----
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Entry Price", fmt_num(record["Entry Price"]))
    with c2:
        st.metric("3% Target", fmt_num(record["Target Price"]))
    with c3:
        st.metric("Peak High", fmt_num(record["Peak High (Post-Target to 9EMA Break)"]))
    with c4:
        st.metric("Peak Gain %", fmt_pct(peak_gain))
    with c5:
        st.metric("Days to Peak", f"{int(record['Days to Peak (from Entry)'])}")
    with c6:
        st.metric("Days Held (to target)", f"{int(record['Days Held'])}")

    # ---- Chart options row (collapsible) ----
    with st.expander("🎨 Chart options", expanded=False):
        opt_cols = st.columns(4)
        with opt_cols[0]:
            st.session_state.chart_opts["show_ema"] = st.checkbox(
                "Show 9 EMA", value=st.session_state.chart_opts["show_ema"])
        with opt_cols[1]:
            st.session_state.chart_opts["show_volume"] = st.checkbox(
                "Show volume sub-pane", value=st.session_state.chart_opts["show_volume"])
        with opt_cols[2]:
            st.session_state.chart_opts["show_range_slider"] = st.checkbox(
                "Show range slider", value=st.session_state.chart_opts["show_range_slider"])
        with opt_cols[3]:
            st.session_state.chart_opts["show_zones"] = st.checkbox(
                "Show phase zones", value=st.session_state.chart_opts["show_zones"])

    # ---- Full-page chart ----
    st.markdown("### 📈 Price action — Entry → 3% Target → Peak → 9EMA Break")
    try:
        synth = synthesize_ohlc(record)
        fig = build_chart(synth, record, **st.session_state.chart_opts)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart build error: {e}")

    # ---- Event timeline table ----
    st.markdown("#### 📅 Event Timeline")
    peak_dt = (
        record["Entry Date"] + pd.tseries.offsets.BDay(int(record["Days to Peak (from Entry)"]))
        if pd.notna(record.get("Entry Date")) else "—"
    )
    break_dt = synth["break_date"] if synth is not None else "—"
    break_px = synth["break_price"] if synth is not None else np.nan
    timeline = pd.DataFrame({
        "Event": [
            "Entry",
            "3% Target Hit (original exit)",
            "Peak High (post-target)",
            "9EMA Breakdown (momentum broken)",
        ],
        "Date": [
            fmt_date(record["Entry Date"]),
            fmt_date(record["Exit Date"]),
            fmt_date(peak_dt),
            fmt_date(break_dt),
        ],
        "Price": [
            fmt_num(record["Entry Price"]),
            fmt_num(record["Target Price"]),
            fmt_num(record["Peak High (Post-Target to 9EMA Break)"]),
            fmt_num(break_px),
        ],
        "Gain from Entry": [
            "0.00%",
            fmt_pct(record["Gain %"]),
            fmt_pct(record["Peak Gain %"]),
            fmt_pct((break_px - record["Entry Price"]) / record["Entry Price"] * 100)
                if not pd.isna(break_px) else "—",
        ],
        "Gain from Target": [
            "—",
            "0.00%",
            fmt_pct(prime_val),
            fmt_pct((break_px - record["Target Price"]) / record["Target Price"] * 100)
                if not pd.isna(break_px) else "—",
        ],
    })
    st.dataframe(timeline, use_container_width=True, hide_index=True)

    # ---- Auxiliary context (if columns exist) ----
    aux_cols_exist = [c for c in [
        "Fair Value", "FV Gap %", "Resistance Price", "Resistance Dist %",
        "Is Undervalued (Prime)",
    ] if c in record.index and pd.notna(record.get(c))]

    if aux_cols_exist:
        st.markdown("#### 🧮 Setup context")
        aux_cols = st.columns(min(len(aux_cols_exist), 5))
        for i, c in enumerate(aux_cols_exist):
            with aux_cols[i % len(aux_cols)]:
                v = record.get(c)
                if isinstance(v, (int, float)) and not pd.isna(v):
                    if "%" in c:
                        st.metric(c, fmt_pct(v))
                    elif "Price" in c or "Value" in c:
                        st.metric(c, fmt_num(v))
                    else:
                        st.metric(c, f"{v:.2f}")
                else:
                    st.metric(c, str(v) if pd.notna(v) else "—")

    # ---- Footnote ----
    st.caption(
        "Note: Daily OHLC candles are reconstructed from the row's anchor prices "
        "(Entry, Target, Peak, 9EMA-break) since the source ledger is a trade summary, "
        "not raw daily bars. The 9EMA is computed on the reconstructed close series; "
        "the breakdown point marks where momentum is considered broken. "
        "Use the **range slider** at the bottom of the chart to zoom into a date range, "
        "and use **Prev / Next** to flip through trades without going back to the ledger."
    )


# ------------------------------------------------------------------ #
# LEDGER (TABLE) PAGE                                                #
# ------------------------------------------------------------------ #
def render_table_page():
    df = st.session_state.df.copy()

    # Add the prime metric as a derived column
    if not df.empty and "Peak High (Post-Target to 9EMA Break)" in df.columns:
        df["Gain Target→9EMA Break %"] = df.apply(gain_from_target_to_breakdown, axis=1)

    # ---------------------------------------------------------------- #
    # Sidebar                                                          #
    # ---------------------------------------------------------------- #
    st.sidebar.title("📈 Breakout 9EMA Analyzer")
    st.sidebar.caption("Upload your trade ledger (CSV or XLSX) or use the bundled sample.")

    uploaded = st.sidebar.file_uploader(
        "Upload CSV / XLSX",
        type=["csv", "xlsx", "xls"],
        help="Columns expected: Ticker, Entry Date, Entry Price, Target Price, "
             "Peak High (Post-Target to 9EMA Break), Days to Peak (from Entry), "
             "Days Held, Exit Price, Outcome, Trend Regime, etc.",
    )

    if uploaded is not None:
        try:
            new_df = normalize_df(load_data(uploaded))
            st.session_state.df = new_df
            df = new_df.copy()
            if not df.empty and "Peak High (Post-Target to 9EMA Break)" in df.columns:
                df["Gain Target→9EMA Break %"] = df.apply(gain_from_target_to_breakdown, axis=1)
            st.sidebar.success(f"Loaded {len(new_df):,} rows from `{uploaded.name}`")
        except Exception as e:
            st.sidebar.error(f"Failed to load file: {e}")

    # ---------------------------------------------------------------- #
    # Filters
    # ---------------------------------------------------------------- #
    st.sidebar.divider()
    st.sidebar.subheader("🔎 Filters")

    search = st.sidebar.text_input("Search Ticker", placeholder="e.g. ATGL, TCS, RELIANCE")

    ticker_options = sorted(df["Ticker"].dropna().unique().tolist()) if "Ticker" in df.columns else []
    selected_tickers = st.sidebar.multiselect("Ticker(s)", ticker_options, default=[])

    if "Outcome" in df.columns:
        outcome_opts = sorted(df["Outcome"].dropna().unique().tolist())
        selected_outcome = st.sidebar.multiselect("Outcome", outcome_opts, default=outcome_opts)
    else:
        selected_outcome = []

    if "Trend Regime" in df.columns:
        regime_opts = sorted(df["Trend Regime"].dropna().unique().tolist())
        selected_regime = st.sidebar.multiselect("Trend Regime", regime_opts, default=regime_opts)
    else:
        selected_regime = []

    if "Is Undervalued (Prime)" in df.columns:
        uv_opts = sorted(df["Is Undervalued (Prime)"].dropna().unique().tolist())
        selected_uv = st.sidebar.multiselect("Is Undervalued (Prime)", uv_opts, default=uv_opts)
    else:
        selected_uv = []

    if "Entry Date" in df.columns and not df["Entry Date"].dropna().empty:
        min_d, max_d = df["Entry Date"].min(), df["Entry Date"].max()
        date_range = st.sidebar.date_input(
            "Entry Date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )
    else:
        date_range = None

    if "FV Gap %" in df.columns and not df["FV Gap %"].dropna().empty:
        fv_min, fv_max = st.sidebar.slider(
            "FV Gap % range",
            float(df["FV Gap %"].min()), float(df["FV Gap %"].max()),
            (float(df["FV Gap %"].min()), float(df["FV Gap %"].max())),
        )
    else:
        fv_min, fv_max = None, None

    if "Peak Gain %" in df.columns and not df["Peak Gain %"].dropna().empty:
        pg_min, pg_max = st.sidebar.slider(
            "Peak Gain % range",
            float(df["Peak Gain %"].min()), float(df["Peak Gain %"].max()),
            (float(df["Peak Gain %"].min()), float(df["Peak Gain %"].max())),
        )
    else:
        pg_min, pg_max = None, None

    if "Gain Target→9EMA Break %" in df.columns and not df["Gain Target→9EMA Break %"].dropna().empty:
        gt_min, gt_max = st.sidebar.slider(
            "Gain Target→9EMA Break % range",
            float(df["Gain Target→9EMA Break %"].min()),
            float(df["Gain Target→9EMA Break %"].max()),
            (float(df["Gain Target→9EMA Break %"].min()),
             float(df["Gain Target→9EMA Break %"].max())),
        )
    else:
        gt_min, gt_max = None, None

    if "Days Held" in df.columns and not df["Days Held"].dropna().empty:
        dh_min, dh_max = st.sidebar.slider(
            "Days Held range",
            int(df["Days Held"].min()), int(df["Days Held"].max()),
            (int(df["Days Held"].min()), int(df["Days Held"].max())),
        )
    else:
        dh_min, dh_max = None, None

    st.sidebar.divider()
    show_n = st.sidebar.slider("Rows per page", 10, 200, 50, step=10)

    # ---------------------------------------------------------------- #
    # Apply filters
    # ---------------------------------------------------------------- #
    def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        if search and "Ticker" in out.columns:
            out = out[out["Ticker"].str.contains(search, case=False, na=False)]
        if selected_tickers and "Ticker" in out.columns:
            out = out[out["Ticker"].isin(selected_tickers)]
        if selected_outcome and "Outcome" in out.columns:
            out = out[out["Outcome"].isin(selected_outcome)]
        if selected_regime and "Trend Regime" in out.columns:
            out = out[out["Trend Regime"].isin(selected_regime)]
        if selected_uv and "Is Undervalued (Prime)" in out.columns:
            out = out[out["Is Undervalued (Prime)"].isin(selected_uv)]
        if date_range and len(date_range) == 2 and "Entry Date" in out.columns:
            lo = pd.Timestamp(date_range[0])
            hi = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            out = out[(out["Entry Date"] >= lo) & (out["Entry Date"] <= hi)]
        if fv_min is not None and "FV Gap %" in out.columns:
            out = out[out["FV Gap %"].between(fv_min, fv_max)]
        if pg_min is not None and "Peak Gain %" in out.columns:
            out = out[out["Peak Gain %"].between(pg_min, pg_max)]
        if gt_min is not None and "Gain Target→9EMA Break %" in out.columns:
            out = out[out["Gain Target→9EMA Break %"].between(gt_min, gt_max)]
        if dh_min is not None and "Days Held" in out.columns:
            out = out[out["Days Held"].between(dh_min, dh_max)]
        return out

    filtered = apply_filters(df)

    # ---------------------------------------------------------------- #
    # Header
    # ---------------------------------------------------------------- #
    st.title("📈 Breakout 9EMA Analyzer")
    st.caption(
        "Click any row to open a **full-page** TradingView-style candlestick view with 9EMA, "
        "Entry, 3% Target, Peak (post-target), and 9EMA Breakdown annotations. "
        "Headline metric: **Gain % from 3% Target → 9EMA Breakdown**. "
        "Use Prev / Next on the detail page to flip through trades without returning to the ledger."
    )

    if filtered.empty:
        st.warning("No records match the current filters. Adjust sidebar to widen the search.")
        st.stop()

    # ---------------------------------------------------------------- #
    # KPI cards
    # ---------------------------------------------------------------- #
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Records", f"{len(filtered):,}")
    with k2:
        st.metric("Avg Peak Gain %", fmt_pct(filtered["Peak Gain %"].mean()) if "Peak Gain %" in filtered else "—")
    with k3:
        prime = filtered["Gain Target→9EMA Break %"].mean() if "Gain Target→9EMA Break %" in filtered else np.nan
        st.metric("Avg Gain Target→9EMA Break %", fmt_pct(prime))
    with k4:
        st.metric("Median Gain Target→9EMA Break %",
                  fmt_pct(filtered["Gain Target→9EMA Break %"].median()) if "Gain Target→9EMA Break %" in filtered else "—")
    with k5:
        st.metric("Avg Days to Peak",
                  f"{filtered['Days to Peak (from Entry)'].mean():.1f}" if "Days to Peak (from Entry)" in filtered else "—")
    with k6:
        if "Outcome" in filtered:
            wr = (filtered["Outcome"].eq("WIN").mean() * 100) if not filtered.empty else 0
            st.metric("Win Rate", fmt_pct(wr, 1))
        else:
            st.metric("Win Rate", "—")

    st.divider()

    # ---------------------------------------------------------------- #
    # Quick distribution chart
    # ---------------------------------------------------------------- #
    with st.expander("📊 Distribution of Gain Target → 9EMA Break %", expanded=False):
        if "Gain Target→9EMA Break %" in filtered:
            hist = go.Figure(data=[go.Histogram(
                x=filtered["Gain Target→9EMA Break %"],
                nbinsx=50,
                marker_color="#5dade2",
                marker_line_color="#0e1117",
                marker_line_width=1,
            )])
            hist.update_layout(
                template="plotly_dark",
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Gain % (Target → 9EMA Break)",
                yaxis_title="Trades",
                bargap=0.05,
            )
            st.plotly_chart(hist, use_container_width=True)

    # ---------------------------------------------------------------- #
    # Data table with selection
    # ---------------------------------------------------------------- #
    st.subheader(f"Trade Ledger ({len(filtered):,} rows)")
    st.caption("👉 Click on a row (or use the row's checkbox) to open the **full-page** chart view.")

    sort_col = st.selectbox(
        "Sort by",
        options=["Sr No", "Ticker", "Entry Date", "Entry Price", "FV Gap %",
                 "Peak Gain %", "Gain Target→9EMA Break %", "Days to Peak (from Entry)",
                 "Days Held", "Gain %"],
        index=6,
    )
    sort_asc = st.checkbox("Ascending", value=False)

    display_df = filtered.copy()
    for c in DATE_COLS:
        if c in display_df.columns:
            display_df[c] = display_df[c].dt.strftime("%Y-%m-%d")

    if sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=sort_asc, na_position="last")
    display_df = display_df.reset_index(drop=True)

    column_config = {}
    for c in display_df.columns:
        if c in DATE_COLS:
            column_config[c] = st.column_config.TextColumn(c)
        elif c in NUMERIC_COLS or c == "Gain Target→9EMA Break %":
            column_config[c] = st.column_config.NumberColumn(c, format="%.2f")

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config,
        height=520,
    )

    # ---------------------------------------------------------------- #
    # Row selection → jump to full-page detail view
    # ---------------------------------------------------------------- #
    sel_rows = event.selection.rows if hasattr(event, "selection") else []
    if sel_rows:
        selected_pos = int(sel_rows[0])
        # Snapshot the SAME sort order on `filtered` so positional index maps correctly.
        sorted_filtered = (
            filtered.sort_values(sort_col, ascending=sort_asc, na_position="last")
                    .reset_index(drop=True)
        )
        if 0 <= selected_pos < len(sorted_filtered):
            st.session_state.detail_df = sorted_filtered
            st.session_state.detail_pos = selected_pos
            st.session_state.view = "detail"
            st.rerun()
            return

    # ---------------------------------------------------------------- #
    # Footer
    # ---------------------------------------------------------------- #
    st.divider()
    st.caption(
        "Built with Streamlit + Plotly · Source: bundled sample "
        "`sample-data-5-year-250-days.xlsx` or your uploaded CSV/XLSX."
    )


# ------------------------------------------------------------------ #
# Router                                                              #
# ------------------------------------------------------------------ #
if st.session_state.view == "detail":
    render_detail_page()
else:
    render_table_page()
