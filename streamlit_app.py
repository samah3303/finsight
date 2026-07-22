"""
FinSight — Streamlit Cloud Entry Point
LLM-Augmented Financial Forecaster with Prophet + LSTM + DeepSeek sentiment.
Imports backend modules directly (no HTTP to localhost).
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Ensure backend is importable ──────────────────────────
BACKEND_PATH = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

from config import config

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="FinSight | AI Financial Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────
from data_fetcher import (
    fetch_historical_data,
    fetch_multi_ticker_data,
    fetch_news_headlines,
    get_current_price,
)
from prophet_model import run_prophet_pipeline
from lstm_model import run_lstm_pipeline
from sentiment import analyze_sentiment as _analyze_sentiment_async, apply_sentiment_adjustment
from portfolio import calculate_portfolio_metrics

# ── Async helper ──────────────────────────────────────────
def _run_async(coro):
    """Run an async coroutine from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Session State ─────────────────────────────────────────
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None
if "ticker_history" not in st.session_state:
    st.session_state.ticker_history = []


# ── Direct backend helpers ────────────────────────────────
def _run_forecast(ticker: str, period: str, forecast_days: int, include_sentiment: bool) -> dict | None:
    """Run the full forecast pipeline directly."""
    try:
        df = fetch_historical_data(ticker, period=period)
    except ValueError as e:
        st.error(str(e))
        return None

    if len(df) < config.lstm_sequence_length + 10:
        st.error(f"Insufficient data: need {config.lstm_sequence_length + 10} trading days, got {len(df)}")
        return None

    current_price = float(df["Close"].iloc[-1])
    last_date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]

    # Prophet
    prophet_points, prophet_metrics, _ = run_prophet_pipeline(df, forecast_days)

    # LSTM
    lstm_points, lstm_metrics, _ = run_lstm_pipeline(
        df["Close"].values.astype(np.float64),
        forecast_days=forecast_days,
        last_date=last_date,
        ticker=ticker,
    )

    # Sentiment
    sentiment = None
    if include_sentiment:
        try:
            sentiment = _run_async(_analyze_sentiment_async(ticker))
        except Exception as e:
            sentiment = {
                "ticker": ticker,
                "headlines": [],
                "overall_score": 0.0,
                "classification": "neutral",
                "per_headline": [],
                "summary": f"Sentiment analysis unavailable: {e}",
            }

    sentiment_score = sentiment["overall_score"] if sentiment else 0.0

    # Combine forecasts
    avg_forecast = []
    for pp, lp in zip(prophet_points, lstm_points):
        avg_forecast.append((pp.forecast + lp.forecast) / 2.0)

    combined_values = apply_sentiment_adjustment(
        avg_forecast, sentiment_score, current_price,
        model_weight=config.model_weight,
    )

    combined_points = []
    for i, (pp, cv) in enumerate(zip(prophet_points, combined_values)):
        combined_points.append({
            "date": pp.date,
            "forecast": cv,
            "lower_bound": pp.lower_bound if pp.lower_bound else cv * 0.95,
            "upper_bound": pp.upper_bound if pp.upper_bound else cv * 1.05,
        })

    predicted_price_30d = combined_values[-1] if combined_values else current_price
    predicted_change_pct = round((predicted_price_30d / current_price - 1) * 100, 2)

    # Historical data
    historical_dates = [d.date() if hasattr(d, "date") else d for d in df.index[-90:]]
    historical_prices = [round(float(p), 4) for p in df["Close"].tail(90).tolist()]

    return {
        "ticker": ticker,
        "forecast_days": forecast_days,
        "historical_dates": historical_dates,
        "historical_prices": historical_prices,
        "prophet_forecast": [{"date": p.date, "forecast": p.forecast, "lower_bound": p.lower_bound, "upper_bound": p.upper_bound} for p in prophet_points],
        "lstm_forecast": [{"date": p.date, "forecast": p.forecast, "lower_bound": p.lower_bound, "upper_bound": p.upper_bound} for p in lstm_points],
        "combined_forecast": combined_points,
        "prophet_metrics": {"mae": prophet_metrics.mae, "rmse": prophet_metrics.rmse, "mape": prophet_metrics.mape},
        "lstm_metrics": {"mae": lstm_metrics.mae, "rmse": lstm_metrics.rmse, "mape": lstm_metrics.mape},
        "sentiment": sentiment,
        "current_price": round(current_price, 2),
        "predicted_price_30d": round(predicted_price_30d, 2),
        "predicted_change_pct": predicted_change_pct,
    }


def _run_portfolio(tickers: list[str], weights: list[float] | None, period: str) -> dict | None:
    """Calculate portfolio metrics directly."""
    try:
        prices = fetch_multi_ticker_data(tickers, period=period)
    except ValueError as e:
        st.error(str(e))
        return None

    if len(prices.columns) < 1:
        st.error("No valid ticker data returned")
        return None

    metrics = calculate_portfolio_metrics(prices, weights=weights, risk_free_rate=config.risk_free_rate)

    daily_returns = prices.pct_change().dropna()
    if weights is None:
        n = len(prices.columns)
        w = [1.0 / n] * n
    else:
        total = sum(weights)
        w = [x / total for x in weights]
    if len(w) < len(prices.columns):
        remaining = 1.0 - sum(w)
        w.extend([remaining / (len(prices.columns) - len(w))] * (len(prices.columns) - len(w)))

    port_returns = (daily_returns * w).sum(axis=1)
    cumulative = (1 + port_returns).cumprod()

    dates_list = [d.date() if hasattr(d, "date") else d for d in cumulative.index]
    cum_vals = [round(float(v), 6) for v in cumulative.tolist()]

    return {
        "tickers": tickers,
        "weights": [round(x, 4) for x in w[:len(tickers)]],
        "cumulative_returns": cum_vals,
        "dates": dates_list,
        "metrics": {
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "annualized_volatility": metrics.annualized_volatility,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "var_95": metrics.var_95,
            "best_day": metrics.best_day,
            "worst_day": metrics.worst_day,
            "positive_days_pct": metrics.positive_days_pct,
        },
    }


def _run_sentiment_analysis(ticker: str) -> dict | None:
    """Analyze sentiment directly."""
    try:
        result = _run_async(_analyze_sentiment_async(ticker))
        return {
            "ticker": result["ticker"],
            "headlines": result["headlines"],
            "overall_score": result["overall_score"],
            "classification": result["classification"],
            "per_headline": result["per_headline"],
            "summary": result["summary"],
        }
    except Exception as e:
        st.error(f"Sentiment analysis failed: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=64)
    st.title("FinSight 📈")

    # API Key status
    if config.deepseek_api_key:
        st.success("🔑 DeepSeek API configured")
    else:
        st.warning("⚠️ DeepSeek API key missing — sentiment disabled")

    st.markdown("---")
    mode = st.radio(
        "**Mode**",
        ["📊 Stock Forecast", "💼 Portfolio Analysis", "🔬 Sentiment Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if mode == "📊 Stock Forecast":
        ticker = st.text_input(
            "**Ticker Symbol**",
            value="AAPL",
            placeholder="e.g., AAPL, GOOGL, BTC-USD",
        ).upper().strip()

        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("**Historical Period**", ["1y", "2y", "5y", "max"], index=0)
        with col2:
            forecast_days = st.slider("**Forecast Days**", min_value=7, max_value=90, value=30, step=1)

        include_sentiment = st.checkbox("Include DeepSeek Sentiment", value=True)
        run_forecast = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    elif mode == "💼 Portfolio Analysis":
        tickers_input = st.text_area(
            "**Portfolio Tickers** (one per line)",
            value="AAPL\nGOOGL\nMSFT\nAMZN",
            height=120,
        )
        portfolio_tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]

        weights_input = st.text_area(
            "**Portfolio Weights** (one per line, optional)",
            value="0.25\n0.25\n0.25\n0.25",
            height=100,
        )

        try:
            portfolio_weights = (
                [float(w.strip()) for w in weights_input.split("\n") if w.strip()]
                if weights_input.strip()
                else None
            )
        except ValueError:
            st.error("Weights must be numeric values")
            portfolio_weights = None

        portfolio_period = st.selectbox("**Historical Period**", ["1y", "2y", "5y", "max"], index=0)
        run_portfolio = st.button("📊 Analyze Portfolio", type="primary", use_container_width=True)

    elif mode == "🔬 Sentiment Analysis":
        sent_ticker = st.text_input(
            "**Ticker Symbol**",
            value="AAPL",
            placeholder="e.g., AAPL, TSLA",
        ).upper().strip()
        run_sentiment = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("FinSight v1.0 | Prophet + LSTM + DeepSeek")


# ── Main Content ──────────────────────────────────────────
st.title("FinSight 📈")
st.caption("LLM-Augmented Financial Forecaster — Prophet · LSTM · DeepSeek")

# Status
ds_status = "✅" if config.deepseek_api_key else "⚠️"
db_status = "✅" if config.database_url else "⚠️"
st.info(f"DeepSeek: {ds_status} | Database: {db_status} | Running in direct mode")


# ═══════════════════════════════════════════
# STOCK FORECAST MODE
# ═══════════════════════════════════════════

if mode == "📊 Stock Forecast" and run_forecast:
    with st.spinner(f"🔮 Forecasting {ticker}... This may take 1-2 minutes."):
        result = _run_forecast(ticker, period, forecast_days, include_sentiment)
    if result:
        st.session_state.forecast_result = result
        if ticker not in st.session_state.ticker_history:
            st.session_state.ticker_history.append(ticker)

if st.session_state.forecast_result:
    r = st.session_state.forecast_result
    st.markdown("---")

    # Header Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Current Price", f"${r['current_price']:,.2f}")
    with col2:
        pred = r["predicted_price_30d"]
        delta = r["predicted_change_pct"]
        st.metric(f"Predicted ({r['forecast_days']}d)", f"${pred:,.2f}", f"{delta:+.2f}%")
    with col3:
        pp = r["prophet_forecast"][-1]["forecast"] if r["prophet_forecast"] else 0
        st.metric("Prophet Pred", f"${pp:,.2f}")
    with col4:
        lp = r["lstm_forecast"][-1]["forecast"] if r["lstm_forecast"] else 0
        st.metric("LSTM Pred", f"${lp:,.2f}")
    with col5:
        sentiment = r.get("sentiment")
        if sentiment:
            score = sentiment["overall_score"]
            emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
            st.metric(f"Sentiment {emoji}", f"{score:+.2f}", sentiment["classification"])

    # Main Price Chart
    st.markdown("---")
    st.subheader("📈 Price Forecast Chart")

    fig = go.Figure()
    hist_dates = pd.to_datetime(r["historical_dates"])
    hist_prices = r["historical_prices"]
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_prices, mode="lines",
                             name="Historical", line=dict(color="#636efa", width=2)))

    fc_dates = pd.to_datetime([p["date"] for p in r["combined_forecast"]])
    fc_vals = [p["forecast"] for p in r["combined_forecast"]]
    fc_lower = [p["lower_bound"] for p in r["combined_forecast"]]
    fc_upper = [p["upper_bound"] for p in r["combined_forecast"]]

    fig.add_trace(go.Scatter(x=fc_dates, y=fc_vals, mode="lines+markers",
                             name="Combined Forecast", line=dict(color="#00cc96", width=3, dash="dash")))

    fig.add_trace(go.Scatter(
        x=fc_dates.tolist() + fc_dates.tolist()[::-1],
        y=fc_upper + fc_lower[::-1],
        fill="toself", fillcolor="rgba(0,204,150,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Interval"))

    prophet_dates = pd.to_datetime([p["date"] for p in r["prophet_forecast"]])
    prophet_vals = [p["forecast"] for p in r["prophet_forecast"]]
    fig.add_trace(go.Scatter(x=prophet_dates, y=prophet_vals, mode="lines",
                             name="Prophet", line=dict(color="#ab63fa", width=1.5, dash="dot"),
                             visible="legendonly"))

    lstm_dates = pd.to_datetime([p["date"] for p in r["lstm_forecast"]])
    lstm_vals = [p["forecast"] for p in r["lstm_forecast"]]
    fig.add_trace(go.Scatter(x=lstm_dates, y=lstm_vals, mode="lines",
                             name="LSTM", line=dict(color="#ffa15a", width=1.5, dash="dot"),
                             visible="legendonly"))

    fig.update_layout(title=f"{r['ticker']} — {r['forecast_days']}-Day Forecast",
                      xaxis_title="Date", yaxis_title="Price ($)",
                      hovermode="x unified", height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Model Metrics
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Prophet Metrics")
        pm = r["prophet_metrics"]
        st.dataframe(pd.DataFrame({
            "Metric": ["MAE", "RMSE", "MAPE"],
            "Value": [f"${pm['mae']:,.2f}", f"${pm['rmse']:,.2f}",
                      f"{pm['mape']:.2f}%" if pm.get("mape") else "N/A"],
        }), hide_index=True, use_container_width=True)
    with col2:
        st.subheader("🤖 LSTM Metrics")
        lm = r["lstm_metrics"]
        st.dataframe(pd.DataFrame({
            "Metric": ["MAE", "RMSE", "MAPE"],
            "Value": [f"${lm['mae']:,.2f}", f"${lm['rmse']:,.2f}",
                      f"{lm['mape']:.2f}%" if lm.get("mape") else "N/A"],
        }), hide_index=True, use_container_width=True)

    # Sentiment Panel
    if sentiment and sentiment.get("per_headline"):
        st.markdown("---")
        st.subheader("🧠 DeepSeek Sentiment Analysis")
        cols = st.columns([1, 3])
        with cols[0]:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=sentiment["overall_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Sentiment Score"},
                delta={"reference": 0},
                gauge={"axis": {"range": [-1, 1]},
                       "steps": [{"range": [-1, -0.3], "color": "#ff4b4b"},
                                 {"range": [-0.3, 0.3], "color": "#f0f0f0"},
                                 {"range": [0.3, 1], "color": "#00cc96"}]}
            ))
            gauge_fig.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(gauge_fig, use_container_width=True)
        with cols[1]:
            st.markdown(f"**Overall:** {sentiment['classification'].upper()}")
            st.markdown(f"**Summary:** {sentiment['summary']}")
            if sentiment.get("per_headline"):
                hdf = pd.DataFrame(sentiment["per_headline"])
                if "title" in hdf.columns:
                    st.dataframe(hdf[["title", "sentiment", "score"]].rename(
                        columns={"title": "Headline", "sentiment": "Sentiment", "score": "Score"}
                    ), hide_index=True, use_container_width=True)

    # Forecast Table
    st.markdown("---")
    st.subheader("📋 Detailed Forecast")
    table_data = []
    for p, l, c in zip(r["prophet_forecast"], r["lstm_forecast"], r["combined_forecast"]):
        table_data.append({
            "Date": c["date"],
            "Prophet": f"${p['forecast']:,.2f}",
            "LSTM": f"${l['forecast']:,.2f}",
            "Combined": f"${c['forecast']:,.2f}",
            "Lower": f"${c['lower_bound']:,.2f}",
            "Upper": f"${c['upper_bound']:,.2f}",
        })
    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True, height=300)


# ═══════════════════════════════════════════
# PORTFOLIO ANALYSIS MODE
# ═══════════════════════════════════════════

if mode == "💼 Portfolio Analysis" and run_portfolio and portfolio_tickers:
    with st.spinner("📊 Analyzing portfolio..."):
        result = _run_portfolio(portfolio_tickers, portfolio_weights, portfolio_period)
    if result:
        st.session_state.portfolio_result = result

if st.session_state.portfolio_result:
    pr = st.session_state.portfolio_result
    st.markdown("---")
    m = pr["metrics"]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Return", f"{m['total_return']:+.2f}%")
    with col2:
        st.metric("Ann. Return", f"{m['annualized_return']:+.2f}%")
    with col3:
        st.metric("Volatility", f"{m['annualized_volatility']:.2f}%")
    with col4:
        st.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
    with col5:
        st.metric("Max Drawdown", f"{m['max_drawdown']:.2f}%", delta_color="inverse")
    with col6:
        st.metric("VaR (95%)", f"{m['var_95']:.2f}%", delta_color="inverse")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Day", f"{m['best_day']:+.2f}%")
    with col2:
        st.metric("Worst Day", f"{m['worst_day']:+.2f}%")
    with col3:
        st.metric("Positive Days", f"{m['positive_days_pct']:.1f}%")
    with col4:
        st.caption(f"Tickers: {', '.join(pr['tickers'])}")

    # Cumulative Returns
    st.markdown("---")
    st.subheader("📈 Cumulative Returns")
    fig = go.Figure()
    cum_returns = pr["cumulative_returns"]
    dates = pd.to_datetime(pr["dates"])
    fig.add_trace(go.Scatter(
        x=dates, y=[(v - 1) * 100 for v in cum_returns],
        mode="lines", fill="tozeroy", name="Portfolio",
        line=dict(color="#636efa", width=2), fillcolor="rgba(99,110,250,0.1)"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="Portfolio Cumulative Return (%)", height=450,
                      template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Allocation Pie
    col1, col2 = st.columns([1, 1])
    with col1:
        pie_fig = go.Figure(go.Pie(labels=pr["tickers"], values=pr["weights"], hole=0.4, textinfo="label+percent"))
        pie_fig.update_layout(title="Portfolio Allocation", height=350, template="plotly_dark")
        st.plotly_chart(pie_fig, use_container_width=True)
    with col2:
        risk_fig = go.Figure(go.Bar(
            x=["Sharpe", "Max DD", "VaR 95", "Volatility"],
            y=[m["sharpe_ratio"], abs(m["max_drawdown"]) / 10, abs(m["var_95"]), m["annualized_volatility"]],
            marker_color=["#00cc96", "#ff4b4b", "#ffa15a", "#ab63fa"],
            text=[f"{m['sharpe_ratio']:.2f}", f"{m['max_drawdown']:.1f}%",
                  f"{m['var_95']:.2f}%", f"{m['annualized_volatility']:.1f}%"],
            textposition="outside"))
        risk_fig.update_layout(title="Risk Metrics Overview", height=350, template="plotly_dark")
        st.plotly_chart(risk_fig, use_container_width=True)


# ═══════════════════════════════════════════
# SENTIMENT ANALYSIS MODE
# ═══════════════════════════════════════════

if mode == "🔬 Sentiment Analysis" and run_sentiment:
    with st.spinner(f"🔍 Analyzing sentiment for {sent_ticker}..."):
        result = _run_sentiment_analysis(sent_ticker)

    if result:
        st.markdown("---")
        st.subheader(f"🧠 Sentiment Analysis: {result['ticker']}")

        col1, col2 = st.columns([1, 2])
        with col1:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number", value=result["overall_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Sentiment Score"},
                gauge={"axis": {"range": [-1, 1]},
                       "steps": [{"range": [-1, -0.3], "color": "#ff4b4b"},
                                 {"range": [-0.3, 0.3], "color": "#f0f0f0"},
                                 {"range": [0.3, 1], "color": "#00cc96"}]}))
            gauge_fig.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col2:
            st.markdown(f"### {result['classification'].upper()}")
            st.markdown(f"**Score:** {result['overall_score']:+.4f}")
            st.markdown(f"**Summary:** {result['summary']}")
            if result.get("headlines"):
                st.markdown("**Analyzed Headlines:**")
                for i, h in enumerate(result["headlines"]):
                    st.caption(f"{i+1}. {h}")
            if result.get("per_headline"):
                st.markdown("**Per-Headline Breakdown:**")
                st.dataframe(pd.DataFrame(result["per_headline"]), hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** FinSight is for educational and demonstration purposes only. "
    "This is not financial advice. Past performance and predictions do not guarantee future results."
)
