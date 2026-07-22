"""
FinSight — Interactive Streamlit Dashboard
LLM-Augmented Financial Forecaster with Prophet + LSTM + DeepSeek sentiment.
"""

import os
import sys
import time
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests

# ─── Page config ───

st.set_page_config(
    page_title="FinSight | AI Financial Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Backend URL ───

API_URL = os.getenv("FINSIGHT_API_URL", "http://localhost:8000")


def api_call(endpoint: str, method: str = "GET", json_data: dict = None) -> dict:
    """Call the FinSight backend API."""
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=60)
        else:
            resp = requests.post(url, json=json_data, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"🚫 Cannot connect to FinSight backend at {API_URL}. Is the server running?")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Try a shorter period or ticker.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
        st.error(f"❌ API error: {detail}")
        return None


# ─── Session State ───

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None
if "ticker_history" not in st.session_state:
    st.session_state.ticker_history = []


# ─── Sidebar ───

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=64)
    st.title("FinSight 📈")

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
            help="Enter any yfinance-supported ticker (stocks, ETFs, crypto)",
        ).upper().strip()

        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox(
                "**Historical Period**",
                ["1y", "2y", "5y", "max"],
                index=0,
            )
        with col2:
            forecast_days = st.slider(
                "**Forecast Days**",
                min_value=7,
                max_value=90,
                value=30,
                step=1,
            )

        include_sentiment = st.checkbox("Include DeepSeek Sentiment", value=True)

        run_forecast = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    elif mode == "💼 Portfolio Analysis":
        tickers_input = st.text_area(
            "**Portfolio Tickers** (one per line)",
            value="AAPL\nGOOGL\nMSFT\nAMZN",
            height=120,
            placeholder="AAPL\nGOOGL\nMSFT",
            help="Enter ticker symbols, one per line",
        )
        portfolio_tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]

        weights_input = st.text_area(
            "**Portfolio Weights** (one per line, optional)",
            value="0.25\n0.25\n0.25\n0.25",
            height=100,
            placeholder="0.25\n0.25\n0.25\n0.25",
            help="Weights must match number of tickers. Leave empty for equal weight.",
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
    st.caption(f"API: {API_URL}")


# ─── Main Content ───

st.title("FinSight 📈")
st.caption("LLM-Augmented Financial Forecaster — Prophet · LSTM · DeepSeek")

# Check backend health
health = api_call("/health")
if health:
    ds_status = "✅" if health.get("deepseek_configured") else "⚠️"
    db_status = "✅" if health.get("database_configured") else "⚠️"
    st.info(f"Backend: {health.get('status', 'unknown')} | DeepSeek: {ds_status} | Database: {db_status}")


# ═══════════════════════════════════════════
# STOCK FORECAST MODE
# ═══════════════════════════════════════════

if mode == "📊 Stock Forecast" and run_forecast:
    with st.spinner(f"🔮 Forecasting {ticker}... This may take 1-2 minutes."):
        result = api_call(
            "/forecast",
            method="POST",
            json_data={
                "ticker": ticker,
                "period": period,
                "forecast_days": forecast_days,
                "include_sentiment": include_sentiment,
            },
        )

    if result:
        st.session_state.forecast_result = result
        # Track ticker history
        if ticker not in st.session_state.ticker_history:
            st.session_state.ticker_history.append(ticker)

if st.session_state.forecast_result:
    r = st.session_state.forecast_result

    # ── Header Metrics ──
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Current Price",
            f"${r['current_price']:,.2f}",
        )
    with col2:
        pred = r["predicted_price_30d"]
        delta = r["predicted_change_pct"]
        st.metric(
            f"Predicted ({r['forecast_days']}d)",
            f"${pred:,.2f}",
            f"{delta:+.2f}%",
            delta_color="normal",
        )
    with col3:
        prophet_price = r["prophet_forecast"][-1]["forecast"] if r["prophet_forecast"] else 0
        st.metric("Prophet Pred", f"${prophet_price:,.2f}")
    with col4:
        lstm_price = r["lstm_forecast"][-1]["forecast"] if r["lstm_forecast"] else 0
        st.metric("LSTM Pred", f"${lstm_price:,.2f}")
    with col5:
        sentiment = r.get("sentiment")
        if sentiment:
            score = sentiment["overall_score"]
            emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
            st.metric(
                f"Sentiment {emoji}",
                f"{score:+.2f}",
                sentiment["classification"],
            )

    # ── Main Price Chart ──
    st.markdown("---")
    st.subheader("📈 Price Forecast Chart")

    fig = go.Figure()

    # Historical prices (last 90 days)
    hist_dates = pd.to_datetime(r["historical_dates"])
    hist_prices = r["historical_prices"]
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_prices,
            mode="lines",
            name="Historical",
            line=dict(color="#636efa", width=2),
        )
    )

    # Combined forecast
    fc_dates = pd.to_datetime([p["date"] for p in r["combined_forecast"]])
    fc_vals = [p["forecast"] for p in r["combined_forecast"]]
    fc_lower = [p["lower_bound"] for p in r["combined_forecast"]]
    fc_upper = [p["upper_bound"] for p in r["combined_forecast"]]

    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=fc_vals,
            mode="lines+markers",
            name="Combined Forecast",
            line=dict(color="#00cc96", width=3, dash="dash"),
        )
    )

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=fc_dates.tolist() + fc_dates.tolist()[::-1],
            y=fc_upper + fc_lower[::-1],
            fill="toself",
            fillcolor="rgba(0,204,150,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Interval",
            showlegend=True,
        )
    )

    # Prophet only
    prophet_dates = pd.to_datetime([p["date"] for p in r["prophet_forecast"]])
    prophet_vals = [p["forecast"] for p in r["prophet_forecast"]]
    fig.add_trace(
        go.Scatter(
            x=prophet_dates,
            y=prophet_vals,
            mode="lines",
            name="Prophet",
            line=dict(color="#ab63fa", width=1.5, dash="dot"),
            visible="legendonly",
        )
    )

    # LSTM only
    lstm_dates = pd.to_datetime([p["date"] for p in r["lstm_forecast"]])
    lstm_vals = [p["forecast"] for p in r["lstm_forecast"]]
    fig.add_trace(
        go.Scatter(
            x=lstm_dates,
            y=lstm_vals,
            mode="lines",
            name="LSTM",
            line=dict(color="#ffa15a", width=1.5, dash="dot"),
            visible="legendonly",
        )
    )

    fig.update_layout(
        title=f"{r['ticker']} — {r['forecast_days']}-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        height=500,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Model Metrics ──
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Prophet Metrics")
        pm = r["prophet_metrics"]
        mdf = pd.DataFrame(
            {
                "Metric": ["MAE", "RMSE", "MAPE"],
                "Value": [
                    f"${pm['mae']:,.2f}",
                    f"${pm['rmse']:,.2f}",
                    f"{pm['mape']:.2f}%" if pm.get("mape") else "N/A",
                ],
            }
        )
        st.dataframe(mdf, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("🤖 LSTM Metrics")
        lm = r["lstm_metrics"]
        ldf = pd.DataFrame(
            {
                "Metric": ["MAE", "RMSE", "MAPE"],
                "Value": [
                    f"${lm['mae']:,.2f}",
                    f"${lm['rmse']:,.2f}",
                    f"{lm['mape']:.2f}%" if lm.get("mape") else "N/A",
                ],
            }
        )
        st.dataframe(ldf, hide_index=True, use_container_width=True)

    # ── Sentiment Panel ──
    if sentiment:
        st.markdown("---")
        st.subheader("🧠 DeepSeek Sentiment Analysis")

        cols = st.columns([1, 3])
        with cols[0]:
            # Sentiment gauge
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=sentiment["overall_score"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Sentiment Score", "font": {"size": 16}},
                    delta={"reference": 0},
                    gauge={
                        "axis": {"range": [-1, 1], "tickwidth": 1},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [-1, -0.3], "color": "#ff4b4b"},
                            {"range": [-0.3, 0.3], "color": "#f0f0f0"},
                            {"range": [0.3, 1], "color": "#00cc96"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": sentiment["overall_score"],
                        },
                    },
                )
            )
            gauge_fig.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(gauge_fig, use_container_width=True)

        with cols[1]:
            st.markdown(f"**Overall:** {sentiment['classification'].upper()}")
            st.markdown(f"**Summary:** {sentiment['summary']}")

            if sentiment.get("per_headline"):
                headlines_df = pd.DataFrame(sentiment["per_headline"])
                if "title" in headlines_df.columns:
                    headlines_df = headlines_df.rename(
                        columns={"title": "Headline", "sentiment": "Sentiment", "score": "Score"}
                    )
                    st.dataframe(
                        headlines_df[["Headline", "Sentiment", "Score"]],
                        hide_index=True,
                        use_container_width=True,
                    )

    # ── Forecast Table ──
    st.markdown("---")
    st.subheader("📋 Detailed Forecast")
    table_data = []
    for i, (p, l, c) in enumerate(
        zip(r["prophet_forecast"], r["lstm_forecast"], r["combined_forecast"])
    ):
        table_data.append(
            {
                "Date": c["date"],
                "Prophet": f"${p['forecast']:,.2f}",
                "LSTM": f"${l['forecast']:,.2f}",
                "Combined": f"${c['forecast']:,.2f}",
                "Lower": f"${c['lower_bound']:,.2f}",
                "Upper": f"${c['upper_bound']:,.2f}",
            }
        )
    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True, height=300)


# ═══════════════════════════════════════════
# PORTFOLIO ANALYSIS MODE
# ═══════════════════════════════════════════

if mode == "💼 Portfolio Analysis" and run_portfolio and portfolio_tickers:
    with st.spinner("📊 Analyzing portfolio... This may take 30-60 seconds."):
        result = api_call(
            "/portfolio",
            method="POST",
            json_data={
                "tickers": portfolio_tickers,
                "weights": portfolio_weights,
                "period": portfolio_period,
            },
        )

    if result:
        st.session_state.portfolio_result = result

if st.session_state.portfolio_result:
    pr = st.session_state.portfolio_result

    st.markdown("---")

    # ── Portfolio Metrics Cards ──
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

    # ── Cumulative Returns Chart ──
    st.markdown("---")
    st.subheader("📈 Cumulative Returns")

    cum_returns = pr["cumulative_returns"]
    dates = pd.to_datetime(pr["dates"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[(v - 1) * 100 for v in cum_returns],  # Convert to percentage
            mode="lines",
            fill="tozeroy",
            name="Portfolio",
            line=dict(color="#636efa", width=2),
            fillcolor="rgba(99,110,250,0.1)",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Portfolio Cumulative Return (%)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        height=450,
        template="plotly_dark",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Allocation Pie ──
    col1, col2 = st.columns([1, 1])
    with col1:
        pie_fig = go.Figure(
            go.Pie(
                labels=pr["tickers"],
                values=pr["weights"],
                hole=0.4,
                textinfo="label+percent",
            )
        )
        pie_fig.update_layout(
            title="Portfolio Allocation",
            height=350,
            template="plotly_dark",
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    with col2:
        # Risk decomposition bar chart
        risk_fig = go.Figure(
            go.Bar(
                x=["Sharpe", "Max DD", "VaR 95", "Volatility"],
                y=[
                    m["sharpe_ratio"],
                    abs(m["max_drawdown"]) / 10,  # Scale for visibility
                    abs(m["var_95"]),
                    m["annualized_volatility"],
                ],
                marker_color=["#00cc96", "#ff4b4b", "#ffa15a", "#ab63fa"],
                text=[
                    f"{m['sharpe_ratio']:.2f}",
                    f"{m['max_drawdown']:.1f}%",
                    f"{m['var_95']:.2f}%",
                    f"{m['annualized_volatility']:.1f}%",
                ],
                textposition="outside",
            )
        )
        risk_fig.update_layout(
            title="Risk Metrics Overview",
            height=350,
            template="plotly_dark",
            yaxis_title="Value",
        )
        st.plotly_chart(risk_fig, use_container_width=True)


# ═══════════════════════════════════════════
# SENTIMENT ANALYSIS MODE
# ═══════════════════════════════════════════

if mode == "🔬 Sentiment Analysis" and run_sentiment:
    with st.spinner(f"🔍 Analyzing sentiment for {sent_ticker}..."):
        result = api_call(f"/sentiment/{sent_ticker}")

    if result:
        st.markdown("---")
        st.subheader(f"🧠 Sentiment Analysis: {result['ticker']}")

        col1, col2 = st.columns([1, 2])
        with col1:
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["overall_score"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Sentiment Score", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [-1, 1]},
                        "steps": [
                            {"range": [-1, -0.3], "color": "#ff4b4b"},
                            {"range": [-0.3, 0.3], "color": "#f0f0f0"},
                            {"range": [0.3, 1], "color": "#00cc96"},
                        ],
                    },
                )
            )
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
                st.dataframe(
                    pd.DataFrame(result["per_headline"]),
                    hide_index=True,
                    use_container_width=True,
                )


# ─── Footer ───

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** FinSight is for educational and demonstration purposes only. "
    "This is not financial advice. Past performance and predictions do not guarantee future results."
)
