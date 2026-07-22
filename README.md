# FinSight 📈

**LLM-Augmented Financial Forecaster** — combining traditional time-series ML (Prophet, LSTM) with DeepSeek-powered news sentiment analysis for stock/crypto prediction.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?logo=streamlit)](https://finsight-yyprnpfbgpuw9e729sqhxt.streamlit.app)

## 🏗 Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        YF[yfinance API] --> DF[Data Fetcher]
        YF --> NH[News Headlines]
    end

    subgraph "ML Models"
        DF --> PROPHET[Prophet Model]
        DF --> LSTM[LSTM PyTorch]
        PROPHET --> |trend + seasonality|ENS[Ensemble]
        LSTM --> |deep learning|ENS
    end

    subgraph "LLM Layer"
        NH --> DSK[DeepSeek API]
        DSK --> SA[Sentiment Analysis]
        SA --> |bullish/bearish/neutral|ENS
    end

    subgraph "Output"
        ENS --> CF[Combined Forecast]
        DF --> PM[Portfolio Metrics]
        CF --> DB[(Neon PostgreSQL)]
        SA --> DB
    end

    subgraph "Frontend"
        API[FastAPI Backend] --> UI[Streamlit Dashboard]
        UI --> CH[Interactive Charts]
        UI --> MC[Metrics Cards]
        UI --> PG[Portfolio Gauge]
    end

    CF --> API
    PM --> API
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Stock/Crypto Search** | Enter any yfinance-supported ticker, fetch historical OHLCV data |
| 🔮 **Dual Forecasting** | Prophet (trend + seasonality) + LSTM (deep learning) |
| 📈 **Model Comparison** | Side-by-side metrics: MAE, RMSE, MAPE |
| 🧠 **LLM Sentiment** | DeepSeek analyzes news headlines for bullish/bearish/neutral signals |
| 🎯 **Combined Signal** | Weighted blend: 70% model forecast + 30% sentiment adjustment |
| 💰 **Portfolio Analytics** | Sharpe ratio, max drawdown, VaR (95%), volatility, returns |
| 📉 **Interactive Charts** | Plotly-powered forecast overlays, confidence bands, risk gauges |
| 🗄 **Persistent Storage** | Neon PostgreSQL — stores forecasts and sentiment history for backtesting |

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **Time-Series** | Facebook Prophet, PyTorch LSTM |
| **LLM** | DeepSeek API (`deepseek-chat`) |
| **Data** | yfinance, Pandas, NumPy |
| **Database** | Neon PostgreSQL (asyncpg) |
| **Frontend** | Streamlit, Plotly |
| **Config** | python-dotenv |

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/finsight.git
cd finsight

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required: DeepSeek API key (get it from https://platform.deepseek.com)
DEEPSEEK_API_KEY=sk-your-api-key

# Optional: Neon PostgreSQL (for persistence — works without it)
DATABASE_URL=postgresql://user:pass@ep-xxxx.us-east-2.aws.neon.tech/finsight
```

### 4. Run

**Start the backend (terminal 1):**
```bash
cd backend
python main.py
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

**Start the frontend (terminal 2):**
```bash
cd frontend
streamlit run app.py
# Dashboard at http://localhost:8501
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Backend health check |
| `POST` | `/forecast` | Generate combined forecast |
| `GET` | `/forecast/{ticker}` | Quick forecast (GET) |
| `POST` | `/sentiment` | Analyze news sentiment |
| `GET` | `/sentiment/{ticker}` | Quick sentiment (GET) |
| `POST` | `/portfolio` | Portfolio risk metrics |

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | For sentiment | — | DeepSeek API key |
| `DATABASE_URL` | No | — | Neon PostgreSQL connection string |
| `FINSIGHT_API_URL` | No | `http://localhost:8000` | Backend URL (frontend only) |

## 📸 Screenshots

> *(Add screenshots of your running dashboard here)*

- `screenshots/forecast.png` — Main forecast view with Prophet + LSTM overlay
- `screenshots/sentiment.png` — DeepSeek sentiment gauge and headline analysis
- `screenshots/portfolio.png` — Portfolio risk metrics dashboard

## 🧪 Example Queries

```python
# Forecast Apple stock
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "period": "1y", "forecast_days": 30, "include_sentiment": true}'

# Quick sentiment for Bitcoin
curl http://localhost:8000/sentiment/BTC-USD

# Portfolio analysis
curl -X POST http://localhost:8000/portfolio \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "GOOGL", "MSFT", "AMZN"], "period": "1y"}'
```

## ⚠️ Disclaimer

FinSight is an **educational and demonstration project**. It is not financial advice. Forecasts are based on historical patterns and sentiment signals — they should not be the sole basis for investment decisions. Past performance does not guarantee future results.

## 📄 License

MIT © 2024

---

Built for the **UAE AI/Quant job market** — showcasing LLM + ML integration, financial engineering, and full-stack data applications.
