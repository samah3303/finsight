"""
LSTM-based time-series forecasting for FinSight.
Uses PyTorch with a sliding-window approach (60-day lookback → 1-day prediction).
"""

import logging
import os
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from models import ForecastPoint, ModelMetrics
from config import config

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─── Model Definition ───

class LSTMForecaster(nn.Module):
    """Stacked LSTM with a linear output head for single-step price prediction."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last_hidden)
        return out.squeeze(-1)  # (batch,)


# ─── Data Preparation ───

def _create_sequences(
    data: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences. X: (n, seq_len, 1), y: (n,)."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def _prepare_lstm_data(
    prices: np.ndarray, seq_length: int = 60
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
    """Scale data, create sequences, split into train/test."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    X, y = _create_sequences(scaled, seq_length)

    # Split: 85% train, 15% test
    split = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Reshape to (batch, seq_len, input_size=1)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    logger.info("LSTM data prepared — train: %d, test: %d", len(X_train), len(X_test))
    return X_train_t, y_train_t, X_test_t, y_test_t, scaler


# ─── Training ───

def train_lstm(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
) -> LSTMForecaster:
    """Train the LSTM model and return the best model by validation loss."""
    model = LSTMForecaster(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(DEVICE)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test.to(DEVICE))
            val_loss = criterion(val_preds, y_test.to(DEVICE)).item()
        model.train()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.debug(
                "Epoch %d/%d — train loss: %.6f, val loss: %.6f",
                epoch + 1,
                epochs,
                avg_loss,
                val_loss,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    if best_state:
        model.load_state_dict(best_state)

    logger.info("LSTM training complete — best val loss: %.6f", best_loss)
    return model


# ─── Evaluation ───

def evaluate_lstm(
    model: LSTMForecaster,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    scaler: MinMaxScaler,
) -> ModelMetrics:
    """Evaluate LSTM on test set, returning inverse-scaled metrics."""
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()
    actual = y_test.numpy()

    # Inverse transform
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_inv = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()

    mae = float(np.mean(np.abs(preds_inv - actual_inv)))
    rmse = float(np.sqrt(np.mean((preds_inv - actual_inv) ** 2)))

    mask = actual_inv != 0
    mape = (
        float(np.mean(np.abs((preds_inv[mask] - actual_inv[mask]) / actual_inv[mask])) * 100)
        if mask.any()
        else None
    )

    return ModelMetrics(mae=round(mae, 4), rmse=round(rmse, 4), mape=round(mape, 2) if mape else None)


# ─── Forecast Future ───

def forecast_lstm_future(
    model: LSTMForecaster,
    last_sequence: np.ndarray,
    scaler: MinMaxScaler,
    steps: int = 30,
    last_date: Optional[date] = None,
) -> list[ForecastPoint]:
    """
    Generate multi-step future forecasts by iteratively appending predictions.

    Args:
        model: Trained LSTMForecaster
        last_sequence: Last `seq_length` scaled prices (shape: seq_length,)
        scaler: Fitted MinMaxScaler for inverse transform
        steps: Number of future days to predict
        last_date: Date of the last known data point

    Returns:
        List of ForecastPoint
    """
    model.eval()
    current_seq = last_sequence.copy()
    predictions = []

    for i in range(steps):
        inp = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(inp).item()
        predictions.append(pred_scaled)
        # Shift window: drop oldest, append new prediction
        current_seq = np.append(current_seq[1:], pred_scaled)

    preds_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    base_date = last_date or date.today()
    forecast_points = []
    for i, p in enumerate(preds_inv):
        forecast_points.append(
            ForecastPoint(
                date=base_date + timedelta(days=i + 1),
                forecast=round(float(p), 4),
            )
        )
    return forecast_points


def save_checkpoint(model: LSTMForecaster, ticker: str) -> str:
    path = os.path.join(CHECKPOINT_DIR, f"lstm_{ticker}.pt")
    torch.save(model.state_dict(), path)
    logger.info("Saved LSTM checkpoint: %s", path)
    return path


def load_checkpoint(ticker: str) -> Optional[LSTMForecaster]:
    path = os.path.join(CHECKPOINT_DIR, f"lstm_{ticker}.pt")
    if not os.path.exists(path):
        return None
    model = LSTMForecaster()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    logger.info("Loaded LSTM checkpoint: %s", path)
    return model


# ─── Pipeline ───

def run_lstm_pipeline(
    prices: np.ndarray,
    forecast_days: int = 30,
    last_date: Optional[date] = None,
    ticker: str = "default",
) -> tuple[list[ForecastPoint], ModelMetrics, LSTMForecaster]:
    """
    End-to-end LSTM pipeline: prepare data, train, evaluate, forecast.

    Returns:
        (forecast_points, metrics, trained_model)
    """
    X_train, y_train, X_test, y_test, scaler = _prepare_lstm_data(
        prices, seq_length=config.lstm_sequence_length
    )

    model = train_lstm(
        X_train, y_train, X_test, y_test,
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        epochs=config.lstm_epochs,
    )

    metrics = evaluate_lstm(model, X_test, y_test, scaler)

    # Last window for future forecasting
    scaled_all = scaler.transform(prices.reshape(-1, 1)).flatten()
    last_seq = scaled_all[-config.lstm_sequence_length:]

    points = forecast_lstm_future(model, last_seq, scaler, forecast_days, last_date)

    save_checkpoint(model, ticker)

    return points, metrics, model
