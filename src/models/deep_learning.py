"""
Deep Learning Models: LSTM + Transformer
==========================================
Two sequence models that exploit the temporal ordering of features:

  LSTMModel
    - Stacked bidirectional LSTM
    - Outputs softmax probabilities over 3 classes
    - Takes a lookback window of recent feature vectors

  TransformerModel
    - Multi-head self-attention encoder
    - Positional encoding over time steps
    - Smaller and faster than LSTMModel; good for long sequences

Both models:
  - Accept 3-D input (batch, seq_len, n_features)
  - Output probability dicts {"bullish", "sideways", "bearish"}
  - Support GPU via .to(device) / torch.cuda if available
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logger.warning("PyTorch not installed; deep learning models disabled.")


# ------------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------------

class LSTMModel:
    """
    Bidirectional stacked LSTM classifier for sequential feature data.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    lookback : int
        Number of time steps in each sequence.
    hidden_size : int
    num_layers : int
    dropout : float
    horizon : int
        Prediction horizon (affects target labelling only).
    """

    def __init__(
        self,
        n_features: int,
        lookback: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        horizon: int = 3,
    ) -> None:
        cfg = config.model
        self.n_features = n_features
        self.lookback = lookback or cfg.lstm_lookback
        self.hidden_size = hidden_size or cfg.lstm_hidden_size
        self.num_layers = num_layers or cfg.lstm_num_layers
        self.dropout = dropout or cfg.lstm_dropout
        self.horizon = horizon
        self.device = torch.device("cuda" if TORCH_OK and torch.cuda.is_available() else "cpu")
        self._model: Optional["_LSTMNet"] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = 1e-3,
    ) -> "LSTMModel":
        if not TORCH_OK:
            logger.warning("PyTorch unavailable; LSTMModel.fit() skipped.")
            return self

        cfg = config.model
        epochs = epochs or cfg.lstm_epochs
        batch_size = batch_size or cfg.lstm_batch_size

        X_seq, y_seq = self._make_sequences(X, y)
        if X_seq is None:
            return self

        dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.LongTensor(y_seq),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model = _LSTMNet(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logger.info("LSTM Epoch %d/%d | Loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, float]:
        if not self._fitted or not TORCH_OK:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}

        X_np = X.values
        if len(X_np) < self.lookback:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}

        seq = X_np[-self.lookback:]
        tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # order: [bearish=0, sideways=1, bullish=2]
        return {
            "bullish": round(float(proba[2]), 4),
            "sideways": round(float(proba[1]), 4),
            "bearish": round(float(proba[0]), 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_sequences(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create overlapping sequences of shape (n_samples, lookback, n_features)."""
        X_np = X.values
        y_np = (y.map({-1: 0, 0: 1, 1: 2})).values.astype(int)

        if len(X_np) < self.lookback + 1:
            logger.warning("Insufficient data for LSTM sequences.")
            return None, None

        seqs, labels = [], []
        for i in range(self.lookback, len(X_np)):
            seqs.append(X_np[i - self.lookback : i])
            labels.append(y_np[i])

        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)

    def save(self, path: str) -> None:
        if self._model is not None and TORCH_OK:
            torch.save(self._model.state_dict(), path)
            logger.info("LSTM model saved to %s", path)

    def load(self, path: str) -> None:
        if not TORCH_OK:
            return
        self._model = _LSTMNet(
            self.n_features, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)
        self._model.load_state_dict(torch.load(path, map_location=self.device))
        self._fitted = True
        logger.info("LSTM model loaded from %s", path)


if TORCH_OK:
    class _LSTMNet(nn.Module):
        def __init__(self, n_features, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = nn.Linear(hidden_size * 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            # Attention pooling over time steps
            attn_weights = torch.softmax(self.attention(out), dim=1)
            context = (out * attn_weights).sum(dim=1)
            return self.classifier(context)


# ------------------------------------------------------------------
# Transformer Model
# ------------------------------------------------------------------

class TransformerModel:
    """
    Lightweight Transformer encoder for sequential financial data.

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    d_model : int
        Transformer embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of encoder layers.
    lookback : int
        Sequence length.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lookback: int = None,
        dropout: float = 0.1,
        horizon: int = 3,
    ) -> None:
        cfg = config.model
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lookback = lookback or cfg.lstm_lookback
        self.dropout = dropout
        self.horizon = horizon
        self.device = torch.device("cuda" if TORCH_OK and torch.cuda.is_available() else "cpu")
        self._model = None
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = 5e-4,
    ) -> "TransformerModel":
        if not TORCH_OK:
            return self

        cfg = config.model
        epochs = epochs or cfg.lstm_epochs
        batch_size = batch_size or cfg.lstm_batch_size

        X_seq, y_seq = self._make_sequences(X, y)
        if X_seq is None:
            return self

        dataset = TensorDataset(torch.FloatTensor(X_seq), torch.LongTensor(y_seq))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model = _TransformerNet(
            n_features=self.n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            seq_len=self.lookback,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("Transformer Epoch %d/%d | Loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, float]:
        if not self._fitted or not TORCH_OK:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}

        X_np = X.values
        if len(X_np) < self.lookback:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}

        seq = X_np[-self.lookback:]
        tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {
            "bullish": round(float(proba[2]), 4),
            "sideways": round(float(proba[1]), 4),
            "bearish": round(float(proba[0]), 4),
        }

    def _make_sequences(self, X, y):
        X_np = X.values
        y_np = y.map({-1: 0, 0: 1, 1: 2}).values.astype(int)
        if len(X_np) < self.lookback + 1:
            return None, None
        seqs, labels = [], []
        for i in range(self.lookback, len(X_np)):
            seqs.append(X_np[i - self.lookback : i])
            labels.append(y_np[i])
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)


if TORCH_OK:
    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model, seq_len, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(seq_len, d_model)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, seq_len, d_model)

        def forward(self, x):
            return self.dropout(x + self.pe[:, : x.size(1)])

    class _TransformerNet(nn.Module):
        def __init__(self, n_features, d_model, n_heads, n_layers, seq_len, dropout):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = _PositionalEncoding(d_model, seq_len, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dropout=dropout,
                dim_feedforward=d_model * 4, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 3),
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.transformer(x)
            x = x.mean(dim=1)   # global average pooling
            return self.classifier(x)
