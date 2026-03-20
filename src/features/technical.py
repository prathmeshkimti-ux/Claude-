"""
Technical Feature Engineering
================================
Computes all technical indicators from OHLCV data and returns a single
feature DataFrame aligned to the price index.

Categories:
  - Trend        : SMA, EMA, MACD, ADX, golden/death cross flags
  - Momentum     : RSI, Stochastic, Williams %R, Rate of Change
  - Volatility   : Bollinger Bands, ATR, Keltner Channels, normalised range
  - Volume       : OBV, VWAP deviation, Volume Oscillator, Accumulation/Distribution
  - Structure    : Support/Resistance levels, breakout flags, liquidity zones
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """
    Compute a rich set of technical indicators from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.
    cfg : DataConfig | None
        Indicator parameter overrides; defaults to global config.
    """

    def __init__(self, df: pd.DataFrame, cfg=None) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.df = df.copy()
        self.cfg = cfg or config.data

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compute_all(self) -> pd.DataFrame:
        """Compute every indicator category and return merged DataFrame."""
        frames = [
            self._trend_features(),
            self._momentum_features(),
            self._volatility_features(),
            self._volume_features(),
            self._market_structure_features(),
        ]
        result = pd.concat([f for f in frames if not f.empty], axis=1)
        return result

    # ------------------------------------------------------------------
    # Trend indicators
    # ------------------------------------------------------------------

    def _trend_features(self) -> pd.DataFrame:
        close = self.df["close"]
        features = pd.DataFrame(index=self.df.index)

        # SMAs
        for p in self.cfg.sma_periods:
            features[f"sma_{p}"] = close.rolling(p).mean()
            features[f"price_to_sma_{p}"] = (close - features[f"sma_{p}"]) / (features[f"sma_{p}"] + 1e-9)

        # EMAs
        for p in self.cfg.ema_periods:
            features[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()
            features[f"price_to_ema_{p}"] = (close - features[f"ema_{p}"]) / (features[f"ema_{p}"] + 1e-9)

        # Golden / death cross (50-day EMA vs 200-day EMA)
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        features["golden_cross"] = (ema50 > ema200).astype(int)
        features["cross_distance"] = (ema50 - ema200) / (ema200 + 1e-9)

        # MACD
        macd_line, signal_line, histogram = self._macd(close)
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_hist"] = histogram
        features["macd_hist_chg"] = histogram.diff()

        # ADX (Average Directional Index)
        adx, plus_di, minus_di = self._adx(self.df, self.cfg.adx_period)
        features["adx"] = adx
        features["plus_di"] = plus_di
        features["minus_di"] = minus_di
        features["di_diff"] = plus_di - minus_di

        # Trend strength classification
        features["trend_strength"] = pd.cut(
            adx,
            bins=[0, 20, 40, 60, 100],
            labels=[0, 1, 2, 3],
            right=True,
        ).astype(float)

        return features

    # ------------------------------------------------------------------
    # Momentum indicators
    # ------------------------------------------------------------------

    def _momentum_features(self) -> pd.DataFrame:
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        features = pd.DataFrame(index=self.df.index)

        # RSI
        rsi = self._rsi(close, self.cfg.rsi_period)
        features["rsi"] = rsi
        features["rsi_overbought"] = (rsi > 70).astype(int)
        features["rsi_oversold"] = (rsi < 30).astype(int)
        features["rsi_norm"] = (rsi - 50) / 50.0   # -1 → +1

        # Stochastic oscillator (%K and %D)
        stoch_k, stoch_d = self._stochastic(high, low, close)
        features["stoch_k"] = stoch_k
        features["stoch_d"] = stoch_d
        features["stoch_signal"] = (stoch_k > stoch_d).astype(int)

        # Williams %R
        williams_r = self._williams_r(high, low, close)
        features["williams_r"] = williams_r

        # Rate of Change (ROC)
        for p in [3, 5, 10, 20]:
            features[f"roc_{p}"] = close.pct_change(p)

        # Price acceleration
        features["momentum_5"] = close.diff(5)
        features["momentum_10"] = close.diff(10)
        features["price_accel"] = close.diff().diff()

        # CCI (Commodity Channel Index)
        features["cci"] = self._cci(high, low, close)

        return features

    # ------------------------------------------------------------------
    # Volatility indicators
    # ------------------------------------------------------------------

    def _volatility_features(self) -> pd.DataFrame:
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        features = pd.DataFrame(index=self.df.index)

        # Bollinger Bands
        bb_mid, bb_upper, bb_lower, bb_width, bb_pct = self._bollinger_bands(
            close, self.cfg.bb_period, self.cfg.bb_std
        )
        features["bb_mid"] = bb_mid
        features["bb_upper"] = bb_upper
        features["bb_lower"] = bb_lower
        features["bb_width"] = bb_width
        features["bb_pct"] = bb_pct         # position within the band (0–1)
        features["bb_squeeze"] = (bb_width < bb_width.rolling(20).quantile(0.1)).astype(int)

        # ATR (Average True Range)
        atr = self._atr(high, low, close, self.cfg.atr_period)
        features["atr"] = atr
        features["atr_pct"] = atr / (close + 1e-9)  # normalised
        features["atr_expansion"] = (atr > atr.rolling(20).mean() * 1.3).astype(int)

        # Normalised daily range
        features["daily_range_pct"] = (high - low) / (close + 1e-9)

        # Annualised realised volatility (20-day)
        log_ret = np.log(close / close.shift(1))
        features["realised_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
        features["realised_vol_5"] = log_ret.rolling(5).std() * np.sqrt(252)

        # Volatility regime
        features["high_vol_regime"] = (
            features["realised_vol_20"] > self.cfg.bb_period / 100   # rough threshold
        ).astype(int)

        # Keltner channel position
        kc_mid = close.ewm(span=20, adjust=False).mean()
        kc_atr = atr * 2
        features["keltner_pos"] = (close - (kc_mid - kc_atr)) / (2 * kc_atr + 1e-9)

        return features

    # ------------------------------------------------------------------
    # Volume indicators
    # ------------------------------------------------------------------

    def _volume_features(self) -> pd.DataFrame:
        close = self.df["close"]
        volume = self.df["volume"]
        high = self.df["high"]
        low = self.df["low"]
        features = pd.DataFrame(index=self.df.index)

        # On Balance Volume
        features["obv"] = self._obv(close, volume)
        features["obv_ema"] = features["obv"].ewm(span=20, adjust=False).mean()
        features["obv_trend"] = (features["obv"] > features["obv_ema"]).astype(int)

        # Volume Oscillator
        vol_short = volume.ewm(span=5, adjust=False).mean()
        vol_long = volume.ewm(span=20, adjust=False).mean()
        features["volume_oscillator"] = (vol_short - vol_long) / (vol_long + 1e-9)

        # Accumulation / Distribution Line
        features["acc_dist"] = self._accumulation_distribution(high, low, close, volume)

        # VWAP deviation (if vwap in df)
        if "vwap" in self.df.columns:
            features["vwap_deviation"] = (close - self.df["vwap"]) / (self.df["vwap"] + 1e-9)

        # Force Index
        features["force_index"] = close.diff(1) * volume

        # Money Flow Index (MFI)
        features["mfi"] = self._mfi(high, low, close, volume)

        # Volume spike (z-score)
        vol_z = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
        features["volume_zscore"] = vol_z
        features["volume_spike"] = (vol_z > 2.0).astype(int)

        # Price-Volume divergence
        features["pv_divergence"] = np.sign(close.diff()) != np.sign(volume.diff())
        features["pv_divergence"] = features["pv_divergence"].astype(int)

        return features

    # ------------------------------------------------------------------
    # Market structure (support / resistance / breakout)
    # ------------------------------------------------------------------

    def _market_structure_features(self) -> pd.DataFrame:
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        features = pd.DataFrame(index=self.df.index)

        # Rolling pivot levels (20-bar)
        features["pivot"] = (high.rolling(20).max() + low.rolling(20).min() + close) / 3
        features["resistance"] = high.rolling(20).max()
        features["support"] = low.rolling(20).min()

        # Distance from support / resistance (normalised)
        r = features["resistance"]
        s = features["support"]
        features["dist_from_resistance"] = (r - close) / (r + 1e-9)
        features["dist_from_support"] = (close - s) / (s + 1e-9)

        # Breakout flags
        features["breakout_up"] = (close > high.rolling(20).max().shift(1)).astype(int)
        features["breakout_down"] = (close < low.rolling(20).min().shift(1)).astype(int)

        # Liquidity zone: high volume near price levels
        vol_ma = self.df["volume"].rolling(20).mean()
        features["high_liquidity_zone"] = (self.df["volume"] > vol_ma * 1.5).astype(int)

        # Swing highs / lows (simple version)
        features["local_high"] = (
            (high == high.rolling(5, center=True).max())
        ).astype(int)
        features["local_low"] = (
            (low == low.rolling(5, center=True).min())
        ).astype(int)

        return features

    # ------------------------------------------------------------------
    # Indicator computation helpers
    # ------------------------------------------------------------------

    def _macd(self, close: pd.Series):
        f, s, sig = self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal
        ema_fast = close.ewm(span=f, adjust=False).mean()
        ema_slow = close.ewm(span=s, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _stochastic(high, low, close, k_period=14, d_period=3):
        low_min = low.rolling(k_period).min()
        high_max = high.rolling(k_period).max()
        stoch_k = 100 * (close - low_min) / (high_max - low_min + 1e-9)
        stoch_d = stoch_k.rolling(d_period).mean()
        return stoch_k, stoch_d

    @staticmethod
    def _williams_r(high, low, close, period=14) -> pd.Series:
        high_max = high.rolling(period).max()
        low_min = low.rolling(period).min()
        return -100 * (high_max - close) / (high_max - low_min + 1e-9)

    @staticmethod
    def _cci(high, low, close, period=20) -> pd.Series:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma_tp) / (0.015 * mean_dev + 1e-9)

    @staticmethod
    def _bollinger_bands(close, period=20, std_mult=2.0):
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        width = (upper - lower) / (mid + 1e-9)
        pct = (close - lower) / (upper - lower + 1e-9)
        return mid, upper, lower, width, pct

    @staticmethod
    def _atr(high, low, close, period=14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _adx(df, period=14):
        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def _obv(close, volume) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def _accumulation_distribution(high, low, close, volume) -> pd.Series:
        clv = ((close - low) - (high - close)) / (high - low + 1e-9)
        return (clv * volume).cumsum()

    @staticmethod
    def _mfi(high, low, close, volume, period=14) -> pd.Series:
        tp = (high + low + close) / 3
        raw_mf = tp * volume
        pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
        mf_ratio = pos_mf.rolling(period).sum() / (neg_mf.rolling(period).sum() + 1e-9)
        return 100 - (100 / (1 + mf_ratio))
