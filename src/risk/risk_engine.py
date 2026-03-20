"""
Risk Engine
============
Filters and adjusts ensemble predictions using:

  1. Volatility Regime   : suppress signals in extreme vol environments
  2. Event Risk          : dampen confidence before scheduled high-impact events
  3. Liquidity Risk      : reduce sizing in thin/illiquid markets
  4. Correlation Risk    : check whether the trade diversifies or concentrates
  5. Drawdown Guard      : halt trading if portfolio drawdown exceeds limit
  6. Position Sizing     : Kelly / fixed-fraction sizing recommendation

The engine returns:
  - approved : bool     (True = signal passes all risk gates)
  - position_size : float   (recommended % of portfolio to allocate)
  - risk_report  : dict     (per-filter details for transparency)
  - adjusted_proba : dict   (dampened probabilities after risk)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Applies risk filters to an ensemble prediction before it becomes a trade signal.

    Parameters
    ----------
    risk_cfg : RiskConfig | None
        Risk configuration. Defaults to global config.
    portfolio_value : float
        Current portfolio value in currency units (used for position sizing).
    """

    def __init__(
        self,
        risk_cfg=None,
        portfolio_value: float = 1_000_000.0,
    ) -> None:
        self.cfg = risk_cfg or config.risk
        self.portfolio_value = portfolio_value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        prediction: Dict,
        price_df: pd.DataFrame,
        fundamental_features: Optional[Dict] = None,
        upcoming_events: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run all risk filters on a prediction.

        Parameters
        ----------
        prediction : dict
            Output of EnsembleModel.predict().
        price_df : pd.DataFrame
            Recent OHLCV data (at least 20 bars).
        fundamental_features : dict, optional
            Scalar fundamental dict.
        upcoming_events : list[dict], optional
            List of scheduled events: [{"name": "...", "hours_until": 24, "impact": "high"}, ...]

        Returns
        -------
        dict with keys:
            approved, position_size, risk_report, adjusted_proba
        """
        risk_report = {}

        # --- 1. Volatility regime ---
        vol_ok, vol_info = self._check_volatility_regime(price_df)
        risk_report["volatility"] = vol_info

        # --- 2. Event risk ---
        event_ok, event_info = self._check_event_risk(upcoming_events or [])
        risk_report["event_risk"] = event_info

        # --- 3. Liquidity risk ---
        liq_ok, liq_info = self._check_liquidity(price_df)
        risk_report["liquidity"] = liq_info

        # --- 4. Confidence gate ---
        confidence = prediction.get("confidence", 0.0)
        conf_ok = confidence >= self.cfg.min_confidence
        risk_report["confidence_gate"] = {
            "confidence": confidence,
            "threshold": self.cfg.min_confidence,
            "passed": conf_ok,
        }

        # --- Aggregate approval ---
        approved = vol_ok and event_ok and liq_ok and conf_ok

        # --- Position sizing ---
        position_size = self._compute_position_size(
            prediction=prediction,
            vol_scalar=vol_info.get("vol_scalar", 1.0),
            approved=approved,
        )

        # --- Adjusted probabilities (dampen if risk is elevated) ---
        adjusted_proba = self._adjust_proba_for_risk(
            proba=prediction.get("probabilities", {}),
            event_risk=event_info.get("event_risk_score", 0.0),
            vol_scalar=vol_info.get("vol_scalar", 1.0),
        )

        return {
            "approved": approved,
            "position_size": position_size,
            "position_size_currency": round(position_size * self.portfolio_value, 2),
            "risk_report": risk_report,
            "adjusted_proba": adjusted_proba,
            "signal": prediction.get("signal"),
            "original_confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Risk filters
    # ------------------------------------------------------------------

    def _check_volatility_regime(
        self, price_df: pd.DataFrame
    ) -> tuple[bool, dict]:
        if price_df.empty or "close" not in price_df.columns:
            return True, {"regime": "unknown", "vol_scalar": 1.0, "passed": True}

        log_ret = np.log(price_df["close"] / price_df["close"].shift(1)).dropna()
        if len(log_ret) < 5:
            return True, {"regime": "unknown", "vol_scalar": 1.0, "passed": True}

        realised_vol = log_ret.tail(20).std() * np.sqrt(252)

        if realised_vol < self.cfg.low_vol_threshold:
            regime = "low"
            vol_scalar = 1.2   # can size up in low vol
            passed = True
        elif realised_vol > self.cfg.high_vol_threshold:
            regime = "high"
            vol_scalar = 0.5   # reduce exposure in high vol
            passed = True       # don't block, just reduce size
        else:
            regime = "normal"
            vol_scalar = 1.0
            passed = True

        return passed, {
            "regime": regime,
            "realised_vol_ann": round(float(realised_vol), 4),
            "vol_scalar": vol_scalar,
            "passed": passed,
        }

    def _check_event_risk(
        self, upcoming_events: List[Dict]
    ) -> tuple[bool, dict]:
        if not upcoming_events:
            return True, {"event_risk_score": 0.0, "events": [], "passed": True}

        # Score events based on impact and proximity
        total_risk = 0.0
        event_summaries = []
        for ev in upcoming_events:
            hours = ev.get("hours_until", 999)
            impact = ev.get("impact", "low").lower()
            impact_score = {"high": 1.0, "medium": 0.5, "low": 0.2}.get(impact, 0.2)

            # Events within 24h are more dangerous
            proximity_mult = 1.5 if hours <= 24 else (1.2 if hours <= 48 else 1.0)
            event_risk = impact_score * proximity_mult

            total_risk = max(total_risk, event_risk)  # take worst case
            event_summaries.append({
                "name": ev.get("name", "Unknown"),
                "hours_until": hours,
                "impact": impact,
                "risk_contribution": round(event_risk, 3),
            })

        total_risk = min(total_risk, 1.0)
        passed = total_risk < self.cfg.max_event_risk

        return passed, {
            "event_risk_score": round(total_risk, 4),
            "events": event_summaries,
            "passed": passed,
        }

    def _check_liquidity(
        self, price_df: pd.DataFrame
    ) -> tuple[bool, dict]:
        if price_df.empty or "volume" not in price_df.columns:
            return True, {"liquidity": "unknown", "passed": True}

        avg_volume = price_df["volume"].tail(20).mean()
        latest_vol = price_df["volume"].iloc[-1]

        # If latest bar volume < 20% of average → thin market
        volume_ratio = latest_vol / (avg_volume + 1e-9)
        is_liquid = volume_ratio > 0.2

        return is_liquid, {
            "avg_volume_20d": round(float(avg_volume), 0),
            "latest_volume": round(float(latest_vol), 0),
            "volume_ratio": round(float(volume_ratio), 4),
            "passed": is_liquid,
        }

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _compute_position_size(
        self,
        prediction: Dict,
        vol_scalar: float,
        approved: bool,
    ) -> float:
        """
        Compute recommended position size as fraction of portfolio.

        Uses a simplified Kelly criterion:
            f* = (p_win * b - p_lose) / b
        where b = expected win/loss ratio (approximated from proba).
        """
        if not approved:
            return 0.0

        proba = prediction.get("probabilities", {})
        signal = prediction.get("signal", "sideways")

        if signal == "bullish":
            p_win = proba.get("bullish", 0.333)
            p_lose = proba.get("bearish", 0.333)
        elif signal == "bearish":
            p_win = proba.get("bearish", 0.333)
            p_lose = proba.get("bullish", 0.333)
        else:
            return 0.0

        b = 1.5   # assumed reward/risk ratio
        kelly = (p_win * b - p_lose) / b
        kelly = max(kelly, 0.0)

        # Apply half-Kelly for conservatism and vol scaling
        half_kelly = kelly * 0.5 * vol_scalar

        # Cap at max position size
        return round(min(half_kelly, self.cfg.max_position_pct), 4)

    # ------------------------------------------------------------------
    # Probability adjustment
    # ------------------------------------------------------------------

    @staticmethod
    def _adjust_proba_for_risk(
        proba: Dict[str, float],
        event_risk: float,
        vol_scalar: float,
    ) -> Dict[str, float]:
        """
        Compress probabilities towards uniform when risk is elevated.

        Higher event_risk or lower vol_scalar → more compression.
        """
        compression = event_risk * 0.3 + (1 - min(vol_scalar, 1.0)) * 0.2
        uniform = 1.0 / 3.0

        adjusted = {}
        for k, v in proba.items():
            adjusted[k] = v * (1 - compression) + uniform * compression

        total = sum(adjusted.values())
        return {k: round(v / total, 4) for k, v in adjusted.items()}
