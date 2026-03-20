"""
Statistical Models: ARIMA + GARCH
====================================
Provides probability-weighted directional forecasts using:

  - ARIMA   : captures linear autocorrelation in returns
  - GARCH   : models volatility clustering → conditional distribution

Output is a dict: {"bullish": p, "sideways": p, "bearish": p} summing to 1.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    logger.warning("statsmodels / arch not installed; statistical models disabled.")


class StatisticalModel:
    """
    ARIMA + GARCH combined forecaster.

    Parameters
    ----------
    arima_order : tuple
        (p, d, q) for ARIMA. Auto-tuned if None.
    garch_p : int
        GARCH lag order for variance equation.
    garch_q : int
        ARCH lag order.
    horizon : int
        Prediction horizon in days.
    bullish_threshold : float
        Minimum return to classify as "bullish".
    """

    def __init__(
        self,
        arima_order: Tuple[int, int, int] = (1, 1, 1),
        garch_p: int = 1,
        garch_q: int = 1,
        horizon: int = 3,
        bullish_threshold: float = 0.015,
    ) -> None:
        self.arima_order = arima_order
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.horizon = horizon
        self.bullish_threshold = bullish_threshold
        self._arima_result = None
        self._garch_result = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> "StatisticalModel":
        """
        Fit ARIMA on returns and GARCH on ARIMA residuals.

        Parameters
        ----------
        returns : pd.Series
            Log-return series (e.g. np.log(close/close.shift(1)).dropna()).
        """
        if not STATSMODELS_OK:
            logger.warning("Statistical models unavailable; fit skipped.")
            return self

        returns = returns.dropna()
        if len(returns) < 100:
            logger.warning("Too few observations (%d) for statistical model.", len(returns))
            return self

        # --- ARIMA ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                arima = ARIMA(returns, order=self.arima_order)
                self._arima_result = arima.fit()
                logger.info("ARIMA%s fitted. AIC=%.2f", self.arima_order, self._arima_result.aic)
            except Exception as exc:
                logger.warning("ARIMA fit failed: %s", exc)

        # --- GARCH on residuals ---
        residuals = (
            self._arima_result.resid if self._arima_result is not None else returns
        )
        try:
            garch = arch_model(
                residuals * 100,   # scale to avoid tiny values
                vol="Garch",
                p=self.garch_p,
                q=self.garch_q,
                dist="Normal",
            )
            self._garch_result = garch.fit(disp="off")
            logger.info("GARCH(%d,%d) fitted.", self.garch_p, self.garch_q)
        except Exception as exc:
            logger.warning("GARCH fit failed: %s", exc)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self) -> Dict[str, float]:
        """
        Return probability distribution over directional outcomes.

        Uses ARIMA point forecast as the mean of a normal distribution
        and GARCH h-step variance as the standard deviation.
        """
        if not self._fitted or not STATSMODELS_OK:
            return self._uniform_proba()

        # ARIMA h-step mean forecast
        try:
            arima_forecast = self._arima_result.forecast(steps=self.horizon)
            mu = float(arima_forecast.iloc[-1]) if hasattr(arima_forecast, "iloc") else float(arima_forecast[-1])
        except Exception:
            mu = 0.0

        # GARCH h-step variance forecast
        try:
            garch_forecast = self._garch_result.forecast(horizon=self.horizon)
            sigma2 = float(garch_forecast.variance.iloc[-1, -1]) / 10_000  # undo *100 scale
            sigma = max(np.sqrt(sigma2), 1e-6)
        except Exception:
            sigma = 0.02  # 2% fallback

        # Map to directional probabilities via CDF
        t = self.bullish_threshold
        p_bull = 1 - stats.norm.cdf(t, loc=mu, scale=sigma)
        p_bear = stats.norm.cdf(-t, loc=mu, scale=sigma)
        p_side = max(0.0, 1.0 - p_bull - p_bear)

        return {
            "bullish": round(float(p_bull), 4),
            "sideways": round(float(p_side), 4),
            "bearish": round(float(p_bear), 4),
        }

    def forecast_volatility(self) -> float:
        """Return annualised volatility forecast from GARCH."""
        if self._garch_result is None:
            return float("nan")
        try:
            fcast = self._garch_result.forecast(horizon=1)
            daily_var = float(fcast.variance.iloc[-1, 0]) / 10_000
            return round(np.sqrt(daily_var * 252), 4)
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _uniform_proba() -> Dict[str, float]:
        return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}
