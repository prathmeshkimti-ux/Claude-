"""
Ensemble Model
================
Combines outputs from all model layers into a final probability distribution.

Architecture:
  Statistical (ARIMA/GARCH) ──┐
  ML (RF + XGB + LGB)         ├──► Weighted Average ──► Calibrated Ensemble
  Deep Learning (LSTM + TF)   │         ↓
  FinBERT Sentiment           ┘    Confidence Score
                                        ↓
                                   Risk Engine Gate
                                        ↓
                                   Final Signal

The ensemble uses configurable weights from config.model.ensemble_weights.
A meta-learner (logistic regression) can be optionally stacked on top.

All probability dicts use keys: "bullish", "sideways", "bearish"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import config

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Weighted ensemble of multi-model probability distributions.

    Parameters
    ----------
    weights : dict | None
        Custom weights for each model layer. Overrides config.
        Keys: "statistical", "ml", "deep_learning", "sentiment", "macro"
    use_meta_learner : bool
        If True, fit a logistic-regression meta-learner (stacking) on top
        of the individual model probabilities.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_meta_learner: bool = False,
    ) -> None:
        self.weights = weights or dict(config.model.ensemble_weights)
        self._normalise_weights()
        self.use_meta_learner = use_meta_learner
        self._meta_learner: Optional[LogisticRegression] = None
        self._meta_scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        statistical_proba: Optional[Dict[str, float]] = None,
        ml_proba: Optional[Dict[str, float]] = None,
        lstm_proba: Optional[Dict[str, float]] = None,
        transformer_proba: Optional[Dict[str, float]] = None,
        sentiment_proba: Optional[Dict[str, float]] = None,
        macro_risk_score: float = 0.5,
    ) -> Dict:
        """
        Combine all model outputs into a final prediction.

        Parameters
        ----------
        statistical_proba : dict
            ARIMA/GARCH probability dict.
        ml_proba : dict
            RF+XGB+LGB ensemble probability dict.
        lstm_proba : dict
            LSTM probability dict.
        transformer_proba : dict
            Transformer probability dict.
        sentiment_proba : dict
            FinBERT / aggregated sentiment probability dict.
        macro_risk_score : float
            0–1 macro risk score from FundamentalFeatures.

        Returns
        -------
        dict with keys:
            probabilities    : {bullish, sideways, bearish}
            signal           : "bullish" | "sideways" | "bearish"
            confidence       : float [0, 1]
            dominant_prob    : float
            model_breakdown  : per-model probabilities
            macro_adjusted   : probabilities after macro adjustment
        """
        model_probas: Dict[str, Dict] = {}

        if statistical_proba:
            model_probas["statistical"] = self._validate(statistical_proba)
        if ml_proba:
            model_probas["ml"] = self._validate(ml_proba)
        if lstm_proba or transformer_proba:
            dl_proba = self._average_probas(
                [p for p in [lstm_proba, transformer_proba] if p is not None]
            )
            model_probas["deep_learning"] = self._validate(dl_proba)
        if sentiment_proba:
            model_probas["sentiment"] = self._validate(sentiment_proba)

        if not model_probas:
            return self._fallback_result()

        # Weighted combination
        if self.use_meta_learner and self._meta_learner is not None:
            ensemble_proba = self._meta_predict(model_probas)
        else:
            ensemble_proba = self._weighted_average(model_probas)

        # Macro risk adjustment
        macro_adjusted = self._apply_macro_adjustment(ensemble_proba, macro_risk_score)

        # Compute confidence
        confidence = self._compute_confidence(macro_adjusted, model_probas)

        # Derive signal
        signal = max(macro_adjusted, key=macro_adjusted.get)

        return {
            "probabilities": macro_adjusted,
            "signal": signal,
            "confidence": round(confidence, 4),
            "dominant_prob": round(max(macro_adjusted.values()), 4),
            "model_breakdown": model_probas,
            "macro_risk_score": round(macro_risk_score, 4),
        }

    def fit_meta_learner(
        self,
        model_proba_history: List[Dict[str, Dict]],
        true_labels: List[str],
    ) -> "EnsembleModel":
        """
        Fit a stacked meta-learner (logistic regression) on historical
        model probability outputs.

        Parameters
        ----------
        model_proba_history : list[dict]
            Each entry: {"statistical": {...}, "ml": {...}, ...}
        true_labels : list[str]
            True directional outcome: "bullish" | "sideways" | "bearish"
        """
        if len(model_proba_history) < 50:
            logger.warning("Too few samples for meta-learner; skipping.")
            return self

        X_meta = self._build_meta_features(model_proba_history)
        y_meta = np.array(true_labels)

        self._meta_scaler = StandardScaler()
        X_scaled = self._meta_scaler.fit_transform(X_meta)

        self._meta_learner = LogisticRegression(
            multi_class="multinomial", max_iter=500, random_state=42
        )
        self._meta_learner.fit(X_scaled, y_meta)
        self.use_meta_learner = True
        logger.info("Meta-learner fitted on %d samples.", len(y_meta))
        return self

    # ------------------------------------------------------------------
    # Probability combination methods
    # ------------------------------------------------------------------

    def _weighted_average(self, model_probas: Dict[str, Dict]) -> Dict[str, float]:
        """Compute weighted average probabilities."""
        total_weight = sum(self.weights.get(k, 0.0) for k in model_probas)
        if total_weight < 1e-9:
            return self._average_probas(list(model_probas.values()))

        bull = side = bear = 0.0
        for key, proba in model_probas.items():
            w = self.weights.get(key, 0.0) / total_weight
            bull += w * proba.get("bullish", 0.333)
            side += w * proba.get("sideways", 0.334)
            bear += w * proba.get("bearish", 0.333)

        return self._normalise_proba({"bullish": bull, "sideways": side, "bearish": bear})

    def _meta_predict(self, model_probas: Dict[str, Dict]) -> Dict[str, float]:
        """Use meta-learner for stacked prediction."""
        X_meta = self._build_meta_features([model_probas])
        X_scaled = self._meta_scaler.transform(X_meta)
        proba = self._meta_learner.predict_proba(X_scaled)[0]
        classes = self._meta_learner.classes_
        result = dict(zip(classes, proba.tolist()))
        return self._normalise_proba({
            "bullish": result.get("bullish", 0.333),
            "sideways": result.get("sideways", 0.334),
            "bearish": result.get("bearish", 0.333),
        })

    @staticmethod
    def _apply_macro_adjustment(
        proba: Dict[str, float], macro_risk: float
    ) -> Dict[str, float]:
        """
        Dampen bullish probability when macro risk is elevated.

        macro_risk = 0 → no adjustment
        macro_risk = 1 → strongly dampen bull, boost bear
        """
        bull = proba["bullish"]
        side = proba["sideways"]
        bear = proba["bearish"]

        # Shift weight from bullish towards bearish proportional to risk
        shift = macro_risk * 0.15  # max 15% shift
        bull_adj = max(0.0, bull - shift)
        bear_adj = min(1.0, bear + shift * 0.5)
        side_adj = max(0.0, side - shift * 0.5)

        return EnsembleModel._normalise_proba({
            "bullish": bull_adj,
            "sideways": side_adj,
            "bearish": bear_adj,
        })

    @staticmethod
    def _compute_confidence(
        final_proba: Dict[str, float],
        model_probas: Dict[str, Dict],
    ) -> float:
        """
        Confidence = dominant probability × model agreement.

        Agreement: fraction of models agreeing on dominant label.
        """
        dominant = max(final_proba, key=final_proba.get)
        dom_prob = final_proba[dominant]

        # Model agreement
        votes = [
            max(p, key=p.get) == dominant for p in model_probas.values()
        ]
        agreement = sum(votes) / max(len(votes), 1)

        return dom_prob * agreement

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _average_probas(proba_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not proba_list:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}
        bull = np.mean([p.get("bullish", 0.333) for p in proba_list])
        side = np.mean([p.get("sideways", 0.334) for p in proba_list])
        bear = np.mean([p.get("bearish", 0.333) for p in proba_list])
        return EnsembleModel._normalise_proba({"bullish": bull, "sideways": side, "bearish": bear})

    @staticmethod
    def _normalise_proba(proba: Dict[str, float]) -> Dict[str, float]:
        total = sum(proba.values())
        if total < 1e-9:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}
        return {k: round(v / total, 4) for k, v in proba.items()}

    @staticmethod
    def _validate(proba: Dict[str, float]) -> Dict[str, float]:
        keys = {"bullish", "sideways", "bearish"}
        for k in keys:
            proba.setdefault(k, 0.333)
        return EnsembleModel._normalise_proba(proba)

    def _normalise_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    @staticmethod
    def _build_meta_features(model_proba_list: List[Dict[str, Dict]]) -> np.ndarray:
        rows = []
        for entry in model_proba_list:
            row = []
            for layer in ["statistical", "ml", "deep_learning", "sentiment", "macro"]:
                p = entry.get(layer, {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333})
                row.extend([p.get("bullish", 0.333), p.get("sideways", 0.334), p.get("bearish", 0.333)])
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _fallback_result() -> Dict:
        return {
            "probabilities": {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333},
            "signal": "sideways",
            "confidence": 0.0,
            "dominant_prob": 0.334,
            "model_breakdown": {},
            "macro_risk_score": 0.5,
        }
