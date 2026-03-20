"""
Machine Learning Models: Random Forest, XGBoost, LightGBM
============================================================
Trains three gradient-boosted / bagging classifiers and produces
softmax probability distributions over {bullish, sideways, bearish}.

Key design decisions:
  - All three models are trained on the same (X, y) pair.
  - Predictions are calibrated with Platt scaling (sigmoid calibration)
    to ensure well-behaved probabilities.
  - Feature importances are aggregated across models for interpretability.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from config import config

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    logger.warning("xgboost not installed.")

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    logger.warning("lightgbm not installed.")


# Label encoding: -1 → 0 (bearish), 0 → 1 (sideways), 1 → 2 (bullish)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = {0: "bearish", 1: "sideways", 2: "bullish"}
LABEL_NAMES_ORDERED = ["bullish", "sideways", "bearish"]


class MLModel:
    """
    Ensemble of RF + XGBoost + LightGBM classifiers.

    Parameters
    ----------
    n_estimators : int
    learning_rate : float
    max_depth : int
    random_state : int
    """

    def __init__(
        self,
        n_estimators: int = None,
        learning_rate: float = None,
        max_depth: int = None,
        random_state: int = None,
    ) -> None:
        cfg = config.model
        self.n_estimators = n_estimators or cfg.n_estimators
        self.learning_rate = learning_rate or cfg.learning_rate
        self.max_depth = max_depth or cfg.max_depth
        self.random_state = random_state or cfg.random_state

        self._rf: Optional[CalibratedClassifierCV] = None
        self._xgb: Optional[CalibratedClassifierCV] = None
        self._lgb: Optional[CalibratedClassifierCV] = None
        self._feature_names: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLModel":
        """
        Train all three models with cross-validated calibration.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (already scaled / cleaned).
        y : pd.Series
            Labels: -1 (bearish), 0 (sideways), +1 (bullish).
        """
        self._feature_names = list(X.columns)
        y_enc = y.map(LABEL_MAP).astype(int)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # --- Random Forest ---
        rf_base = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self._rf = CalibratedClassifierCV(rf_base, cv=cv, method="sigmoid")
        self._rf.fit(X, y_enc)
        logger.info("Random Forest fitted.")

        # --- XGBoost ---
        if XGB_OK:
            xgb_base = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=self.random_state,
                verbosity=0,
            )
            self._xgb = CalibratedClassifierCV(xgb_base, cv=cv, method="sigmoid")
            self._xgb.fit(X, y_enc)
            logger.info("XGBoost fitted.")

        # --- LightGBM ---
        if LGB_OK:
            lgb_base = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                objective="multiclass",
                num_class=3,
                random_state=self.random_state,
                verbose=-1,
                class_weight="balanced",
            )
            self._lgb = CalibratedClassifierCV(lgb_base, cv=cv, method="sigmoid")
            self._lgb.fit(X, y_enc)
            logger.info("LightGBM fitted.")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Return averaged probability distribution from all available models.

        Returns
        -------
        dict with keys "bullish", "sideways", "bearish"
        """
        if not self._fitted:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}

        proba_list = []
        for model in [self._rf, self._xgb, self._lgb]:
            if model is not None:
                p = model.predict_proba(X)
                proba_list.append(p)

        if not proba_list:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}

        avg_proba = np.mean(proba_list, axis=0).squeeze()

        # Calibrated CV wraps produce shape (1, 3) for single sample
        if avg_proba.ndim == 2:
            avg_proba = avg_proba[0]

        # avg_proba order: [bearish=0, sideways=1, bullish=2]
        return {
            "bullish": round(float(avg_proba[2]), 4),
            "sideways": round(float(avg_proba[1]), 4),
            "bearish": round(float(avg_proba[0]), 4),
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Return accuracy and macro-F1 on a held-out set."""
        y_enc = y.map(LABEL_MAP).astype(int)
        preds = []
        for model in [self._rf, self._xgb, self._lgb]:
            if model is not None:
                preds.append(model.predict(X))

        if not preds:
            return {}

        y_pred = np.round(np.mean(preds, axis=0)).astype(int)
        return {
            "accuracy": round(accuracy_score(y_enc, y_pred), 4),
            "f1_macro": round(f1_score(y_enc, y_pred, average="macro"), 4),
        }

    def feature_importances(self) -> pd.DataFrame:
        """Return averaged feature importances across RF and tree models."""
        records = []

        if self._rf is not None:
            # Calibrated CV doesn't expose feature_importances_ directly
            try:
                base_rf = self._rf.calibrated_classifiers_[0].estimator
                fi = pd.Series(base_rf.feature_importances_, index=self._feature_names)
                records.append(fi)
            except Exception:
                pass

        if XGB_OK and self._xgb is not None:
            try:
                base_xgb = self._xgb.calibrated_classifiers_[0].estimator
                fi = pd.Series(base_xgb.feature_importances_, index=self._feature_names)
                records.append(fi)
            except Exception:
                pass

        if LGB_OK and self._lgb is not None:
            try:
                base_lgb = self._lgb.calibrated_classifiers_[0].estimator
                fi = pd.Series(base_lgb.feature_importances_, index=self._feature_names)
                records.append(fi)
            except Exception:
                pass

        if not records:
            return pd.DataFrame()

        avg_fi = pd.concat(records, axis=1).mean(axis=1).sort_values(ascending=False)
        return avg_fi.reset_index().rename(columns={"index": "feature", 0: "importance"})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("MLModel saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MLModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("MLModel loaded from %s", path)
        return obj
