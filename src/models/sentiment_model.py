"""
FinBERT Sentiment Model
========================
Uses the pre-trained FinBERT model (ProsusAI/finbert) to extract
fine-grained financial sentiment from news headlines and article snippets.

FinBERT produces three-class probabilities:
  positive / neutral / negative

These are then mapped to a [-1, +1] sentiment score and directional proba.

The model is lazy-loaded on first use to avoid startup cost.
Falls back to VADER if transformers / GPU unavailable.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False
    logger.warning("transformers not installed; FinBERT disabled.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False


FINBERT_MODEL_ID = "ProsusAI/finbert"
MAX_TOKENS = 512
BATCH_SIZE = 16


class FinBERTSentiment:
    """
    FinBERT-based financial sentiment classifier.

    Lazy-loads the model on first call to avoid startup cost.
    Automatically falls back to VADER if transformers unavailable.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier. Defaults to ProsusAI/finbert.
    device : int
        GPU device index. -1 = CPU.
    use_cache : bool
        Cache sentence-level results to avoid recomputing.
    """

    def __init__(
        self,
        model_id: str = FINBERT_MODEL_ID,
        device: int = -1,
        use_cache: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.use_cache = use_cache
        self._pipeline = None
        self._vader = SentimentIntensityAnalyzer() if VADER_OK else None
        self._cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single text string.

        Returns
        -------
        dict with keys:
            positive, neutral, negative (0–1 probabilities, sum = 1)
            compound  (-1 → +1 weighted score)
            label     ("positive" | "neutral" | "negative")
        """
        if not text or not text.strip():
            return self._neutral_result()

        if self.use_cache and text in self._cache:
            return self._cache[text]

        if TRANSFORMERS_OK:
            result = self._finbert_score(text)
        elif VADER_OK:
            result = self._vader_score(text)
        else:
            result = self._neutral_result()

        if self.use_cache:
            self._cache[text] = result

        return result

    def score_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """Score a batch of texts efficiently."""
        return [self.score_text(t) for t in texts]

    def aggregate(self, texts: List[str]) -> Dict[str, float]:
        """
        Score a list of texts and return aggregate statistics.

        Returns
        -------
        dict with keys:
            avg_compound, avg_positive, avg_negative, avg_neutral,
            bullish_ratio, bearish_ratio, dominant_label
        """
        if not texts:
            return {
                "avg_compound": 0.0,
                "avg_positive": 0.33,
                "avg_negative": 0.33,
                "avg_neutral": 0.34,
                "bullish_ratio": 0.0,
                "bearish_ratio": 0.0,
                "dominant_label": "neutral",
            }

        results = self.score_texts(texts)
        compounds = [r["compound"] for r in results]
        positives = [r["positive"] for r in results]
        negatives = [r["negative"] for r in results]
        neutrals = [r["neutral"] for r in results]
        labels = [r["label"] for r in results]

        return {
            "avg_compound": round(float(np.mean(compounds)), 4),
            "avg_positive": round(float(np.mean(positives)), 4),
            "avg_negative": round(float(np.mean(negatives)), 4),
            "avg_neutral": round(float(np.mean(neutrals)), 4),
            "bullish_ratio": round(labels.count("positive") / len(labels), 4),
            "bearish_ratio": round(labels.count("negative") / len(labels), 4),
            "dominant_label": max(set(labels), key=labels.count),
        }

    def get_directional_proba(self, texts: List[str]) -> Dict[str, float]:
        """
        Map aggregated FinBERT output to directional price movement proba.

        Returns
        -------
        dict with keys "bullish", "sideways", "bearish"
        """
        agg = self.aggregate(texts)
        p_bull = agg["avg_positive"]
        p_bear = agg["avg_negative"]
        p_side = agg["avg_neutral"]

        # Renormalise
        total = p_bull + p_bear + p_side
        if total < 1e-6:
            return {"bullish": 0.333, "sideways": 0.334, "bearish": 0.333}

        return {
            "bullish": round(p_bull / total, 4),
            "sideways": round(p_side / total, 4),
            "bearish": round(p_bear / total, 4),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            logger.info("Loading FinBERT model '%s'...", self.model_id)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_id,
                tokenizer=self.model_id,
                device=self.device,
                top_k=None,
            )
            logger.info("FinBERT loaded.")
        except Exception as exc:
            logger.warning("FinBERT load failed: %s. Falling back to VADER.", exc)
            self._pipeline = None

    def _finbert_score(self, text: str) -> Dict[str, float]:
        self._load_pipeline()

        if self._pipeline is None:
            return self._vader_score(text) if VADER_OK else self._neutral_result()

        # Truncate to avoid token limit
        text_trunc = text[:1000]
        try:
            output = self._pipeline(text_trunc)[0]
        except Exception as exc:
            logger.debug("FinBERT inference failed: %s", exc)
            return self._neutral_result()

        label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
        proba = {label_map.get(item["label"].lower(), "neutral"): item["score"] for item in output}

        p_pos = proba.get("positive", 0.0)
        p_neg = proba.get("negative", 0.0)
        p_neu = proba.get("neutral", 0.0)
        compound = p_pos - p_neg

        return {
            "positive": round(p_pos, 4),
            "neutral": round(p_neu, 4),
            "negative": round(p_neg, 4),
            "compound": round(compound, 4),
            "label": max(proba, key=proba.get),
        }

    def _vader_score(self, text: str) -> Dict[str, float]:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        p_pos = scores["pos"]
        p_neg = scores["neg"]
        p_neu = scores["neu"]
        label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
        return {
            "positive": round(p_pos, 4),
            "neutral": round(p_neu, 4),
            "negative": round(p_neg, 4),
            "compound": round(compound, 4),
            "label": label,
        }

    @staticmethod
    def _neutral_result() -> Dict[str, float]:
        return {
            "positive": 0.33,
            "neutral": 0.34,
            "negative": 0.33,
            "compound": 0.0,
            "label": "neutral",
        }
