"""Evaluation contract; splitting must avoid subject/video leakage."""
from typing import Mapping, Protocol, Sequence


class EmotionEvaluator(Protocol):
    """Future implementation: per-class metrics, macro F1, confusion matrix."""
    def evaluate(self, y_true: Sequence[str], y_pred: Sequence[str],
                 classes: Sequence[str]) -> Mapping[str, object]: ...
