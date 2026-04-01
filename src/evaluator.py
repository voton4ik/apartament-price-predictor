"""
Метрики качества регрессии: MAE, RMSE, R².
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Считает MAE, RMSE и R².

    Args:
        y_true: истинные значения.
        y_pred: предсказания.

    Returns:
        Словарь с ключами mae, rmse, r2.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    result = {"mae": mae, "rmse": rmse, "r2": r2}
    logger.debug("Метрики: %s", result)
    return result
