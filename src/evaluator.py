"""
Метрики качества регрессии: MAE, RMSE, R², MAPE.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def _mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(y_t) > 1e-9
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Считает MAE, RMSE, R² и MAPE (%).

    Args:
        y_true: истинные значения.
        y_pred: предсказания.

    Returns:
        Словарь с ключами mae, rmse, r2, mape.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = _mape_percent(y_true, y_pred)
    result = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
    logger.debug("Метрики: %s", result)
    return result


def format_price_rub(value: float) -> str:
    """Форматирует число как цену в рублях с разделителем тысяч."""
    return f"{value:,.0f} ₽"
