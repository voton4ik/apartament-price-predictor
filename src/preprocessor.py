"""
Предобработка данных: проверка признаков и разбиение train/test.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_generator import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def validate_features(df: pd.DataFrame) -> None:
    """Проверяет наличие и базовые ограничения по признакам."""
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {sorted(missing)}")

    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError("В данных есть пропуски; заполните или удалите строки.")


def split_features_target(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет X, y и выполняет train_test_split 80/20.

    Returns:
        X_train, X_test, y_train, y_test
    """
    validate_features(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        "Train: %d, Test: %d (test_size=%.0f%%)",
        len(X_train),
        len(X_test),
        test_size * 100,
    )
    return X_train, X_test, y_train, y_test
