"""
Обучение LinearRegression (со StandardScaler) и RandomForestRegressor.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_linear_pipeline() -> Pipeline:
    """Pipeline: масштабирование + линейная регрессия."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Обучает линейную модель."""
    pipe = build_linear_pipeline()
    pipe.fit(X_train, y_train)
    logger.info("LinearRegression обучена")
    return pipe


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Обучает Random Forest без масштабирования признаков."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info(
        "RandomForestRegressor обучен (n_estimators=%d)",
        n_estimators,
    )
    return model


def predict_linear(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X))


def predict_forest(model: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X))


def forest_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
) -> Dict[str, float]:
    """Возвращает важность признаков по имени."""
    imp = model.feature_importances_
    return {name: float(v) for name, v in zip(feature_names, imp, strict=True)}
