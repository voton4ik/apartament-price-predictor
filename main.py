"""
Точка входа: загрузка данных, обучение моделей, метрики, графики.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from tabulate import tabulate

from src.data_generator import FEATURE_COLUMNS, TARGET_COLUMN, load_or_generate_csv
from src.evaluator import evaluate_regression
from src.models import (
    forest_feature_importance,
    predict_forest,
    predict_linear,
    train_linear_regression,
    train_random_forest,
)
from src.preprocessor import split_features_target
from src.visualizer import save_all_plots

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "apartments.csv"
PLOTS_DIR = BASE_DIR / "plots"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
    )


def _configure_stdout_utf8() -> None:
    """На Windows cp1251 ломает Unicode в print; UTF-8 предпочтительнее."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def main() -> int:
    _configure_stdout_utf8()
    _configure_logging()
    log = logging.getLogger("main")

    try:
        df = load_or_generate_csv(DATA_PATH, n_rows=1_500, random_state=42)
    except FileNotFoundError:
        log.exception("Файл данных не найден и не может быть создан")
        return 1
    except ValueError as e:
        log.error("Ошибка данных: %s", e)
        return 1

    try:
        X_train, X_test, y_train, y_test = split_features_target(
            df, test_size=0.2, random_state=42
        )
    except ValueError as e:
        log.error("Предобработка: %s", e)
        return 1

    # 3. LinearRegression
    lin_model = train_linear_regression(X_train, y_train)
    y_pred_lin = predict_linear(lin_model, X_test)
    metrics_lin = evaluate_regression(y_test.values, y_pred_lin)
    log.info(
        "LinearRegression — MAE: %.2f, RMSE: %.2f, R^2: %.4f",
        metrics_lin["mae"],
        metrics_lin["rmse"],
        metrics_lin["r2"],
    )

    # 4. RandomForest
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    y_pred_rf = predict_forest(rf_model, X_test)
    metrics_rf = evaluate_regression(y_test.values, y_pred_rf)
    log.info(
        "RandomForest — MAE: %.2f, RMSE: %.2f, R^2: %.4f",
        metrics_rf["mae"],
        metrics_rf["rmse"],
        metrics_rf["r2"],
    )

    # 5. Таблица сравнения
    table = tabulate(
        [
            [
                "LinearRegression",
                f"{metrics_lin['mae']:.2f}",
                f"{metrics_lin['rmse']:.2f}",
                f"{metrics_lin['r2']:.4f}",
            ],
            [
                "RandomForest",
                f"{metrics_rf['mae']:.2f}",
                f"{metrics_rf['rmse']:.2f}",
                f"{metrics_rf['r2']:.4f}",
            ],
        ],
        headers=["Модель", "MAE", "RMSE", "R^2"],
        tablefmt="github",
    )
    log.info("Сравнение моделей (тест):\n%s", table)
    print(table)

    importance = forest_feature_importance(rf_model, FEATURE_COLUMNS)

    # 6. Графики
    try:
        save_all_plots(
            PLOTS_DIR,
            y_test.values,
            y_pred_lin,
            y_pred_rf,
            metrics_lin,
            metrics_rf,
            importance,
            df,
            FEATURE_COLUMNS,
            TARGET_COLUMN,
        )
    except OSError as e:
        log.error("Не удалось сохранить графики: %s", e)
        return 1

    # 7. Лучшая модель по R²
    if metrics_lin["r2"] >= metrics_rf["r2"]:
        best_name, best_r2 = "LinearRegression", metrics_lin["r2"]
    else:
        best_name, best_r2 = "RandomForestRegressor", metrics_rf["r2"]

    msg = f"Лучшая модель: {best_name} (R^2={best_r2:.4f})"
    log.info(msg)
    print(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
