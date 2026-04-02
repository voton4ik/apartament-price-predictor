"""
Точка входа: загрузка данных, обучение моделей, метрики, графики, интерактивное предсказание.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from src.data_generator import FEATURE_COLUMNS, TARGET_COLUMN, load_or_generate_csv
from src.evaluator import evaluate_regression, format_price_rub
from src.models import (
    forest_feature_importance,
    predict_forest,
    predict_gradient_boosting,
    predict_linear,
    train_gradient_boosting,
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


def _log_metrics(name: str, m: dict[str, float]) -> None:
    log = logging.getLogger("main")
    log.info(
        "%s — MAE: %s, RMSE: %s, R^2: %.4f, MAPE: %.1f%%",
        name,
        format_price_rub(m["mae"]),
        format_price_rub(m["rmse"]),
        m["r2"],
        m["mape"],
    )


def _pick_best_model(
    metrics_lin: dict[str, float],
    metrics_rf: dict[str, float],
    metrics_gbr: dict[str, float],
) -> tuple[str, float, str]:
    candidates = [
        ("LinearRegression", metrics_lin["r2"], "LinearRegression"),
        ("RandomForestRegressor", metrics_rf["r2"], "RandomForest"),
        ("GradientBoostingRegressor", metrics_gbr["r2"], "GradientBoosting"),
    ]
    best_name, best_r2, short = max(candidates, key=lambda t: t[1])
    return best_name, best_r2, short


def _interactive_predict(
    best_key: str,
    lin_model,
    rf_model,
    gbr_model,
) -> None:
    log = logging.getLogger("main")
    print("\nВведите параметры квартиры для предсказания цены")
    try:
        area_m2 = int(input("Площадь (м², 20–200): ").strip())
        rooms = int(input("Количество комнат (1–6): ").strip())
        floor = int(input("Этаж (1–25): ").strip())
        total_floors = int(input("Этажей в доме (3–30): ").strip())
        distance_center = float(input("Расстояние до центра (км, 0.5–30): ").strip())
        year_built = int(input("Год постройки (1960–2024): ").strip())
        condition = int(input("Состояние (0–3): ").strip())

        row = pd.DataFrame(
            [
                [
                    area_m2,
                    rooms,
                    floor,
                    total_floors,
                    distance_center,
                    year_built,
                    condition,
                ]
            ],
            columns=FEATURE_COLUMNS,
        )

        if best_key == "LinearRegression":
            pred = float(predict_linear(lin_model, row)[0])
        elif best_key == "RandomForestRegressor":
            pred = float(predict_forest(rf_model, row)[0])
        else:
            pred = float(predict_gradient_boosting(gbr_model, row)[0])

        out = f"Предсказанная цена: {format_price_rub(pred)}"
        log.info(out)
        print(out)
    except ValueError as e:
        log.warning("Некорректный ввод: %s", e)
        print("Ошибка ввода: укажите корректные числа в допустимых диапазонах.")


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

    lin_model = train_linear_regression(X_train, y_train)
    y_pred_lin = predict_linear(lin_model, X_test)
    metrics_lin = evaluate_regression(y_test.values, y_pred_lin)
    _log_metrics("LinearRegression", metrics_lin)

    rf_model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    y_pred_rf = predict_forest(rf_model, X_test)
    metrics_rf = evaluate_regression(y_test.values, y_pred_rf)
    _log_metrics("RandomForest", metrics_rf)

    gbr_model = train_gradient_boosting(X_train, y_train, random_state=42)
    y_pred_gbr = predict_gradient_boosting(gbr_model, X_test)
    metrics_gbr = evaluate_regression(y_test.values, y_pred_gbr)
    _log_metrics("GradientBoosting", metrics_gbr)

    best_key, best_r2, best_short = _pick_best_model(
        metrics_lin, metrics_rf, metrics_gbr
    )
    best_estimator = lin_model
    if best_key == "RandomForestRegressor":
        best_estimator = rf_model
    elif best_key == "GradientBoostingRegressor":
        best_estimator = gbr_model

    table = tabulate(
        [
            [
                "LinearRegression",
                format_price_rub(metrics_lin["mae"]),
                format_price_rub(metrics_lin["rmse"]),
                f"{metrics_lin['r2']:.4f}",
                f"{metrics_lin['mape']:.1f}%",
            ],
            [
                "RandomForest",
                format_price_rub(metrics_rf["mae"]),
                format_price_rub(metrics_rf["rmse"]),
                f"{metrics_rf['r2']:.4f}",
                f"{metrics_rf['mape']:.1f}%",
            ],
            [
                "GradientBoosting",
                format_price_rub(metrics_gbr["mae"]),
                format_price_rub(metrics_gbr["rmse"]),
                f"{metrics_gbr['r2']:.4f}",
                f"{metrics_gbr['mape']:.1f}%",
            ],
        ],
        headers=["Модель", "MAE", "RMSE", "R^2", "MAPE"],
        tablefmt="github",
    )
    print(table, flush=True)
    log.info("Таблица сравнения моделей выведена в консоль")

    _interactive_predict(best_key, lin_model, rf_model, gbr_model)

    importance = forest_feature_importance(rf_model, FEATURE_COLUMNS)

    try:
        save_all_plots(
            PLOTS_DIR,
            y_test.values,
            y_pred_lin,
            y_pred_rf,
            y_pred_gbr,
            metrics_lin,
            metrics_rf,
            metrics_gbr,
            importance,
            df,
            FEATURE_COLUMNS,
            TARGET_COLUMN,
            best_estimator,
            best_short,
            X_train,
            y_train,
        )
    except OSError as e:
        log.error("Не удалось сохранить графики: %s", e)
        return 1

    msg = f"Лучшая модель: {best_key} (R^2={best_r2:.4f})"
    log.info(msg)
    print(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
