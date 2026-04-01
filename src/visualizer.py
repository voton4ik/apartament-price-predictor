"""
Визуализация результатов: scatter, метрики, важность признаков, остатки, heatmap, распределение.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

PLOT_DPI = 150


def _ensure_plots_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_1_scatter_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred_linear: np.ndarray,
    y_pred_forest: np.ndarray,
    out_path: Path,
) -> None:
    """Scatter: реальные vs предсказанные цены для обеих моделей."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    lim_min = min(y_true.min(), y_pred_linear.min(), y_pred_forest.min())
    lim_max = max(y_true.max(), y_pred_linear.max(), y_pred_forest.max())

    for ax, pred, title in zip(
        axes,
        (y_pred_linear, y_pred_forest),
        ("LinearRegression", "RandomForest"),
        strict=True,
    ):
        ax.scatter(y_true, pred, alpha=0.5, s=12, edgecolors="none")
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="идеал")
        ax.set_xlabel("Реальная цена, EUR")
        ax.set_ylabel("Предсказанная цена, EUR")
        ax.set_title(title)
        ax.legend()
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Реальные vs предсказанные цены")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def plot_2_metrics_bar(
    metrics_linear: Mapping[str, float],
    metrics_forest: Mapping[str, float],
    out_path: Path,
) -> None:
    """Столбчатая диаграмма MAE, RMSE, R²."""
    labels = ["MAE", "RMSE", "R²"]
    keys = ["mae", "rmse", "r2"]
    x = np.arange(len(labels))
    width = 0.35

    lin_vals = [metrics_linear[k] for k in keys]
    for_vals = [metrics_forest[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, lin_vals, width, label="LinearRegression")
    ax.bar(x + width / 2, for_vals, width, label="RandomForest")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Значение")
    ax.set_title("Сравнение метрик на тестовой выборке")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def plot_3_feature_importance(
    importance: Dict[str, float],
    out_path: Path,
) -> None:
    """Горизонтальный bar chart важности признаков (RandomForest)."""
    names = list(importance.keys())
    values = list(importance.values())
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color="steelblue")
    ax.set_xlabel("Важность признака")
    ax.set_title("RandomForest: feature_importances_")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def plot_4_residuals(
    y_true: np.ndarray,
    y_pred_linear: np.ndarray,
    y_pred_forest: np.ndarray,
    out_path: Path,
) -> None:
    """Остатки (y - y_pred) для обеих моделей."""
    res_lin = y_true - y_pred_linear
    res_for = y_true - y_pred_forest

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    configs = (
        (axes[0], y_pred_linear, res_lin, "LinearRegression"),
        (axes[1], y_pred_forest, res_for, "RandomForest"),
    )
    for ax, pred, res, title in configs:
        ax.scatter(pred, res, alpha=0.5, s=12, edgecolors="none")
        ax.axhline(0, color="r", linestyle="--", lw=1)
        ax.set_xlabel("Предсказанная цена")
        ax.set_ylabel("Остаток (реальное − предсказание)")
        ax.set_title(title)

    fig.suptitle("График остатков")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def plot_5_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    out_path: Path,
) -> None:
    """Тепловая карта корреляций признаков и целевой переменной."""
    cols = feature_cols + [target_col]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Корреляция признаков и цены")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def plot_6_price_distribution(
    prices: pd.Series,
    out_path: Path,
) -> None:
    """Гистограмма распределения цен."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(prices, kde=True, ax=ax, color="teal", edgecolor="white")
    ax.set_xlabel("Цена, EUR")
    ax.set_ylabel("Частота")
    ax.set_title("Распределение цен в датасете")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def save_all_plots(
    plots_dir: Path,
    y_test: np.ndarray,
    y_pred_linear: np.ndarray,
    y_pred_forest: np.ndarray,
    metrics_linear: Mapping[str, float],
    metrics_forest: Mapping[str, float],
    importance: Dict[str, float],
    df_full: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    """Сохраняет все шесть графиков в PNG (dpi=150)."""
    d = _ensure_plots_dir(plots_dir)
    plot_1_scatter_actual_vs_predicted(
        y_test, y_pred_linear, y_pred_forest, d / "plot_1_actual_vs_predicted.png"
    )
    plot_2_metrics_bar(
        metrics_linear, metrics_forest, d / "plot_2_metrics_comparison.png"
    )
    plot_3_feature_importance(importance, d / "plot_3_feature_importance.png")
    plot_4_residuals(
        y_test, y_pred_linear, y_pred_forest, d / "plot_4_residuals.png"
    )
    plot_5_correlation_heatmap(
        df_full, feature_cols, target_col, d / "plot_5_correlation_heatmap.png"
    )
    plot_6_price_distribution(df_full[target_col], d / "plot_6_price_distribution.png")
