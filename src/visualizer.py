"""
Визуализация результатов: scatter, метрики, важность признаков, остатки, heatmap, распределение, learning curve.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

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
    y_pred_gbr: np.ndarray,
    out_path: Path,
) -> None:
    """Scatter: реальные vs предсказанные цены для трёх моделей."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    preds = (y_pred_linear, y_pred_forest, y_pred_gbr)
    titles = ("LinearRegression", "RandomForest", "GradientBoosting")
    lim_min = min(y_true.min(), *(p.min() for p in preds))
    lim_max = max(y_true.max(), *(p.max() for p in preds))

    for ax, pred, title in zip(axes, preds, titles, strict=True):
        ax.scatter(y_true, pred, alpha=0.5, s=10, edgecolors="none")
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="идеал")
        ax.set_xlabel("Реальная цена, ₽")
        ax.set_ylabel("Предсказанная цена, ₽")
        ax.set_title(title)
        ax.legend()
        ax.ticklabel_format(style="sci", axis="both", scilimits=(6, 6))

    fig.suptitle("Реальные vs предсказанные цены")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def plot_2_metrics_bar(
    metrics_linear: Mapping[str, float],
    metrics_forest: Mapping[str, float],
    metrics_gbr: Mapping[str, float],
    out_path: Path,
) -> None:
    """Столбчатые диаграммы MAE, RMSE (₽), R² и MAPE (%)."""
    models = ("LinearRegression", "RandomForest", "GradientBoosting")
    all_m = (metrics_linear, metrics_forest, metrics_gbr)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    w = 0.25
    x0 = np.arange(1)

    for ax, key, title, ylabel in zip(
        axes,
        ("mae", "rmse", "r2", "mape"),
        ("MAE", "RMSE", "R²", "MAPE"),
        ("₽", "₽", "R²", "%"),
        strict=True,
    ):
        for i, (name, m) in enumerate(zip(models, all_m, strict=True)):
            off = (i - 1) * w
            ax.bar(x0 + off, [m[key]], w, label=name)
        ax.set_xticks(x0)
        ax.set_xticklabels([title])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} на тесте")
        if key in ("mae", "rmse"):
            ax.ticklabel_format(style="plain", axis="y")
    axes[0].legend(loc="upper left", fontsize=8)

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
    y_pred_gbr: np.ndarray,
    out_path: Path,
) -> None:
    """Остатки (y - y_pred) для трёх моделей."""
    res_lin = y_true - y_pred_linear
    res_for = y_true - y_pred_forest
    res_gbr = y_true - y_pred_gbr

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    configs = (
        (axes[0], y_pred_linear, res_lin, "LinearRegression"),
        (axes[1], y_pred_forest, res_for, "RandomForest"),
        (axes[2], y_pred_gbr, res_gbr, "GradientBoosting"),
    )
    for ax, pred, res, title in configs:
        ax.scatter(pred, res, alpha=0.5, s=10, edgecolors="none")
        ax.axhline(0, color="r", linestyle="--", lw=1)
        ax.set_xlabel("Предсказанная цена, ₽")
        ax.set_ylabel("Остаток (реальное − предсказание), ₽")
        ax.set_title(title)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))

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
    ax.set_title("Корреляция признаков и цены (₽)")
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
    ax.set_xlabel("Цена, ₽")
    ax.set_ylabel("Частота")
    ax.set_title("Распределение цен в датасете")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def plot_7_learning_curve(
    best_estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_title: str,
    out_path: Path,
    random_state: int = 42,
) -> None:
    """
    Кривая обучения: R² на обучении и при кросс-валидации по долям выборки.

    Доли обучающей выборки: 10 % … 100 % с шагом 10 %. Для оценки на валидации
    используется cross_val_score.
    """
    rng = np.random.RandomState(random_state)
    n = len(X_train)
    order = rng.permutation(n)
    X_ord = X_train.iloc[order].reset_index(drop=True)
    y_ord = y_train.iloc[order].reset_index(drop=True)

    train_sizes: list[int] = []
    train_r2: list[float] = []
    val_r2: list[float] = []

    for k in range(1, 11):
        frac = k / 10.0
        n_i = max(5, int(np.ceil(frac * n)))
        if n_i > n:
            n_i = n
        Xi = X_ord.iloc[:n_i]
        yi = y_ord.iloc[:n_i]
        train_sizes.append(n_i)

        est_fit = clone(best_estimator)
        est_fit.fit(Xi, yi)
        train_r2.append(float(r2_score(yi, est_fit.predict(Xi))))

        n_splits = min(5, n_i)
        if n_splits < 2:
            n_splits = 2
        val_r2.append(
            float(
                cross_val_score(
                    clone(best_estimator),
                    Xi,
                    yi,
                    cv=n_splits,
                    scoring="r2",
                ).mean()
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_r2, "o-", label="Обучение", color="C0")
    ax.plot(train_sizes, val_r2, "s-", label="Валидация", color="C1")
    ax.set_xlabel("Размер обучающей выборки")
    ax.set_ylabel("R²")
    ax.set_title(f"Кривая обучения: {model_title}")
    ax.legend()
    ax.set_xticks(train_sizes)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранён график: %s", out_path)


def save_all_plots(
    plots_dir: Path,
    y_test: np.ndarray,
    y_pred_linear: np.ndarray,
    y_pred_forest: np.ndarray,
    y_pred_gbr: np.ndarray,
    metrics_linear: Mapping[str, float],
    metrics_forest: Mapping[str, float],
    metrics_gbr: Mapping[str, float],
    importance: Dict[str, float],
    df_full: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    best_estimator: Any,
    best_model_title: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Сохраняет все семь графиков в PNG (dpi=150)."""
    d = _ensure_plots_dir(plots_dir)
    plot_1_scatter_actual_vs_predicted(
        y_test,
        y_pred_linear,
        y_pred_forest,
        y_pred_gbr,
        d / "plot_1_actual_vs_predicted.png",
    )
    plot_2_metrics_bar(
        metrics_linear,
        metrics_forest,
        metrics_gbr,
        d / "plot_2_metrics_comparison.png",
    )
    plot_3_feature_importance(importance, d / "plot_3_feature_importance.png")
    plot_4_residuals(
        y_test,
        y_pred_linear,
        y_pred_forest,
        y_pred_gbr,
        d / "plot_4_residuals.png",
    )
    plot_5_correlation_heatmap(
        df_full, feature_cols, target_col, d / "plot_5_correlation_heatmap.png"
    )
    plot_6_price_distribution(
        df_full[target_col], d / "plot_6_price_distribution.png"
    )
    plot_7_learning_curve(
        best_estimator,
        X_train,
        y_train,
        best_model_title,
        d / "plot_7_learning_curve.png",
    )
