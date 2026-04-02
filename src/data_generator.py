"""
Генерация и загрузка синтетических данных о квартирах.

Формирует CSV с признаками и целевой переменной price_rub по формуле с шумом.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "area_m2",
    "rooms",
    "floor",
    "total_floors",
    "distance_center",
    "year_built",
    "condition",
]
TARGET_COLUMN = "price_rub"

RNG_SEED = 42


def generate_apartments(
    n_rows: int = 1_500,
    random_state: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Создаёт синтетический датасет квартир.

    Args:
        n_rows: число строк (рекомендуется 1000–2000).
        random_state: seed для воспроизводимости.

    Returns:
        DataFrame с признаками и price_rub.
    """
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1")

    rng = np.random.default_rng(random_state)

    area_m2 = rng.integers(20, 201, size=n_rows)
    rooms = rng.integers(1, 7, size=n_rows)
    total_floors = rng.integers(3, 31, size=n_rows)
    floor = np.array(
        [rng.integers(1, int(tf) + 1) for tf in total_floors],
        dtype=np.int64,
    )
    distance_center = rng.uniform(0.5, 30.0, size=n_rows).round(2)
    year_built = rng.integers(1960, 2025, size=n_rows)
    condition = rng.integers(0, 4, size=n_rows)

    df = pd.DataFrame(
        {
            "area_m2": area_m2,
            "rooms": rooms,
            "floor": floor,
            "total_floors": total_floors,
            "distance_center": distance_center,
            "year_built": year_built,
            "condition": condition,
        }
    )

    am = df["area_m2"].to_numpy(dtype=np.float64)
    rm = df["rooms"].to_numpy(dtype=np.float64)
    fl = df["floor"].to_numpy(dtype=np.float64)
    tf = df["total_floors"].to_numpy(dtype=np.float64)
    dc = df["distance_center"].to_numpy(dtype=np.float64)
    yb = df["year_built"].to_numpy(dtype=np.float64)
    cd = df["condition"].to_numpy(dtype=np.float64)

    base = (
        am * 120_000.0
        + rm * 150_000.0
        + cd * 200_000.0
        - dc * 15_000.0
        + (yb - 1960.0) * 3_000.0
        - np.maximum(tf - fl, 0.0) * 5_000.0
    )
    noise = rng.normal(loc=0.0, scale=base * 0.12, size=n_rows)
    price = np.clip(base + noise, 1_500_000.0, 35_000_000.0)
    df[TARGET_COLUMN] = np.round(price, 0)

    logger.info("Сгенерировано %d строк синтетических данных", n_rows)
    return df


def load_or_generate_csv(
    csv_path: Path,
    n_rows: int = 1_500,
    random_state: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Загружает CSV или создаёт его при отсутствии файла.

    Args:
        csv_path: путь к apartments.csv.
        n_rows: число строк при генерации.
        random_state: seed.

    Returns:
        DataFrame с данными.

    Raises:
        ValueError: если в файле нет нужных колонок.
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Ошибка чтения CSV: %s", csv_path)
            raise

        missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
        if missing:
            raise ValueError(f"В CSV отсутствуют колонки: {sorted(missing)}")

        logger.info("Загружено %d строк из %s", len(df), csv_path)
        return df

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_apartments(n_rows=n_rows, random_state=random_state)
    df.to_csv(csv_path, index=False)
    logger.info("Сохранён новый файл данных: %s", csv_path)
    return df
