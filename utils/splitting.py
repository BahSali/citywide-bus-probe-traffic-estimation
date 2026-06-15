from __future__ import annotations

import numpy as np
import pandas as pd


def make_split_indices(
    mask: np.ndarray,
    timestamps: pd.DatetimeIndex,
    strategy: str = "by_days",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    val_days: int = 2,
    test_days: int = 3,
    min_observed_targets: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strategy == "by_days":
        train_idx, val_idx, test_idx = _split_by_days(
            timestamps=timestamps,
            val_days=val_days,
            test_days=test_days,
        )
    elif strategy == "ratio":
        train_idx, val_idx, test_idx = _split_by_ratio(
            n_steps=len(mask),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return (
        _filter_observed(mask, train_idx, min_observed_targets),
        _filter_observed(mask, val_idx, min_observed_targets),
        _filter_observed(mask, test_idx, min_observed_targets),
    )


def _split_by_days(
    timestamps: pd.DatetimeIndex,
    val_days: int,
    test_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = pd.DatetimeIndex(timestamps)
    days = timestamps.normalize()
    unique_days = days.unique().sort_values()

    if test_days < 1:
        raise ValueError("test_days must be at least 1.")

    if val_days < 0:
        raise ValueError("val_days cannot be negative.")

    if val_days + test_days >= len(unique_days):
        raise ValueError("Not enough days for the requested validation/test split.")

    test_block = unique_days[-test_days:]
    val_block = unique_days[-(test_days + val_days):-test_days]
    train_block = unique_days[:-(test_days + val_days)]

    indices = np.arange(len(timestamps))

    train_idx = indices[np.asarray(days.isin(train_block))]
    val_idx = indices[np.asarray(days.isin(val_block))]
    test_idx = indices[np.asarray(days.isin(test_block))]

    return train_idx, val_idx, test_idx


def _split_by_ratio(
    n_steps: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    train_end = int(n_steps * train_ratio)
    val_end = int(n_steps * (train_ratio + val_ratio))

    indices = np.arange(n_steps)

    return (
        indices[:train_end],
        indices[train_end:val_end],
        indices[val_end:],
    )


def _filter_observed(
    mask: np.ndarray,
    indices: np.ndarray,
    min_observed_targets: int,
) -> np.ndarray:
    observed_count = mask.sum(axis=1)
    keep = observed_count[indices] >= min_observed_targets

    return indices[keep]