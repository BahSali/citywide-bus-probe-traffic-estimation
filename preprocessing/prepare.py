from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from preprocessing.alignment import align_datasets
from preprocessing.missing import MissingValueHandler
from preprocessing.scaling import PerSegmentStandardScaler
from utils.splitting import make_split_indices
from utils.windowing import create_temporal_windows


@dataclass(frozen=True)
class Dataset:
    source: pd.DataFrame
    reference: pd.DataFrame
    target_mask: pd.DataFrame
    segment_ids: list[str]

    X_all: np.ndarray
    y_all: np.ndarray
    mask_all: np.ndarray

    train: tuple[np.ndarray, np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray, np.ndarray]

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    target_scaler: PerSegmentStandardScaler | None

    @property
    def input_window(self) -> int:
        return self.X_all.shape[1]


def prepare_dataset(
    source: pd.DataFrame,
    reference: pd.DataFrame,
    cfg,
) -> Dataset:
    source, reference, segment_ids = align_datasets(source, reference)

    X, _ = MissingValueHandler()(source.to_numpy())
    y, mask = MissingValueHandler()(reference.to_numpy())

    train_idx, val_idx, test_idx = make_split_indices(
        mask=mask,
        timestamps=source.index,
        strategy=cfg.split.strategy,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
        val_days=cfg.split.val_days,
        test_days=cfg.split.test_days,
        min_observed_targets=cfg.split.min_observed_targets,
    )

    X, y, target_scaler = _scale_data(
        X=X,
        y=y,
        mask=mask,
        train_idx=train_idx,
        cfg=cfg,
    )

    X = np.nan_to_num(X, nan=cfg.features.pad_value)

    X_all = create_temporal_windows(
        values=X,
        timestamps=source.index,
        recent_steps=cfg.features.recent_steps,
        daily_lags=cfg.features.daily_lags,
        weekly_lags=cfg.features.weekly_lags,
        similar_days=cfg.features.similar_days,
        similarity_metric=cfg.features.similarity_metric,
        pad_value=cfg.features.pad_value,
    )

    target_mask = pd.DataFrame(
        mask,
        index=reference.index,
        columns=segment_ids,
    )

    return Dataset(
        source=source,
        reference=reference,
        target_mask=target_mask,
        segment_ids=segment_ids,
        X_all=X_all,
        y_all=y,
        mask_all=mask,
        train=(X_all[train_idx], y[train_idx], mask[train_idx]),
        val=(X_all[val_idx], y[val_idx], mask[val_idx]),
        test=(X_all[test_idx], y[test_idx], mask[test_idx]),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        target_scaler=target_scaler,
    )


def _scale_data(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    train_idx: np.ndarray,
    cfg,
) -> tuple[np.ndarray, np.ndarray, PerSegmentStandardScaler | None]:
    target_scaler = None

    if cfg.normalization.input:
        input_mask = np.isfinite(X)
        input_scaler = PerSegmentStandardScaler(eps=cfg.normalization.eps)
        X = input_scaler.fit(X, input_mask, train_idx).transform(X)

    if cfg.normalization.target:
        target_scaler = PerSegmentStandardScaler(eps=cfg.normalization.eps)
        y = target_scaler.fit(y, mask, train_idx).transform(y)

    return X, y, target_scaler