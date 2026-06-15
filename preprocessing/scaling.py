from __future__ import annotations

import numpy as np


class PerSegmentStandardScaler:
    """Standardize each road segment independently."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        train_idx: np.ndarray,
    ) -> "PerSegmentStandardScaler":

        train_values = values[train_idx]
        train_mask = mask[train_idx]

        n_segments = train_values.shape[1]

        mean = np.zeros(n_segments, dtype=np.float32)
        std = np.ones(n_segments, dtype=np.float32)

        for segment in range(n_segments):

            valid_values = train_values[train_mask[:, segment], segment]

            if valid_values.size == 0:
                continue

            mean[segment] = valid_values.mean()

            segment_std = valid_values.std()

            if segment_std > self.eps:
                std[segment] = segment_std

        self.mean_ = mean
        self.std_ = std

        return self

    def transform(self, values: np.ndarray) -> np.ndarray:

        self._check_is_fitted()

        if values.ndim == 2:
            return (
                values - self.mean_[None, :]
            ) / self.std_[None, :]

        if values.ndim == 3:
            return (
                values - self.mean_[None, None, :]
            ) / self.std_[None, None, :]

        raise ValueError(
            f"Unsupported input shape: {values.shape}"
        )

    def inverse_transform(
        self,
        values: np.ndarray,
    ) -> np.ndarray:

        self._check_is_fitted()

        if values.ndim == 2:
            return (
                values * self.std_[None, :]
            ) + self.mean_[None, :]

        if values.ndim == 3:
            return (
                values * self.std_[None, None, :]
            ) + self.mean_[None, None, :]

        raise ValueError(
            f"Unsupported input shape: {values.shape}"
        )

    def _check_is_fitted(self) -> None:

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError(
                "The scaler must be fitted before use."
            )