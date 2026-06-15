from __future__ import annotations

import numpy as np


class MissingValueHandler:
    """Keep missing values explicit and return the corresponding validity mask."""

    def __call__(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = values.astype(np.float32, copy=False)
        mask = np.isfinite(values)

        return values, mask