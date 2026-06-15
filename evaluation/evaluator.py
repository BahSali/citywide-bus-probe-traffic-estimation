from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import mae, rmse, dtw_distance


def evaluate_per_segment(
    y: np.ndarray,
    y_hat: np.ndarray,
    mask: np.ndarray,
    segment_ids: list[str],
) -> pd.DataFrame:

    rows = []

    for idx, segment_id in enumerate(segment_ids):

        valid = mask[:, idx]

        if valid.sum() == 0:
            continue

        rows.append(
            {
                "segment_id": segment_id,
                "MAE": mae(
                    y[:, idx],
                    y_hat[:, idx],
                    valid,
                ),
                "RMSE": rmse(
                    y[:, idx],
                    y_hat[:, idx],
                    valid,
                ),
                "DTW": dtw_distance(
                    y[valid, idx],
                    y_hat[valid, idx],
                ),
            }
        )

    return pd.DataFrame(rows)


def aggregate_metrics(
    results: pd.DataFrame,
) -> dict[str, float]:

    metric_columns = [
        "MAE",
        "RMSE",
        "DTW",
    ]

    return {
        metric: float(results[metric].mean())
        for metric in metric_columns
    }