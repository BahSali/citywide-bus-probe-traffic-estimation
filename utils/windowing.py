from __future__ import annotations

import numpy as np
import pandas as pd


def create_temporal_windows(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    recent_steps: int,
    daily_lags: int = 2,
    weekly_lags: int = 1,
    similar_days: int = 1,
    similarity_metric: str = "cosine",
    pad_value: float = 0.0,
) -> np.ndarray:
    timestamps = pd.DatetimeIndex(timestamps)
    time_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    day_to_indices = _group_indices_by_day(timestamps)

    windows = [
        _build_window(
            values=values,
            timestamps=timestamps,
            time_to_idx=time_to_idx,
            day_to_indices=day_to_indices,
            target_idx=i,
            recent_steps=recent_steps,
            daily_lags=daily_lags,
            weekly_lags=weekly_lags,
            similar_days=similar_days,
            similarity_metric=similarity_metric,
            pad_value=pad_value,
        )
        for i in range(len(timestamps))
    ]

    return np.stack(windows).astype(np.float32)


def _build_window(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    time_to_idx: dict[pd.Timestamp, int],
    day_to_indices: dict,
    target_idx: int,
    recent_steps: int,
    daily_lags: int,
    weekly_lags: int,
    similar_days: int,
    similarity_metric: str,
    pad_value: float,
) -> np.ndarray:
    target_time = timestamps[target_idx]
    n_segments = values.shape[1]

    blocks = [
        _recent_block(
            values=values,
            timestamps=timestamps,
            day_to_indices=day_to_indices,
            target_idx=target_idx,
            recent_steps=recent_steps,
            pad_value=pad_value,
        )
    ]

    for lag in range(1, daily_lags + 1):
        blocks.append(
            _lag_block(
                values=values,
                time_to_idx=time_to_idx,
                target_time=target_time,
                lag=pd.Timedelta(days=lag),
                n_segments=n_segments,
                pad_value=pad_value,
            )
        )

    for lag in range(1, weekly_lags + 1):
        blocks.append(
            _lag_block(
                values=values,
                time_to_idx=time_to_idx,
                target_time=target_time,
                lag=pd.Timedelta(weeks=lag),
                n_segments=n_segments,
                pad_value=pad_value,
            )
        )

    if similar_days > 0:
        blocks.extend(
            _similar_day_blocks(
                values=values,
                timestamps=timestamps,
                day_to_indices=day_to_indices,
                target_idx=target_idx,
                recent_steps=recent_steps,
                n_days=similar_days,
                similarity_metric=similarity_metric,
                pad_value=pad_value,
            )
        )

    return np.vstack(blocks)


def _recent_block(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    day_to_indices: dict,
    target_idx: int,
    recent_steps: int,
    pad_value: float,
) -> np.ndarray:
    target_day = timestamps[target_idx].date()
    n_segments = values.shape[1]

    previous = [
        idx for idx in day_to_indices[target_day]
        if idx < target_idx
    ][-recent_steps:]

    block = np.full((recent_steps, n_segments), pad_value, dtype=np.float32)

    if previous:
        block[-len(previous):] = values[previous]

    return block


def _lag_block(
    values: np.ndarray,
    time_to_idx: dict[pd.Timestamp, int],
    target_time: pd.Timestamp,
    lag: pd.Timedelta,
    n_segments: int,
    pad_value: float,
) -> np.ndarray:
    lagged_time = target_time - lag

    if lagged_time not in time_to_idx:
        return np.full((1, n_segments), pad_value, dtype=np.float32)

    return values[time_to_idx[lagged_time]][None, :]


def _similar_day_blocks(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    day_to_indices: dict,
    target_idx: int,
    recent_steps: int,
    n_days: int,
    similarity_metric: str,
    pad_value: float,
) -> list[np.ndarray]:
    target_time = timestamps[target_idx]
    target_profile = _previous_profile(
        values=values,
        timestamps=timestamps,
        target_idx=target_idx,
        recent_steps=recent_steps,
    )

    candidates = []

    for day, indices in day_to_indices.items():
        if day >= target_time.date():
            continue

        matched = [
            idx for idx in indices
            if timestamps[idx].time() == target_time.time()
        ]

        if not matched:
            continue

        candidate_idx = matched[0]
        candidate_profile = _previous_profile(
            values=values,
            timestamps=timestamps,
            target_idx=candidate_idx,
            recent_steps=recent_steps,
        )

        score = _mean_segment_similarity(
            target_profile,
            candidate_profile,
            metric=similarity_metric,
        )
        candidates.append((score, candidate_idx))

    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = [idx for _, idx in candidates[:n_days]]

    n_segments = values.shape[1]
    blocks = [values[idx][None, :] for idx in selected]

    while len(blocks) < n_days:
        blocks.append(
            np.full((1, n_segments), pad_value, dtype=np.float32)
        )

    return blocks


def _previous_profile(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    target_idx: int,
    recent_steps: int,
) -> np.ndarray:
    target_day = timestamps[target_idx].date()
    n_segments = values.shape[1]

    profile = np.full((recent_steps, n_segments), np.nan, dtype=np.float32)

    for offset in range(1, recent_steps + 1):
        idx = target_idx - offset

        if idx >= 0 and timestamps[idx].date() == target_day:
            profile[-offset] = values[idx]

    return profile


def _mean_segment_similarity(
    first: np.ndarray,
    second: np.ndarray,
    metric: str,
) -> float:
    scores = [
        _masked_similarity(first[:, i], second[:, i], metric)
        for i in range(first.shape[1])
    ]

    if not scores:
        return -1.0

    return float(np.nanmean(scores))


def _masked_similarity(
    first: np.ndarray,
    second: np.ndarray,
    metric: str,
) -> float:
    mask = np.isfinite(first) & np.isfinite(second)

    if mask.sum() < 2:
        return -1.0

    x = first[mask]
    y = second[mask]

    if x.std() == 0.0 or y.std() == 0.0:
        return -1.0

    if metric == "cosine":
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        if denominator == 0.0:
            return -1.0
        return float(np.dot(x, y) / denominator)

    if metric == "correlation":
        return float(np.corrcoef(x, y)[0, 1])

    raise ValueError(f"Unknown similarity metric: {metric}")


def _group_indices_by_day(
    timestamps: pd.DatetimeIndex,
) -> dict:
    groups = {}

    for idx, ts in enumerate(timestamps):
        groups.setdefault(ts.date(), []).append(idx)

    return groups