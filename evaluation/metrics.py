from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, y_hat: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(y - y_hat)[mask].mean())


def rmse(y: np.ndarray, y_hat: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(((y - y_hat) ** 2)[mask].mean()))


def dtw_distance(
    first: np.ndarray,
    second: np.ndarray,
    normalize: bool = True,
) -> float:
    n = len(first)
    m = len(second)

    if n == 0 or m == 0:
        return float("nan")

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    steps = np.full((n + 1, m + 1), np.inf, dtype=np.float32)

    cost[0, 0] = 0.0
    steps[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance = abs(first[i - 1] - second[j - 1])

            candidates = (
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

            best = int(np.argmin(candidates))

            if best == 0:
                prev_i, prev_j = i - 1, j
            elif best == 1:
                prev_i, prev_j = i, j - 1
            else:
                prev_i, prev_j = i - 1, j - 1

            cost[i, j] = distance + cost[prev_i, prev_j]
            steps[i, j] = 1.0 + steps[prev_i, prev_j]

    if not normalize:
        return float(cost[n, m])

    return float(cost[n, m] / steps[n, m])