from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_speed_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="raise")

    df = df.set_index(time_col)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def load_adjacency_matrix(
    path: str | Path,
    ordered_nodes: list[str],
) -> np.ndarray:
    adjacency = pd.read_csv(path, index_col=0)

    missing_rows = set(ordered_nodes) - set(adjacency.index)
    missing_cols = set(ordered_nodes) - set(adjacency.columns)

    if missing_rows or missing_cols:
        raise ValueError(
            "Adjacency matrix does not contain all required segment identifiers."
        )

    adjacency = adjacency.loc[ordered_nodes, ordered_nodes]
    adjacency = adjacency.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return adjacency.to_numpy(dtype=np.float32)