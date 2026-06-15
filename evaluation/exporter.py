from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_results(
    output_dir: str | Path,
    stib_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    test_idx,
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stib_test = stib_df.iloc[test_idx]
    reference_test = reference_df.iloc[test_idx]
    prediction_test = prediction_df.iloc[test_idx]

    stib_test.to_csv(
        output_dir / "stib_test.csv"
    )

    reference_test.to_csv(
        output_dir / "reference_test.csv"
    )

    prediction_test.to_csv(
        output_dir / "prediction_test.csv"
    )

    metrics_df.to_csv(
        output_dir / "metrics_per_segment.csv",
        index=False,
    )