from __future__ import annotations

import pandas as pd


def align_reference_timeline(
    source: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Align reference data to the source timeline."""

    return reference.reindex(source.index)


def get_common_segments(
    source: pd.DataFrame,
    reference: pd.DataFrame,
) -> list[str]:
    """Return the ordered set of shared road segments."""

    common_segments = sorted(
        set(source.columns) & set(reference.columns)
    )

    if not common_segments:
        raise ValueError(
            "No common road segments were found between the input datasets."
        )

    return common_segments


def align_datasets(
    source: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Align two traffic datasets in both temporal and spatial dimensions.
    """

    reference = align_reference_timeline(source, reference)

    segments = get_common_segments(
        source=source,
        reference=reference,
    )

    return (
        source[segments],
        reference[segments],
        segments,
    )