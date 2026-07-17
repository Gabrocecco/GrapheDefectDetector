"""Build the modeling dataset: per-defect features -> per-sample dataset.

A rendered graphene sample can contain more than one detected defect, so
:mod:`graphene_defect_detector.features` produces one row *per defect box*.
This module aggregates those rows back to one row *per sample* (the unit
``total_energy`` is defined on) and attaches the physical target label.

Ported from the dataset-building cell of ``main_playground.ipynb``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .features import ShapeFeatures, extract_features_from_dir

logger = logging.getLogger(__name__)

#: Columns aggregated as an area-weighted mean across a sample's defects.
_AREA_WEIGHTED_COLUMNS = (
    "circularity",
    "solidity",
    "compactness",
    "feret_diameter",
    "eccentricity",
)

#: Columns aggregated as a plain sum across a sample's defects.
_SUMMED_COLUMNS = ("area_px", "num_pixels", "perimeter_px")


def build_defect_table(
    masks_dir: str | Path,
    xyz_dir: str | Path,
    contours_dir: str | Path | None = None,
) -> pd.DataFrame:
    """One row per detected+segmented defect, with its raw shape features."""
    rows: list[ShapeFeatures] = extract_features_from_dir(masks_dir, xyz_dir, contours_dir)
    return pd.DataFrame.from_records([r.as_dict() for r in rows])


def aggregate_by_sample(defect_table: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-defect rows into one row per source sample.

    Defects belonging to the same sample are combined so that a sample with
    a large, area-dominant defect is not diluted by several tiny ones: area,
    pixel count and perimeter are summed, while shape descriptors
    (circularity, solidity, compactness, feret diameter, eccentricity) are
    combined as an area-weighted mean.
    """
    if defect_table.empty:
        return defect_table

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        weights = group["area_px"]
        total_weight = weights.sum()
        weighted = {
            col: (group[col] * weights).sum() / total_weight for col in _AREA_WEIGHTED_COLUMNS
        }
        summed = {col: group[col].sum() for col in _SUMMED_COLUMNS}
        return pd.Series(
            {
                "n_atoms": group["n_atoms"].max(),  # identical within a group
                **summed,
                **weighted,
            }
        )

    aggregated = defect_table.groupby("sample_id").apply(_aggregate)
    return aggregated.reset_index()


def merge_energy_labels(
    sample_table: pd.DataFrame,
    energy_csv: str | Path,
    id_column: str = "sample_id",
    file_name_column: str = "file_name",
    energy_column: str = "total_energy",
) -> pd.DataFrame:
    """Attach the physical target (``total_energy``) to each sample row."""
    energy_df = pd.read_csv(energy_csv)
    merged = sample_table.merge(
        energy_df[[file_name_column, energy_column]],
        left_on=id_column,
        right_on=file_name_column,
    )
    if file_name_column != id_column:
        merged = merged.drop(columns=[file_name_column])

    merged[f"{energy_column}_per_atom"] = merged[energy_column] / merged["n_atoms"]
    return merged


def build_modeling_dataset(
    masks_dir: str | Path,
    xyz_dir: str | Path,
    energy_csv: str | Path,
    contours_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full features -> aggregate -> merge chain.

    Returns ``(defect_table, sample_table, modeling_table)`` -- the raw
    per-defect rows, the per-sample aggregation, and the final table with
    ``total_energy`` attached, in that order (mirrors ``box_dataframe.csv``,
    ``groupedDataframe.csv`` and ``energy_dataframe.csv`` from the original
    notebook).
    """
    defect_table = build_defect_table(masks_dir, xyz_dir, contours_dir)
    sample_table = aggregate_by_sample(defect_table)
    modeling_table = merge_energy_labels(sample_table, energy_csv)
    return defect_table, sample_table, modeling_table
