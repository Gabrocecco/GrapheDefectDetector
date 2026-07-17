import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from graphene_defect_detector.features import (
    extract_shape_features,
    read_atom_count,
    sample_id_from_filename,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("graphene_67_bonds_cropped_box_0_thresh_.png", "graphene_67"),
        ("graphene_100134_bonds.png", "graphene_100134"),
        ("graphene_4363_bonds_cropped_box_2_thresh_.png", "graphene_4363"),
    ],
)
def test_sample_id_from_filename(filename: str, expected: str) -> None:
    assert sample_id_from_filename(filename) == expected


def test_read_atom_count(xyz_dir: Path) -> None:
    if not (xyz_dir / "graphene_100134.xyz").exists():
        pytest.skip("Sample .xyz not available")
    assert read_atom_count(xyz_dir, "graphene_100134") == 434


def test_extract_shape_features_matches_historical_output(
    sample_mask_path: Path, xyz_dir: Path, data_dir: Path
) -> None:
    """Regression test: this is a direct port, so the numbers must match
    data/tables/box_dataframe.csv (the historical thesis run's own output)
    to the precision floating point round-tripping through CSV allows."""
    historical_csv = data_dir / "tables" / "box_dataframe.csv"
    if not historical_csv.exists():
        pytest.skip("Historical reference table not available")

    historical = pd.read_csv(historical_csv)
    expected = historical[
        (historical["nome_imm"] == "graphene_100134") & (historical["area_px"] == 48.5)
    ].iloc[0]

    result = extract_shape_features(sample_mask_path, xyz_dir)

    assert result.sample_id == "graphene_100134"
    assert result.n_atoms == expected["numero_atomi"]
    assert result.area_px == pytest.approx(expected["area_px"])
    assert result.num_pixels == expected["num_pixels"]
    assert result.perimeter_px == pytest.approx(expected["perimeter_px"])
    assert result.circularity == pytest.approx(expected["circularity"])
    assert result.solidity == pytest.approx(expected["solidity"])
    assert result.compactness == pytest.approx(expected["compactness"])
    assert result.feret_diameter == pytest.approx(expected["feret_diameter"])
    assert result.eccentricity == pytest.approx(expected["eccentricity"])


def test_extract_shape_features_handles_empty_mask(tmp_path: Path, xyz_dir: Path) -> None:
    """A uniform (contour-less) mask must fall back to placeholder values,
    not raise."""
    if not (xyz_dir / "graphene_100134.xyz").exists():
        pytest.skip("Sample .xyz not available")

    blank = np.zeros((10, 10), dtype="uint8")
    mask_path = tmp_path / "graphene_100134_bonds_cropped_box_0_thresh_.png"
    cv2.imwrite(str(mask_path), blank)

    result = extract_shape_features(mask_path, xyz_dir)

    assert result.area_px == 1.0
    assert result.perimeter_px == 0.0
    assert not math.isnan(result.circularity)
