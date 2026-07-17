from pathlib import Path

import pandas as pd
import pytest

from graphene_defect_detector.dataset import aggregate_by_sample, merge_energy_labels


@pytest.fixture
def defect_table() -> pd.DataFrame:
    # Two defects for "sample_a" (areas 10 and 30) and one for "sample_b".
    return pd.DataFrame.from_records(
        [
            {
                "sample_id": "sample_a",
                "n_atoms": 100,
                "area_px": 10.0,
                "num_pixels": 50,
                "perimeter_px": 5.0,
                "circularity": 1.0,
                "solidity": 2.0,
                "compactness": 0.1,
                "feret_diameter": 0.5,
                "eccentricity": 0.2,
            },
            {
                "sample_id": "sample_a",
                "n_atoms": 100,
                "area_px": 30.0,
                "num_pixels": 70,
                "perimeter_px": 15.0,
                "circularity": 0.5,
                "solidity": 4.0,
                "compactness": 0.2,
                "feret_diameter": 0.7,
                "eccentricity": 0.6,
            },
            {
                "sample_id": "sample_b",
                "n_atoms": 200,
                "area_px": 5.0,
                "num_pixels": 20,
                "perimeter_px": 2.0,
                "circularity": 0.9,
                "solidity": 1.0,
                "compactness": 0.05,
                "feret_diameter": 0.3,
                "eccentricity": 0.1,
            },
        ]
    )


def test_aggregate_by_sample_sums_area_and_weights_shape_descriptors(
    defect_table: pd.DataFrame,
) -> None:
    result = aggregate_by_sample(defect_table).set_index("sample_id")

    # area/pixels/perimeter: plain sum
    assert result.loc["sample_a", "area_px"] == pytest.approx(40.0)
    assert result.loc["sample_a", "num_pixels"] == pytest.approx(120.0)
    assert result.loc["sample_a", "perimeter_px"] == pytest.approx(20.0)

    # circularity: area-weighted mean = (1.0*10 + 0.5*30) / 40 = 0.625
    assert result.loc["sample_a", "circularity"] == pytest.approx(0.625)

    # single-defect sample: aggregation is a no-op on shape descriptors
    assert result.loc["sample_b", "area_px"] == pytest.approx(5.0)
    assert result.loc["sample_b", "circularity"] == pytest.approx(0.9)


def test_merge_energy_labels_attaches_target_and_per_atom_column(tmp_path: Path) -> None:
    sample_table = pd.DataFrame(
        {"sample_id": ["sample_a", "sample_b"], "n_atoms": [100, 200], "area_px": [40.0, 5.0]}
    )
    energy_csv = tmp_path / "energy.csv"
    pd.DataFrame({"file_name": ["sample_a", "sample_b"], "total_energy": [-1000.0, -400.0]}).to_csv(
        energy_csv, index=False
    )

    merged = merge_energy_labels(sample_table, energy_csv)

    assert set(merged["sample_id"]) == {"sample_a", "sample_b"}
    assert "file_name" not in merged.columns
    row = merged.set_index("sample_id").loc["sample_a"]
    assert row["total_energy"] == pytest.approx(-1000.0)
    assert row["total_energy_per_atom"] == pytest.approx(-10.0)


def test_merge_energy_labels_drops_unlabeled_samples(tmp_path: Path) -> None:
    sample_table = pd.DataFrame({"sample_id": ["sample_a", "no_label"], "n_atoms": [100, 50]})
    energy_csv = tmp_path / "energy.csv"
    pd.DataFrame({"file_name": ["sample_a"], "total_energy": [-1000.0]}).to_csv(
        energy_csv, index=False
    )

    merged = merge_energy_labels(sample_table, energy_csv)

    assert list(merged["sample_id"]) == ["sample_a"]
