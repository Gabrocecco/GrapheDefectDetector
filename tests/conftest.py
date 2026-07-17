from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    return repo_root / "data"


@pytest.fixture(scope="session")
def sample_mask_path(data_dir: Path) -> Path:
    """A real committed defect mask: graphene_100134's first detected box.

    Its expected feature values are cross-checked in test_features.py against
    data/tables/box_dataframe.csv (the historical run's own output), giving a
    real regression test rather than a hand-picked synthetic case.
    """
    path = data_dir / "masks" / "graphene_100134_bonds_cropped_box_0_thresh_.png"
    if not path.exists():
        pytest.skip(f"Sample data not available: {path}")
    return path


@pytest.fixture(scope="session")
def xyz_dir(data_dir: Path) -> Path:
    return data_dir / "xyz"
