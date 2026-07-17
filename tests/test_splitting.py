from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from graphene_defect_detector.splitting import split_yolo_dataset


def _make_pair(directory: Path, name: str) -> None:
    (directory / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    cv2.imwrite(str(directory / f"{name}.png"), np.zeros((8, 8, 3), dtype="uint8"))


def test_split_yolo_dataset_moves_pairs_and_respects_fraction(tmp_path: Path) -> None:
    for i in range(10):
        _make_pair(tmp_path, f"sample_{i}")

    train_dir, test_dir = split_yolo_dataset(tmp_path, test_fraction=0.3, seed=0)

    train_labels = sorted(train_dir.glob("*.txt"))
    test_labels = sorted(test_dir.glob("*.txt"))

    assert len(test_labels) == 3
    assert len(train_labels) == 7
    # Every label has its image sitting right next to it.
    for label in [*train_labels, *test_labels]:
        assert label.with_suffix(".png").exists()
    # Nothing left behind in the source directory.
    assert not list(tmp_path.glob("*.txt"))


def test_split_yolo_dataset_is_deterministic_with_seed(tmp_path: Path) -> None:
    for i in range(10):
        _make_pair(tmp_path, f"sample_{i}")

    train_dir, test_dir = split_yolo_dataset(tmp_path, test_fraction=0.2, seed=42)
    first_test_names = {p.stem for p in test_dir.glob("*.txt")}

    # Rebuild the same source layout and split again with the same seed.
    for p in [*train_dir.iterdir(), *test_dir.iterdir()]:
        p.rename(tmp_path / p.name)
    train_dir.rmdir()
    test_dir.rmdir()

    _, test_dir_2 = split_yolo_dataset(tmp_path, test_fraction=0.2, seed=42)
    second_test_names = {p.stem for p in test_dir_2.glob("*.txt")}

    assert first_test_names == second_test_names
