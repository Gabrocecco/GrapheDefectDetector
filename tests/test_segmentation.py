from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from graphene_defect_detector.segmentation import crop_to_mask, crops_to_masks


def test_crop_to_mask_preserves_shape(tmp_path: Path) -> None:
    image = np.zeros((20, 20, 3), dtype="uint8")
    image[5:15, 5:15] = (0, 0, 255)  # a red square, like a rendered bond
    crop_path = tmp_path / "graphene_1_bonds_cropped_box_0.png"
    cv2.imwrite(str(crop_path), image)

    mask = crop_to_mask(crop_path)

    assert mask.shape == image.shape


def test_crop_to_mask_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        crop_to_mask(tmp_path / "does_not_exist.png")


def test_crops_to_masks_processes_every_image(tmp_path: Path) -> None:
    crop_dir = tmp_path / "crops"
    mask_dir = tmp_path / "masks"
    crop_dir.mkdir()

    for i in range(3):
        image = np.zeros((10, 10, 3), dtype="uint8")
        image[2:8, 2:8] = (0, 0, 255)
        cv2.imwrite(str(crop_dir / f"graphene_1_bonds_cropped_box_{i}.png"), image)

    written = crops_to_masks(crop_dir, mask_dir)

    assert len(written) == 3
    assert all(p.exists() for p in written)
