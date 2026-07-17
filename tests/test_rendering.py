from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from graphene_defect_detector.rendering import crop_to_content, resolve_bond_color


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("C", "C", "red"),
        ("C", "O", "blue"),
        ("O", "C", "blue"),
        ("O", "H", "white"),
        ("C", "H", "yellow"),
        ("H", "C", "yellow"),
        ("N", "C", None),
        ("N", "O", None),
    ],
)
def test_resolve_bond_color(a: str, b: str, expected: str | None) -> None:
    assert resolve_bond_color(a, b) == expected


def test_crop_to_content_removes_empty_border() -> None:
    data = np.zeros((100, 100, 3), dtype="uint8")
    data[40:60, 30:70] = 255  # a bright block off-center
    image = Image.fromarray(data)

    cropped = crop_to_content(image)

    assert cropped.size == (40, 20)  # (width, height) of the bright block


def test_crop_to_content_returns_original_for_blank_image() -> None:
    image = Image.new("RGB", (10, 10))
    cropped = crop_to_content(image)
    assert cropped.size == image.size


def test_render_xyz_to_png_requires_chemfiles(tmp_path: Path) -> None:
    pytest.importorskip("chemfiles")

    from graphene_defect_detector.rendering import render_xyz_to_png

    xyz_path = tmp_path / "toy.xyz"
    xyz_path.write_text("2\nToy\nC 0.0 0.0 0.0\nC 1.4 0.0 0.0\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    result = render_xyz_to_png(xyz_path, output_dir, resolution=64)

    assert result.exists()
    assert result.name == "toy_bonds.png"
