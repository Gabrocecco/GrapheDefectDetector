"""Render simulated ``.xyz`` atomic structures to 2D bond-map images.

The original simulation dataset provides atomic coordinates, not images. This
module reproduces each sample as a flat PNG by drawing a line for every
guessed bond, colored by the pair of elements it connects. Object detection
(:mod:`graphene_defect_detector.detection`) then runs on these renders.

Ported from the thesis's ``Utils.generate_bonds_png`` / ``Utils.crop_image`` /
``Utils.from_xyz_to_png`` (``lib/lib_utils.py``). The rendering math is kept
bit-for-bit identical to the original; only structure, naming and I/O were
cleaned up.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from chemfiles import Trajectory
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

#: Bond color by element pair, matching the original single-layer-carbon
#: rendering convention (only these four combinations are colored; any other
#: pair is left undrawn, same as the original behavior).
_BOND_COLORS: dict[frozenset[str], str] = {
    frozenset({"C"}): "red",  # C-C
    frozenset({"C", "O"}): "blue",
    frozenset({"O", "H"}): "white",
    frozenset({"C", "H"}): "yellow",
}


def resolve_bond_color(element_a: str, element_b: str) -> str | None:
    """Return the render color for a bond between two elements, or ``None``."""
    return _BOND_COLORS.get(frozenset({element_a, element_b}))


def crop_to_content(image: Image.Image) -> Image.Image:
    """Crop away the empty (all-zero) border around the rendered structure."""
    data = np.asarray(image)
    intensity = data if data.ndim == 2 else data.max(axis=2)

    non_empty_columns = np.where(intensity.max(axis=0) > 0)[0]
    non_empty_rows = np.where(intensity.max(axis=1) > 0)[0]
    if non_empty_columns.size == 0 or non_empty_rows.size == 0:
        return image

    top, bottom = int(non_empty_rows.min()), int(non_empty_rows.max())
    left, right = int(non_empty_columns.min()), int(non_empty_columns.max())

    box = (left, top, right + 1, bottom + 1)  # PIL crop box: (l, t, r, b)
    return image.crop(box)


def render_xyz_to_png(
    xyz_path: str | Path,
    output_dir: str | Path,
    resolution: int = 320,
) -> Path:
    """Render one ``.xyz`` structure to ``<output_dir>/<name>_bonds.png``.

    Bonds are guessed by ``chemfiles`` and drawn as colored line segments on a
    black canvas, then the image is cropped to its non-empty content.
    """
    xyz_path = Path(xyz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = xyz_path.stem

    with Trajectory(str(xyz_path)) as trajectory:
        frame = trajectory.read()

    frame.guess_bonds()
    if frame.topology.bonds_count() == 0:
        logger.warning("No bonds guessed for %s", name)

    canvas = Image.new("RGB", (resolution, resolution))
    draw = ImageDraw.Draw(canvas)

    for atom_a, atom_b in frame.topology.bonds:
        x1 = round(frame.positions[atom_a][0] * 2) + resolution / 2
        y1 = round(frame.positions[atom_a][1] * 2) + resolution / 2
        x2 = round(frame.positions[atom_b][0] * 2) + resolution / 2
        y2 = round(frame.positions[atom_b][1] * 2) + resolution / 2

        color = resolve_bond_color(
            frame.atoms[atom_a].name, frame.atoms[atom_b].name
        )
        draw.line([(x1, y1), (x2, y2)], fill=color, width=0)

    canvas = crop_to_content(canvas)
    output_path = output_dir / f"{name}_bonds.png"
    canvas.save(output_path)
    return output_path


def render_dataset(
    xyz_dir: str | Path,
    output_dir: str | Path,
    resolution: int = 320,
    limit: int | None = None,
) -> list[Path]:
    """Render every ``.xyz`` file in ``xyz_dir`` (or up to ``limit`` of them)."""
    xyz_dir = Path(xyz_dir)
    xyz_paths = sorted(xyz_dir.glob("*.xyz"))
    if limit is not None:
        xyz_paths = xyz_paths[:limit]

    rendered: list[Path] = []
    for i, xyz_path in enumerate(xyz_paths, start=1):
        rendered.append(render_xyz_to_png(xyz_path, output_dir, resolution))
        if i % 100 == 0:
            logger.info("Rendered %d/%d structures", i, len(xyz_paths))

    logger.info("Rendered %d structures to %s", len(rendered), output_dir)
    return rendered
