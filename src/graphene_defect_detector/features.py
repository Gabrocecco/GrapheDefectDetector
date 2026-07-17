"""Geometric feature extraction for a single defect mask.

Ported from ``Test.extract_shape_features_edited`` and
``Features.from_thresh_to_contours_print_features``
(``defect_analysis.py`` / ``lib_fetaure_analysis.py``).

The original module also defined a large family of unused feature
extractors copy-pasted from an unrelated melanoma-classification project
(GLCM/LBP texture, HOG, Fourier moments, Haralick, edge density, plus a
``Test.test_melanoma_mask`` helper). None of them were ever called by the
thesis's own pipeline -- only the shape features below feed the regressor
-- so they were dropped rather than translated. See ``REFACTORING_PLAN.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: A defect's source sample id is recovered from its filename by cutting
#: at the first occurrence of this marker (renders are named
#: "<sample_id>_bonds.png", crops/masks inherit that prefix).
_SAMPLE_ID_MARKER = "_bonds"


@dataclass
class ShapeFeatures:
    """Geometric measurements of one segmented defect."""

    image_name: str
    sample_id: str
    n_atoms: int
    area_px: float
    num_pixels: int
    perimeter_px: float
    circularity: float
    solidity: float
    compactness: float
    feret_diameter: float
    eccentricity: float

    def as_dict(self) -> dict:
        return {
            "image_name": self.image_name,
            "sample_id": self.sample_id,
            "n_atoms": self.n_atoms,
            "area_px": self.area_px,
            "num_pixels": self.num_pixels,
            "perimeter_px": self.perimeter_px,
            "circularity": self.circularity,
            "solidity": self.solidity,
            "compactness": self.compactness,
            "feret_diameter": self.feret_diameter,
            "eccentricity": self.eccentricity,
        }


def sample_id_from_filename(mask_filename: str) -> str:
    """Recover the originating ``.xyz`` sample id from a mask/crop filename."""
    return mask_filename.split(_SAMPLE_ID_MARKER)[0]


def read_atom_count(xyz_dir: str | Path, sample_id: str) -> int:
    """Read the atom count (first line) of ``<xyz_dir>/<sample_id>.xyz``."""
    xyz_path = Path(xyz_dir) / f"{sample_id}.xyz"
    with xyz_path.open("r", encoding="utf-8") as fh:
        return int(fh.readline().strip())


def extract_shape_features(
    mask_path: str | Path,
    xyz_dir: str | Path,
    contour_output_path: str | Path | None = None,
) -> ShapeFeatures:
    """Measure the largest contour in a defect mask.

    ``xyz_dir`` must contain the original ``.xyz`` structure the mask was
    rendered from, so the atom count can be attached to the feature row.

    If no contour is found (an empty or fully-uniform mask), degenerate
    placeholder values are returned instead of raising, matching the
    original pipeline's behavior on the handful of masks this happens to
    (very small or malformed detections).

    .. note::
       ``solidity`` here is **not** the standard ``area / convex_hull_area``
       ratio. The original code divided the contour area by the *number of
       vertices* in the convex hull (``cv2.convexHull(..., returnPoints=False).size``)
       rather than the hull's area -- a bug in the source thesis code. It is
       reproduced as-is because the published regressor (R^2 ~= 0.97) was
       trained on features computed this way; silently "fixing" it would
       change every downstream number without re-validating the model. See
       ``docs/known_quirks.md`` for a corrected reference implementation.
    """
    mask_path = Path(mask_path)
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1
    )
    contours = [c for c in contours if cv2.contourArea(c) >= 1]

    num_pixels = image.shape[0] * image.shape[1]

    if contours:
        largest = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        circularity = 4 * np.pi * area / (perimeter**2)
        solidity = area / cv2.convexHull(largest, returnPoints=False).size
        compactness = perimeter**2 / area

        _, (diam_x, diam_y), _ = cv2.minAreaRect(largest)
        feret_diameter = max(diam_x, diam_y)
        if diam_x < diam_y:
            diam_x, diam_y = diam_y, diam_x
        eccentricity = np.sqrt(1 - (diam_y / diam_x) ** 2)

        if contour_output_path is not None:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(annotated, contours, -1, (0, 0, 255), 1)
            cv2.imwrite(str(contour_output_path), annotated)
    else:
        logger.warning("No contours found in %s; using placeholder values", mask_path)
        # Area is set to 1 (not 0) to avoid downstream division-by-zero when
        # this row is aggregated (see dataset.aggregate_by_sample).
        area, perimeter = 1.0, 0.0
        circularity, solidity, compactness = 1.0, 0.0, 0.0
        feret_diameter, eccentricity = 1.0, 0.0

        if contour_output_path is not None:
            cv2.imwrite(
                str(contour_output_path), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            )

    sample_id = sample_id_from_filename(mask_path.name)
    n_atoms = read_atom_count(xyz_dir, sample_id)

    return ShapeFeatures(
        image_name=sample_id,
        sample_id=sample_id,
        n_atoms=n_atoms,
        area_px=area,
        num_pixels=num_pixels,
        perimeter_px=perimeter,
        circularity=circularity,
        solidity=solidity,
        compactness=compactness / num_pixels,
        feret_diameter=feret_diameter / np.sqrt(num_pixels),
        eccentricity=eccentricity,
    )


def extract_features_from_dir(
    masks_dir: str | Path,
    xyz_dir: str | Path,
    contours_dir: str | Path | None = None,
) -> list[ShapeFeatures]:
    """Run :func:`extract_shape_features` over every mask in ``masks_dir``."""
    masks_dir = Path(masks_dir)
    if contours_dir is not None:
        contours_dir = Path(contours_dir)
        contours_dir.mkdir(parents=True, exist_ok=True)

    results: list[ShapeFeatures] = []
    for mask_path in sorted(masks_dir.iterdir()):
        contour_output = (
            contours_dir / f"{mask_path.stem}_contour.png"
            if contours_dir is not None
            else None
        )
        results.append(extract_shape_features(mask_path, xyz_dir, contour_output))

    logger.info("Extracted shape features for %d masks", len(results))
    return results
