"""Turn a cropped defect image into a highlighted binary-ish mask.

This stage takes a single defect crop (a small colored render, produced by
:mod:`graphene_defect_detector.detection`) and produces a version where the
defect's void region is pushed toward white and everything else toward
black/red, so :mod:`graphene_defect_detector.features` can find a clean
contour around it.

Ported from ``Utils.from_crops_to_thresh`` (``lib/lib_utils.py``), kept
numerically identical to the original two-threshold trick:

1. Binarize the grayscale crop at threshold 0 (any non-zero pixel -> white).
2. Find contours on that binary image and draw them, in white, back onto a
   *copy of the original color crop* (not onto the binary mask).
3. Threshold that contour-annotated color image again, inverted at 245.

Step 3 is applied per BGR channel, which does not produce a clean black/white
mask -- the graphene lattice bonds (drawn in red) survive as colored
speckle around the highlighted void, visible in the thesis's own figures
(section 3.1, "Crop di YOLO dopo la trasformazione"). This is preserved
deliberately: :mod:`graphene_defect_detector.features` was tuned against
masks with exactly this look, and changing it would silently change the
downstream geometric features and the regressor's numbers. See
``docs/known_quirks.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

_CONTOUR_INTENSITY_THRESHOLD = 245
_CONTOUR_LINE_THICKNESS = 3


def crop_to_mask(image_path: str | Path) -> np.ndarray:
    """Apply the two-threshold defect-highlighting transform to one crop."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    annotated = image.copy()
    cv2.drawContours(
        annotated,
        contours,
        contourIdx=-1,
        color=(255, 255, 255),
        thickness=_CONTOUR_LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    _, mask = cv2.threshold(
        annotated,
        _CONTOUR_INTENSITY_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )
    return mask


def crops_to_masks(crop_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Apply :func:`crop_to_mask` to every crop in ``crop_dir``."""
    crop_dir = Path(crop_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for crop_path in sorted(crop_dir.iterdir()):
        if crop_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask = crop_to_mask(crop_path)
        output_path = output_dir / f"{crop_path.stem}_thresh.png"
        cv2.imwrite(str(output_path), mask)
        written.append(output_path)

    logger.info("Wrote %d defect masks to %s", len(written), output_dir)
    return written
