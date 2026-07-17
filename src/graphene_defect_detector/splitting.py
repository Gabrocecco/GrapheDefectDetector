"""Split a flat YOLO-labeled dataset into ``train/`` and ``test/`` folders.

Ported from ``Utils.split_dataset`` (``lib/lib_utils.py``). The original
implementation shelled out to ``mv`` via ``os.system``, which does not work
on Windows and breaks on paths containing spaces. This version uses
``shutil.move`` and ``pathlib`` throughout, so it behaves the same way on
Linux and Windows.

It is also stricter about what it moves: the original moved *every*
remaining file in the source directory into ``train/`` (including any
non-image, non-label file that happened to be sitting there); this version
only ever moves recognized ``(label, image)`` pairs.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

from .config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def split_yolo_dataset(
    source_dir: str | Path,
    test_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[Path, Path]:
    """Split ``<image>.png`` / ``<image>.txt`` pairs into ``train/`` and ``test/``.

    Returns ``(train_dir, test_dir)``.
    """
    source_dir = Path(source_dir)
    train_dir = source_dir / "train"
    test_dir = source_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    label_paths = sorted(source_dir.glob("*.txt"))
    pairs: list[tuple[Path, Path]] = []
    for label_path in label_paths:
        image_path = _find_image_for_label(label_path)
        if image_path is None:
            logger.warning("No image found for label %s, skipping", label_path)
            continue
        pairs.append((label_path, image_path))

    rng = random.Random(seed)
    test_count = int(len(pairs) * test_fraction)
    test_pairs = set(rng.sample(pairs, test_count)) if test_count else set()

    for label_path, image_path in pairs:
        destination = test_dir if (label_path, image_path) in test_pairs else train_dir
        shutil.move(str(label_path), str(destination / label_path.name))
        shutil.move(str(image_path), str(destination / image_path.name))

    logger.info(
        "Split %d pairs: %d train, %d test", len(pairs), len(pairs) - test_count, test_count
    )
    return train_dir, test_dir


def _find_image_for_label(label_path: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = label_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None
