"""YOLOv8-based defect detection and cropping.

Runs the trained defect detector over rendered graphene images and saves one
cropped PNG per detected bounding box, one directory of crops per source
image. Downstream, :mod:`graphene_defect_detector.segmentation` turns each
crop into a binary defect mask and :mod:`graphene_defect_detector.features`
measures its geometry.

Ported from ``Utils.crop_from_folder`` (``lib/lib_utils.py``).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
from ultralytics import YOLO

from .config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

#: Relative path (from the dataset yaml) to the label config for training.
DEFAULT_CLASS_NAME = "defect"


def load_model(weights_path: str | Path) -> YOLO:
    """Load a YOLO model (trained weights or a base architecture)."""
    return YOLO(str(weights_path))


def crop_detections(
    image_dir: str | Path,
    output_dir: str | Path,
    model: YOLO,
    confidence: float | None = None,
) -> list[Path]:
    """Detect defects in every image of ``image_dir`` and save cropped boxes.

    For each source image ``name.png``, writes ``name_cropped_box_<i>.png``
    for every bounding box the model finds, ``i`` in detection order.
    Returns the list of paths written.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    written: list[Path] = []
    predict_kwargs = {} if confidence is None else {"conf": confidence}

    for image_path in image_paths:
        results = model.predict(source=str(image_path), verbose=False, **predict_kwargs)
        boxes = results[0].boxes
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Could not read image %s, skipping", image_path)
            continue

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = math.floor(x1), math.floor(y1)
            x2, y2 = math.ceil(x2), math.ceil(y2)

            cropped = image[y1:y2, x1:x2]
            crop_name = f"{image_path.stem}_cropped_box_{index}.png"
            crop_path = output_dir / crop_name
            cv2.imwrite(str(crop_path), cropped)
            written.append(crop_path)

    logger.info("Wrote %d defect crops to %s", len(written), output_dir)
    return written


def write_dataset_yaml(
    splits_dir: str | Path,
    output_path: str | Path,
    class_name: str = DEFAULT_CLASS_NAME,
) -> Path:
    """Write a YOLO dataset yaml pointing at an absolute ``splits_dir``.

    Ultralytics resolves relative ``path:`` entries against its own
    ``datasets_dir`` setting rather than the yaml's location, which is a
    common source of "file not found" surprises. Writing an absolute path
    here makes the generated config portable across machines and OSes
    without depending on Ultralytics' global settings.
    """
    import yaml

    splits_dir = Path(splits_dir).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "path": str(splits_dir),
        "train": "train/",
        "val": "test/",
        "names": {0: class_name},
    }
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return output_path


def train(
    dataset_yaml: str | Path,
    weights_path: str | Path = "yolov8n.pt",
    epochs: int = 100,
    **train_kwargs,
) -> YOLO:
    """Train (or fine-tune) a YOLO model on the defect dataset."""
    model = load_model(weights_path)
    model.train(data=str(dataset_yaml), epochs=epochs, **train_kwargs)
    return model
