"""Pipeline configuration.

All stages of the pipeline (render -> detect -> segment -> extract features ->
build dataset -> train) read their input/output locations from a single
:class:`PipelineConfig` instead of hardcoded, machine-specific paths. This is
what makes the pipeline reproducible on any machine (Linux or Windows) and
from any working directory.

Paths in the YAML config are resolved relative to the config file's own
location, so the same ``configs/pipeline.yaml`` works regardless of the
current working directory the CLI is invoked from.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

#: Feature columns used by the regressor, in the original thesis order.
FEATURE_COLUMNS: tuple[str, ...] = (
    "area_px",
    "num_pixels",
    "circularity",
    "solidity",
    "compactness",
    "feret_diameter",
    "eccentricity",
)

TARGET_COLUMN = "total_energy"

#: Image extensions recognized throughout the pipeline (renders, crops, masks).
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

#: Default hyperparameters for the GradientBoostingRegressor, as used in the thesis.
DEFAULT_REGRESSOR_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}


@dataclass
class PipelineConfig:
    """Filesystem layout for one run of the pipeline.

    Every path is absolute after construction (see :meth:`from_yaml` /
    :meth:`resolve`), so downstream code never has to guess a base directory.
    """

    root: Path
    xyz_dir: Path
    renders_dir: Path
    crops_dir: Path
    masks_dir: Path
    contours_dir: Path
    tables_dir: Path
    weights_path: Path
    energy_csv: Path
    yolo_labeled_dir: Path
    yolo_splits_dir: Path
    render_resolution: int = 320

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a config from a YAML file, resolving relative paths against it."""
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        base = path.parent
        root = (base / raw.get("root", ".")).resolve()

        path_fields = {f.name for f in fields(cls) if f.name != "render_resolution"}
        kwargs: dict[str, Any] = {"root": root}
        for name in path_fields - {"root"}:
            if name not in raw:
                raise KeyError(f"Missing required key '{name}' in {path}")
            kwargs[name] = (root / raw[name]).resolve()

        kwargs["render_resolution"] = int(raw.get("render_resolution", 320))
        return cls(**kwargs)

    def ensure_output_dirs(self) -> None:
        """Create the directories the pipeline writes to, if missing."""
        for directory in (
            self.renders_dir,
            self.crops_dir,
            self.masks_dir,
            self.contours_dir,
            self.tables_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
