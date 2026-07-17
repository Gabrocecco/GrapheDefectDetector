# Graphene Defect Detector

[![CI](https://github.com/Gabrocecco/GrapheDefectDetector/actions/workflows/ci.yml/badge.svg)](https://github.com/Gabrocecco/GrapheDefectDetector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicting the **total energy** of simulated nanographene samples from the
geometry of their structural defects, using YOLOv8 object detection, OpenCV
shape analysis, and gradient-boosted regression.

Originally developed during an engineering thesis internship at
**CNR-ISMN** (Istituto per lo Studio dei Materiali Nanostrutturati),
University of Bologna — see [`docs/thesis_summary.md`](docs/thesis_summary.md)
for the full write-up this codebase implements. This repository is a
from-scratch engineering rewrite of that thesis's original codebase: the
pipeline, features, and model are unchanged, but every module was rebuilt as
a typed, tested, cross-platform Python package. See
[`REFACTORING_PLAN.md`](REFACTORING_PLAN.md) for what changed and why, and
[`docs/known_quirks.md`](docs/known_quirks.md) for two feature-computation
quirks kept intentionally identical to the original.

## Why

Graphene's exceptional properties come from its 2D honeycomb lattice, but
real samples are rarely defect-free — production leaves voids and
irregularities in the atomic structure. This project asks: can a sample's
physical properties be predicted directly from the *shape* of its defects,
automatically extracted from an image?

## Pipeline

```
.xyz atomic structure
        │  rendering.render_dataset       (chemfiles: guess bonds, draw them)
        ▼
  rendered PNG
        │  detection.crop_detections      (YOLOv8: detect + crop defect boxes)
        ▼
  defect crop(s)
        │  segmentation.crops_to_masks    (OpenCV: highlight the void)
        ▼
  defect mask(s)
        │  features.extract_shape_features (OpenCV: area, perimeter, circularity, ...)
        ▼
  per-defect shape features
        │  dataset.aggregate_by_sample + merge_energy_labels
        ▼
  per-sample modeling dataset  ──►  modeling.train_regressor  ──►  total_energy prediction
```

Each arrow is one module in `src/graphene_defect_detector/` and one
subcommand of the `gdd` CLI.

## Results

A `GradientBoostingRegressor` trained on 7 area-weighted shape descriptors
(area, pixel count, circularity, solidity, compactness, Feret diameter,
eccentricity) of ~2000 samples' defects predicts `total_energy` with
**R² ≈ 0.97** on a held-out test set. See
[`notebooks/demo.ipynb`](notebooks/demo.ipynb) for the full run and plots.

## Installation

```bash
git clone https://github.com/Gabrocecco/GrapheDefectDetector.git
cd GrapheDefectDetector

conda env create -f environment.yml
conda activate graphene-defect-detector
pip install -e ".[dev]"
```

Works on Linux (the primary development target) and Windows. Without conda,
any Python ≥3.10 virtualenv + `pip install -e ".[dev]"` works too.

## Quickstart

```bash
# Full pipeline: render -> detect -> segment -> build the modeling dataset
gdd run-pipeline --limit 50

# Or step by step:
gdd render --limit 50
gdd detect
gdd segment
gdd build-dataset --output data/tables/energy_dataframe.csv

# Train and evaluate the regressor
gdd train --dataset data/tables/energy_dataframe.csv --model-out models/regressor.joblib
```

All paths come from [`configs/pipeline.yaml`](configs/pipeline.yaml) — pass
`--config path/to/other.yaml` to point at a different data layout. See
`gdd --help` / `gdd <command> --help` for every option, or
[`notebooks/demo.ipynb`](notebooks/demo.ipynb) for an annotated walkthrough
with plots.

To retrain the YOLOv8 detector itself, see
[`notebooks/train_yolo.ipynb`](notebooks/train_yolo.ipynb).

## Project layout

```
src/graphene_defect_detector/   the package (rendering, detection, segmentation,
                                 features, dataset, modeling, plots, cli)
configs/                        pipeline.yaml, yolo_dataset.yaml
data/                            .xyz inputs + every intermediate pipeline artifact
                                 (see data/README.md)
models/                          YOLOv8 weights (see models/README.md)
notebooks/                       demo.ipynb, train_yolo.ipynb
tests/                           pytest suite (unit + regression tests against
                                 the historical thesis output)
docs/                            known_quirks.md, thesis_summary.md
```

## Testing & linting

```bash
pytest
ruff check src tests
black --check src tests
mypy src/graphene_defect_detector
```

`pre-commit install` wires the same checks into `git commit` (see
[`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

## Citation

If you use this work, please cite the original thesis (see the
`preferred-citation` entry in [`CITATION.cff`](CITATION.cff)):

> Ceccolini, G. (2023). *Modelli Predittivi basati su Computer Vision per
> Proprietà Fisiche di Nanografene*. Alma Mater Studiorum – Università di
> Bologna.

## Acknowledgments

Thesis advisor Prof. Paolo Bellavista (University of Bologna); internship
supervisor Dr. Francesco Mercuri and the DAIMON research group at CNR-ISMN.

## License

[MIT](LICENSE)
