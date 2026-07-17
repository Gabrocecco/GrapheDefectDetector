# Method summary

Condensed from *"Modelli Predittivi basati su Computer Vision per Proprietà
Fisiche di Nanografene"* (Gabriele Ceccolini, University of Bologna, A.Y.
2022/2023), an internship project at CNR-ISMN. This page summarizes the
method and findings in English; it is not a full translation of the thesis.

## Motivation

Graphene's exceptional mechanical and electronic properties come from its
single-atom-thick honeycomb lattice, but industrial-scale production rarely
yields a defect-free sample: voids and irregularities are common in the
atomic structure. This project asks whether a sample's physical properties
(specifically, its **total energy**, a proxy for structural stability) can be
predicted directly from the geometry of those defects, extracted
automatically from an image of the structure.

## Data

The dataset consists of ~4,300 simulated graphene structures in `.xyz`
format (element + 3D coordinates per atom), each with a ground-truth
`total_energy` computed by the originating atomistic simulation
(`total_energy = n_atoms * E_carbon - E_formation`, i.e. lower/more negative
energy corresponds to a more stable, less defective structure).

## Stage 1 — Rendering

Since the simulations produce coordinates, not images, each `.xyz` structure
is rendered to a 2D bond-map: bonds are guessed from interatomic distances
(`chemfiles`) and drawn as colored line segments on a black canvas, then
cropped to content. This produces the flat PNG images the rest of the
pipeline operates on. (`rendering.py`)

## Stage 2 — Defect detection (YOLOv8)

A **YOLOv8** object detector (single class: `defect`) was trained on 100
manually annotated renders (80 train / 20 test, ~100 epochs) to localize
defect regions as bounding boxes. YOLO's single-stage architecture — predict
class + bounding box coordinates in one forward pass over the whole image,
via a Backbone/Neck/Head structure — was chosen over two-stage detectors
(e.g. R-CNN family) for its inference speed and competitive accuracy with a
small, hand-labeled training set. Each detected box is cropped out of the
source render for the next stage. (`detection.py`)

## Stage 3 — Segmentation

Each crop is turned into a mask that highlights the defect's void region, by
binarizing the crop, finding contours, drawing them back onto the crop, and
re-thresholding — see [`known_quirks.md`](known_quirks.md) §2 for the exact
(and slightly unusual) sequence, preserved from the original implementation.
(`segmentation.py`)

## Stage 4 — Shape feature extraction (OpenCV)

The largest contour in each mask is measured for: area, perimeter,
circularity (`4*pi*area/perimeter^2`), solidity, compactness
(`perimeter^2/area`), Feret diameter (via the minimum-area bounding
rectangle), and eccentricity. See [`known_quirks.md`](known_quirks.md) §1 for
a caveat about the `solidity` feature specifically. (`features.py`)

## Stage 5 — Aggregation and correlation analysis

A sample can have more than one detected defect; per-defect features are
combined into one row per sample by summing area/pixel-count/perimeter and
taking an area-weighted mean of the remaining shape descriptors — larger
defects dominate a sample's aggregate geometry, matching the intuition that
they should matter more for stability. (`dataset.py`)

Correlation analysis (heatmap + pairplot over the aggregated features) showed
a strong, near-linear relationship between defect area and `total_energy`:
samples with larger voids consistently have higher (less stable) energy.
Highly-correlated feature pairs (e.g. area vs. perimeter) were identified so
redundant features could be excluded from modeling.

## Stage 6 — Predictive model

A `GradientBoostingRegressor` (scikit-learn) was trained on 7 selected,
weakly-correlated shape features to predict `total_energy`, achieving
**R² ≈ 0.97** on a held-out test set (~2000 samples with at least one
detected defect, 90/10 train/test split). Feature-importance analysis
confirmed defect area as the dominant predictor, though the full feature set
outperforms area alone. (`modeling.py`, `plots.py`)

## Limitations and future work

The pipeline was trained and validated entirely on simulated data. The
thesis's own exploration of generalizing to real microscopy images (Section
3.4.1 of the original document) found that YOLO's detections don't transfer
directly — real images are far more heterogeneous (noise, scale, resolution)
than the clean, uniform simulated renders it was trained on. A
denoising/normalization preprocessing step showed promise but mixed results;
see `data/real_world_demo/` for the surviving experiment artifacts. Closing
that sim-to-real gap (more diverse training data, or a preprocessing step
that reliably maps real images into the simulated-render distribution) is
the natural next step toward a tool usable directly on production samples.
