# Refactoring Plan — Graphene Defect Detector

This document describes the radical refactoring of the original thesis codebase
(*"Modelli Predittivi basati su Computer Vision per Proprietà Fisiche di
Nanografene"*, G. Ceccolini, University of Bologna, A.Y. 2022/2023) into a
professional, cross-platform research repository.

The **science is preserved**; only the engineering is rebuilt. The target
predictor still estimates the `total_energy` of a simulated nanographene sample
from the geometry of its structural defects (reported R² ≈ 0.97).

---

## 1. Goals

- **English everywhere** — identifiers, comments, docstrings, docs.
- **Cross-platform** — Linux is the primary environment, but the code must also
  run on Windows. No `os.system('mv ...')`, no `str.split('/')` path parsing, no
  hardcoded absolute paths.
- **Installable package** with a clean `src/` layout and a single CLI.
- **Reproducible** — every pipeline stage runs from config, no magic paths.
- **Tested & linted** — pytest, ruff, black, mypy, pre-commit, CI.
- **Honest git hygiene** — the build environment is no longer tracked.

## 2. Original pipeline (preserved semantics)

| Stage | Original code | New module |
|-------|---------------|------------|
| 1. Render `.xyz` → `.png` (bond drawing) | `Utils.generate_bonds_png`, `from_xyz_to_png` | `rendering.py` |
| 2. YOLOv8 detection + crop defects | `Utils.crop_from_folder` | `detection.py` |
| 3. Crop → binary threshold (void mask) | `Utils.from_crops_to_thresh` | `segmentation.py` |
| 4. OpenCV shape features | `Test.extract_shape_features_edited`, `Features.from_thresh_to_contours_print_features` | `features.py` |
| 5. Aggregate + merge energy + regressor | `main_playground.ipynb` | `dataset.py`, `modeling.py`, `plots.py` |

Shape features retained: `area_px`, `num_pixels`, `perimeter_px`, `circularity`,
`solidity`, `compactness`, `feret_diameter`, `eccentricity`. Per-sample
aggregation keeps the original **area-weighted mean** (simple sum for area /
perimeter / pixel counts).

## 3. Problems being fixed

- Fake classes (`Test`, `Utils`, `Features`) used as bare namespaces → module functions.
- Dermatology ("melanoma") leftovers copied from another project → deleted.
- Large commented-out dead-code blocks → deleted.
- Hardcoded `/home/gabro/GrapheDetectProject/...` paths → config + CLI arguments.
- `os.system('mv')` and `path.split('/')` → `shutil` / `pathlib`, OS-agnostic.
- `try/except: print()` around imports (then `NameError`) → real dependencies.
- Typos: `lib_fetaure_analysis.py`, `numero_atomi`, `trasforma_in_rosso_nero`.
- No `.gitignore`; the whole `.conda/` env (8k files) and `__pycache__` tracked.
- `environment.yml` is a Linux conda lock with build strings → cross-platform env.

## 4. Target layout

```
graphene-defect-detector/
├── pyproject.toml
├── README.md · LICENSE · CITATION.cff · .gitignore · .pre-commit-config.yaml
├── environment.yml              # cross-platform, unpinned build strings
├── configs/
│   ├── pipeline.yaml
│   └── yolo_dataset.yaml
├── src/graphene_defect_detector/
│   ├── config.py
│   ├── rendering.py
│   ├── detection.py
│   ├── segmentation.py
│   ├── features.py
│   ├── dataset.py
│   ├── modeling.py
│   ├── plots.py
│   └── cli.py
├── scripts/
├── notebooks/demo.ipynb
├── tests/
├── data/README.md
└── docs/
```

## 5. Phases

- **Phase 0 — hygiene.** `.gitignore`; untrack `.conda/` and `__pycache__/`
  (kept in history, removed from tree). Weights and data stay committed.
- **Phase 1 — skeleton.** `pyproject.toml`, `src/` package, English module names,
  functions + type hints + docstrings + `logging`.
- **Phase 2 — port stages.** Translate, delete dead/foreign code, remove hardcoded
  paths, make cross-platform.
- **Phase 3 — CLI + config.** `gdd render|detect|segment|features|build|train|predict`
  driven by `configs/pipeline.yaml`.
- **Phase 4 — tests + CI.** pytest fixtures, ruff/black/mypy, GitHub Actions
  (Linux primary, Windows smoke), pre-commit.
- **Phase 5 — docs.** English README + method notes distilled from the thesis,
  data-acquisition guide, GrapheNet citation.

## 6. Decisions on record

- **Git history:** keep history; stop tracking the conda env and bytecode caches.
  No history rewrite / force-push.
- **Weights & data:** `best_100_campioni_new.pt` and the dataset stay committed.
- **Primary OS:** Linux; Windows must also work.
