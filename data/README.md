# `data/`

Everything here is committed to the repository (see `REFACTORING_PLAN.md` for
why — the original decision was to keep weights and data in-tree rather than
introduce Git LFS or an external download step). Directories map onto
`configs/pipeline.yaml`.

| Directory | Produced by | Contents |
|---|---|---|
| `xyz/` | (input) atomistic simulations | ~4.3k `.xyz` structures: atom count, then one `<element> x y z` line per atom |
| `renders/` | `gdd render` / `rendering.render_dataset` | bond-map PNGs rendered from `xyz/` |
| `crops/` | `gdd detect` / `detection.crop_detections` | one PNG per YOLO-detected defect bounding box |
| `masks/` | `gdd segment` / `segmentation.crops_to_masks` | highlighted defect masks (see `docs/known_quirks.md` §2 for why these aren't clean binary images) |
| `contours/` | `gdd extract-features --with-contours` | debug images with the measured contour drawn on top of each mask |
| `tables/` | `gdd build-dataset` | feature tables and the ground-truth `dataset_graphene.csv`; see `tables/README.md` |
| `yolo_labeled/` | Label Studio export | flat `<name>.png` + `<name>.txt` YOLO-format detection labels (single class: `defect`) used to train the detector |
| `yolo_splits/` | `splitting.split_yolo_dataset` | `yolo_labeled/` already split into `train/` and `test/` |
| `real_world_demo/` | manual, exploratory | a couple of real microscopy images and intermediate results from the thesis's "can this generalize beyond simulated data?" experiment (Section 3.4.1) — kept as a starting point, not a supported pipeline path |

Regenerating everything from `xyz/` (the only directory that cannot be
regenerated -- it's the raw simulation output):

```bash
gdd run-pipeline
```
