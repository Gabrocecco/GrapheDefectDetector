# `models/`

- `defect_detector_yolov8.pt` — YOLOv8 weights fine-tuned to detect defects
  (voids) in rendered graphene bond-maps. Trained on 80 of the 100 hand-labeled
  images in `data/yolo_splits/train/` for 100 epochs (see
  `notebooks/train_yolo.ipynb`). Loaded by `detection.load_model` /
  `gdd detect`.

Retrain with `notebooks/train_yolo.ipynb` or:

```bash
gdd split-yolo-labels --source data/yolo_labeled --test-fraction 0.2
python -c "from graphene_defect_detector.detection import write_dataset_yaml; \
    write_dataset_yaml('data/yolo_labeled', 'configs/yolo_dataset.resolved.yaml')"
```

then `model.train(data="configs/yolo_dataset.resolved.yaml", epochs=100)` via
`graphene_defect_detector.detection.train`.

A `regressor.joblib` (the trained `GradientBoostingRegressor`, saved by
`gdd train --model-out models/regressor.joblib`) is not committed by default;
regenerate it with `gdd build-dataset && gdd train --dataset data/tables/energy_dataframe.csv --model-out models/regressor.joblib`.
