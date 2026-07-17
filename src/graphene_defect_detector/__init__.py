"""Graphene Defect Detector.

Predicts physical properties (total energy) of simulated nanographene
samples from the geometry of their structural defects, detected with a
YOLOv8 object detector and measured with OpenCV.

Pipeline stages, one module each:

    rendering    .xyz atomic structure -> rendered bond-map PNG
    detection    YOLOv8 defect detection + cropping
    segmentation crop -> highlighted defect mask
    features     mask -> geometric shape features
    dataset      per-defect features -> per-sample modeling dataset
    modeling     GradientBoostingRegressor training/evaluation
    plots        diagnostic plots
    splitting    train/test split for YOLO label data

See ``configs/pipeline.yaml`` for the default directory layout and
``graphene_defect_detector.cli`` for the command-line entry point (``gdd``).
"""

__version__ = "0.1.0"
