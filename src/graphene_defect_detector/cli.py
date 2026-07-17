"""Command-line entry point for the pipeline (``gdd``).

Every subcommand reads its I/O locations from a :class:`PipelineConfig`
(``--config``, default ``configs/pipeline.yaml``), so the same command works
unmodified on any machine or OS, as long as the config's relative paths are
respected.

Examples
--------
    gdd render --limit 50
    gdd detect
    gdd segment
    gdd extract-features --with-contours
    gdd build-dataset --output data/tables/energy_dataframe.csv
    gdd train --dataset data/tables/energy_dataframe.csv --model-out models/regressor.joblib
    gdd run-pipeline --limit 50   # render -> detect -> segment -> features -> dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import PipelineConfig
from .dataset import build_modeling_dataset

DEFAULT_CONFIG = Path("configs/pipeline.yaml")


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the pipeline config (default: {DEFAULT_CONFIG})",
    )


def cmd_render(args: argparse.Namespace) -> None:
    from .rendering import render_dataset

    config = PipelineConfig.from_yaml(args.config)
    config.ensure_output_dirs()
    render_dataset(config.xyz_dir, config.renders_dir, config.render_resolution, args.limit)


def cmd_detect(args: argparse.Namespace) -> None:
    from .detection import crop_detections, load_model

    config = PipelineConfig.from_yaml(args.config)
    config.ensure_output_dirs()
    model = load_model(config.weights_path)
    crop_detections(config.renders_dir, config.crops_dir, model, args.confidence)


def cmd_segment(args: argparse.Namespace) -> None:
    from .segmentation import crops_to_masks

    config = PipelineConfig.from_yaml(args.config)
    config.ensure_output_dirs()
    crops_to_masks(config.crops_dir, config.masks_dir)


def cmd_extract_features(args: argparse.Namespace) -> None:
    from .features import extract_features_from_dir

    config = PipelineConfig.from_yaml(args.config)
    config.ensure_output_dirs()
    contours_dir = config.contours_dir if args.with_contours else None
    rows = extract_features_from_dir(config.masks_dir, config.xyz_dir, contours_dir)

    import pandas as pd

    output_path = args.output or (config.tables_dir / "defect_features.csv")
    pd.DataFrame.from_records([r.as_dict() for r in rows]).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    config = PipelineConfig.from_yaml(args.config)
    config.ensure_output_dirs()
    contours_dir = config.contours_dir if args.with_contours else None

    defect_table, sample_table, modeling_table = build_modeling_dataset(
        config.masks_dir, config.xyz_dir, config.energy_csv, contours_dir
    )

    output_path = args.output or (config.tables_dir / "energy_dataframe.csv")
    defect_table.to_csv(config.tables_dir / "box_dataframe.csv", index=False)
    sample_table.to_csv(config.tables_dir / "grouped_dataframe.csv", index=False)
    modeling_table.to_csv(output_path, index=False)
    print(f"Wrote modeling dataset ({len(modeling_table)} samples) to {output_path}")


def cmd_train(args: argparse.Namespace) -> None:
    import pandas as pd

    from .modeling import evaluate, train_regressor

    dataset = pd.read_csv(args.dataset)
    trained = train_regressor(dataset, test_size=args.test_size, random_state=args.seed)
    metrics = evaluate(trained)
    print(f"Test MSE: {metrics['mse']:.4f} | Test R^2: {metrics['r2']:.4f}")

    if args.model_out:
        trained.save(args.model_out)
        print(f"Saved model to {args.model_out}")


def cmd_split_yolo_labels(args: argparse.Namespace) -> None:
    from .splitting import split_yolo_dataset

    train_dir, test_dir = split_yolo_dataset(args.source, args.test_fraction, args.seed)
    print(f"train: {train_dir}\ntest: {test_dir}")


def cmd_run_pipeline(args: argparse.Namespace) -> None:
    cmd_render(args)
    cmd_detect(args)
    cmd_segment(args)
    cmd_build_dataset(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdd", description="Graphene Defect Detector pipeline CLI"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("render", help="Render .xyz structures to PNG")
    _add_config_arg(p)
    p.add_argument("--limit", type=int, default=None, help="Render at most N structures")
    p.set_defaults(func=cmd_render)

    p = subparsers.add_parser("detect", help="Detect and crop defects with YOLO")
    _add_config_arg(p)
    p.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
    p.set_defaults(func=cmd_detect)

    p = subparsers.add_parser("segment", help="Threshold defect crops into masks")
    _add_config_arg(p)
    p.set_defaults(func=cmd_segment)

    p = subparsers.add_parser("extract-features", help="Measure shape features of each mask")
    _add_config_arg(p)
    p.add_argument(
        "--with-contours", action="store_true", help="Also save annotated contour images"
    )
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=cmd_extract_features)

    p = subparsers.add_parser(
        "build-dataset", help="Extract features, aggregate by sample, merge total_energy"
    )
    _add_config_arg(p)
    p.add_argument("--with-contours", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=cmd_build_dataset)

    p = subparsers.add_parser("train", help="Train the total_energy regressor")
    p.add_argument("--dataset", type=Path, required=True, help="CSV from build-dataset")
    p.add_argument("--model-out", type=Path, default=None)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = subparsers.add_parser("split-yolo-labels", help="Split labeled data into train/test")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_split_yolo_labels)

    p = subparsers.add_parser("run-pipeline", help="render -> detect -> segment -> build-dataset")
    _add_config_arg(p)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--confidence", type=float, default=None)
    p.add_argument("--with-contours", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=cmd_run_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
