import pytest

from graphene_defect_detector.cli import build_parser


@pytest.mark.parametrize(
    "command",
    [
        "render",
        "detect",
        "segment",
        "extract-features",
        "build-dataset",
        "train",
        "split-yolo-labels",
        "run-pipeline",
    ],
)
def test_subcommand_is_registered(command: str) -> None:
    parser = build_parser()
    # argparse raises SystemExit(2) on missing required args; what matters
    # here is that the subcommand itself is recognized, not a full run.
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--help"])


def test_version_flag() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])


def test_train_requires_dataset_argument() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
