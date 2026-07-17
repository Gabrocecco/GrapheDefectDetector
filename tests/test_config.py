from pathlib import Path

from graphene_defect_detector.config import PipelineConfig


def test_from_yaml_resolves_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "data" / "xyz").mkdir(parents=True)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
        root: .
        xyz_dir: data/xyz
        renders_dir: data/renders
        crops_dir: data/crops
        masks_dir: data/masks
        contours_dir: data/contours
        tables_dir: data/tables
        weights_path: models/weights.pt
        energy_csv: data/tables/energy.csv
        yolo_labeled_dir: data/yolo_labeled
        yolo_splits_dir: data/yolo_splits
        render_resolution: 128
        """,
        encoding="utf-8",
    )

    config = PipelineConfig.from_yaml(config_path)

    assert config.root == tmp_path.resolve()
    assert config.xyz_dir == (tmp_path / "data" / "xyz").resolve()
    assert config.render_resolution == 128
    assert config.xyz_dir.is_absolute()


def test_ensure_output_dirs_creates_missing_directories(tmp_path: Path) -> None:
    config = PipelineConfig(
        root=tmp_path,
        xyz_dir=tmp_path / "xyz",
        renders_dir=tmp_path / "renders",
        crops_dir=tmp_path / "crops",
        masks_dir=tmp_path / "masks",
        contours_dir=tmp_path / "contours",
        tables_dir=tmp_path / "tables",
        weights_path=tmp_path / "weights.pt",
        energy_csv=tmp_path / "energy.csv",
        yolo_labeled_dir=tmp_path / "labeled",
        yolo_splits_dir=tmp_path / "splits",
    )

    config.ensure_output_dirs()

    for directory in (
        config.renders_dir,
        config.crops_dir,
        config.masks_dir,
        config.contours_dir,
        config.tables_dir,
    ):
        assert directory.is_dir()
