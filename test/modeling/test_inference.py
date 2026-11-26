from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.modeling.data_builder import ModelDataBuilder
from src.modeling.inference import ModelInference


@pytest.fixture
def inference():
    builder = Mock(spec=ModelDataBuilder)
    builder.config = {"target": "ppr_points_per_game", "split": {"prediction_season": 2026}}

    return ModelInference(
        pipeline=object(), data_builder=builder, tracking_dir="mlruns",
        registered_model="registered_model", model_version=2, source_run_id="training-run",
    )


def test_from_registry_rejects_a_version_without_training_run():
    with (
        patch("src.modeling.inference.load_mlflow_model", return_value=(object(), SimpleNamespace(run_id=None))),
        patch("src.modeling.inference.load_data_builder_config") as load_config,
    ):
        with pytest.raises(ValueError, match="no source training run"):
            ModelInference.from_registry("data", "mlruns", "registered_model")

    load_config.assert_not_called()


def test_init_stores_common_metadata_without_loading_splits(inference):
    assert inference.config is inference.data_builder.config
    assert inference.tracking_dir == "mlruns"
    assert inference.registered_model == "registered_model"
    assert inference.model_version == 2
    assert inference.source_run_id == "training-run"
    inference.data_builder.get_data_splits.assert_not_called()


@pytest.mark.parametrize("missing", ["tracking_dir", "registered_model", "model_version", "source_run_id"])
def test_setup_mlflow_requires_registry_metadata(inference, missing):
    setattr(inference, missing, None)

    with pytest.raises(ValueError, match="from_registry"):
        inference.setup_mlflow(phase="test")


@pytest.mark.parametrize("suffix, expected", [(None, "test"), ("predictions", "predictions"), ("", "")])
def test_setup_mlflow_defaults_or_preserves_run_name_suffix(inference, suffix, expected):
    source_run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="original-experiment"),
        data=SimpleNamespace(tags={"model_type": "lstm", "model_family": "sequence"}),
    )
    with (
        patch("src.modeling.inference.set_mlflow_tracking_uri") as tracking,
        patch("src.modeling.inference.mlflow.get_run", return_value=source_run),
        patch("src.modeling.inference.mlflow.get_experiment", return_value=SimpleNamespace(name="legacy_experiment")),
        patch("src.modeling.inference.setup_mlflow_run", return_value="inference-run") as setup,
        patch("src.modeling.inference.mlflow.log_dict") as log_config,
        patch("src.modeling.inference.datetime") as clock,
    ):
        clock.now.return_value.strftime.return_value = "timestamp"
        run_id = inference.setup_mlflow(phase="test", run_name_suffix=suffix)

    assert run_id == "inference-run"
    tracking.assert_called_once_with("mlruns")
    setup.assert_called_once_with(
        experiment_name="legacy_experiment",
        run_name=f"registered_model_v2_{expected}_timestamp",
        tracking_dir="mlruns", tags={"phase": "test", "model_family": "sequence", "model_type": "lstm"},
        params={"source_run_id": "training-run", "model_version": 2},
    )
    log_config.assert_called_once_with(inference.config, "data_prep_config.json", run_id="inference-run")


@pytest.mark.parametrize("split", ["prediction", "eval", "test"])
def test_build_predictions_csv_path_creates_directory(tmp_path, split):
    directory = tmp_path / "predictions"

    actual = ModelInference.build_predictions_csv_path(str(directory), "registered_model", 2, "run", split=split)

    assert directory.is_dir()
    assert Path(actual) == directory / f"registered_model_v2_{split}_run.csv"
