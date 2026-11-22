import json
import os
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from mlflow.tracking import MlflowClient

from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.lstm import LSTMModel
from src.modeling.sequence.lstm_trainer import LSTMModelTrainer
from src.modeling.sequence.pipeline import LSTMPipeline
from src.modeling.sequence.preprocessor import SequencePreprocessor
from src.modeling.tabular.data_builder import TabularModelDataBuilder
from src.modeling.tabular.trainer import TabularModelTrainer
from src.modeling.utils import load_data_builder_config


@pytest.mark.parametrize("input_size", [1, 5])
def test_build_signature_describes_raw_inputs_and_per_step_outputs(input_size):
    signature = LSTMPipeline.build_signature(input_size)
    X_spec, lengths_spec = signature.inputs.inputs
    output_spec, = signature.outputs.inputs

    assert X_spec.name == "X"
    assert X_spec.type == np.dtype("float32")
    assert X_spec.shape == (-1, -1, input_size)
    assert lengths_spec.name == "lengths"
    assert lengths_spec.type == np.dtype("int64")
    assert lengths_spec.shape == (-1,)
    assert output_spec.type == np.dtype("float32")
    assert output_spec.shape == (-1, -1)


@pytest.fixture
def trainer(tmp_path):
    data = SequenceDataset(
        X=torch.tensor([
            [[1.0, float("nan")], [float("nan"), float("nan")], [float("nan"), float("nan")], [float("nan"), float("nan")]],
            [[2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [float("nan"), float("nan")]],
        ]),
        lengths=torch.tensor([1, 3]),
        identity=pd.DataFrame(),
        feature_columns=["feature_a", "feature_b"],
        y=torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]]),
        loss_mask=torch.tensor([[True, False, False, False], [False, False, True, False]]),
        sample_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]),
    )
    builder = SequenceModelDataBuilder(str(tmp_path), {
        "target": "ppr_points_per_game",
        "positions": ["WR", "RB"],
        "split": {"prediction_season": 2026},
    })
    builder._data_splits = {
        "train": data, "eval": replace(data, sample_weight=None),
        "test": replace(data, sample_weight=None),
        "prediction": replace(data, y=torch.zeros_like(data.y), loss_mask=torch.zeros_like(data.loss_mask), sample_weight=None),
    }
    old_tracking_uri = mlflow.get_tracking_uri()
    trainer = LSTMModelTrainer(
        LSTMModel(input_size=2, hidden_size=4, num_layers=2, dropout=0.2),
        builder, SequencePreprocessor(), batch_size=2, tracking_dir=str(tmp_path / "mlruns"),
    )
    yield trainer
    mlflow.set_tracking_uri(old_tracking_uri)


@pytest.fixture
def fitted_pipeline(trainer):
    trainer.fit(epochs=1)
    return LSTMPipeline(trainer.model, trainer.preprocessor, trainer.prepared_data["train"].feature_columns)


def test_logged_pipeline_round_trip_includes_config_preprocessing_and_metrics(trainer, tmp_path):
    run_id = trainer.setup_mlflow(extra_params={"seed": 42})
    history = trainer.fit(epochs=2, run_id=run_id)
    test_mse = trainer.evaluate(split="test", run_id=run_id)

    run = mlflow.get_run(run_id)
    assert mlflow.get_experiment(run.info.experiment_id).name == "ppr_points_per_game_rb_wr"
    assert run.data.tags["model_family"] == "sequence"
    assert run.data.tags["model_type"] == "lstm"
    assert run.data.tags["phase"] == "train"
    assert run.data.params["hidden_size"] == "4"
    assert run.data.params["num_layers"] == "2"
    assert run.data.params["imputation_strategy"] == "mean"
    assert load_data_builder_config(run_id) == trainer.data_builder.config
    assert run.data.metrics["test_mse"] == pytest.approx(test_mse)
    assert run.data.metrics["test_rmse"] == pytest.approx(test_mse ** 0.5)
    metric_history = MlflowClient().get_metric_history(run_id, "train_mse")
    assert [metric.step for metric in metric_history] == [1, 2]
    assert [metric.value for metric in metric_history] == pytest.approx([item["train_mse"] for item in history])

    model_uri = "models:/ppr_points_per_game_rb_wr_lstm/1"
    loaded = mlflow.pyfunc.load_model(model_uri)
    pipeline = loaded.unwrap_python_model()
    assert pipeline.feature_columns == ["feature_a", "feature_b"]
    assert next(pipeline.model.parameters()).device.type == "cpu"
    np.testing.assert_allclose(
        pipeline.preprocessor.pipeline.named_steps["imputer"].statistics_,
        trainer.preprocessor.pipeline.named_steps["imputer"].statistics_,
    )
    data = trainer.data["prediction"]
    inputs = {"X": data.X.numpy(), "lengths": data.lengths.numpy()}
    in_memory = LSTMPipeline(trainer.model, trainer.preprocessor, trainer.prepared_data["train"].feature_columns)
    expected = in_memory.predict_dataset(data)
    np.testing.assert_allclose(loaded.predict(inputs), expected, rtol=1e-5, atol=1e-6)
    trainer.model.eval()
    with torch.no_grad():
        direct = trainer.model(trainer.prepared_data["prediction"].X, data.lengths).numpy()
    np.testing.assert_allclose(expected, direct, rtol=1e-5, atol=1e-6)
    assert np.isfinite(expected).all()
    assert (expected[~data.real_step_mask.numpy()] == 0.0).all()

    # The logged signature must accept widths/batch sizes different from training.
    wider = np.pad(inputs["X"], ((0, 0), (0, 3), (0, 0)), constant_values=np.nan)
    wider_predictions = loaded.predict({"X": wider, "lengths": inputs["lengths"]})
    np.testing.assert_allclose(wider_predictions[:, :4], expected, rtol=1e-5, atol=1e-6)
    assert (wider_predictions[:, 4:] == 0.0).all()
    smaller = loaded.predict({"X": inputs["X"][:1, :1], "lengths": inputs["lengths"][:1]})
    np.testing.assert_allclose(smaller, expected[:1, :1], rtol=1e-5, atol=1e-6)

    # Confirm saved code/artifacts work outside the repository in a fresh process.
    arrays_path = tmp_path / "inference.npz"
    np.savez(arrays_path, **inputs, expected=expected)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([
        sys.executable, "-c",
        "import sys, mlflow, numpy as np; "
        "mlflow.set_tracking_uri(sys.argv[1]); "
        "arrays = np.load(sys.argv[2]); "
        f"model = mlflow.pyfunc.load_model('{model_uri}'); "
        "actual = model.predict({'X': arrays['X'], 'lengths': arrays['lengths']}); "
        "np.testing.assert_allclose(actual, arrays['expected'], rtol=1e-5, atol=1e-6)",
        trainer.tracking_dir, str(arrays_path),
    ], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_tabular_and_sequence_runs_share_base_model_experiment(trainer, tmp_path):
    builder = Mock(spec=TabularModelDataBuilder)
    builder.target = trainer.target
    builder.positions = trainer.data_builder.positions
    builder.config = trainer.data_builder.config
    builder.get_data_splits.return_value = {}
    tabular = TabularModelTrainer(
        data_dir=str(tmp_path), tracking_dir=trainer.tracking_dir,
        data_builder=builder, model_type="ridge",
    )

    sequence_run = mlflow.get_run(trainer.setup_mlflow())
    tabular_run = mlflow.get_run(tabular.setup_mlflow())

    assert sequence_run.info.experiment_id == tabular_run.info.experiment_id
    assert sequence_run.data.tags["model_family"] == "sequence"
    assert tabular_run.data.tags["model_family"] == "tabular"
    assert sequence_run.data.tags["model_type"] == "lstm"
    assert tabular_run.data.tags["model_type"] == "ridge"


def test_trainer_log_model_delegates_fitted_components_and_raw_example(trainer):
    trainer.fit(epochs=1)
    model_info = object()

    with patch.object(LSTMPipeline, "log_pipeline", return_value=model_info) as log_pipeline:
        actual = trainer.log_model("training-run")

    assert actual is model_info
    log_pipeline.assert_called_once()
    kwargs = log_pipeline.call_args.kwargs
    assert kwargs["model"] is trainer.model
    assert kwargs["preprocessor"] is trainer.preprocessor
    assert kwargs["feature_columns"] == trainer.prepared_data["train"].feature_columns
    assert kwargs["run_id"] == "training-run"
    assert kwargs["registered_model_name"] == "ppr_points_per_game_rb_wr_lstm"
    np.testing.assert_allclose(kwargs["input_example"]["X"], trainer.data["train"].X[:2].numpy(), equal_nan=True)
    np.testing.assert_array_equal(kwargs["input_example"]["lengths"], trainer.data["train"].lengths[:2].numpy())


def test_log_pipeline_stages_artifacts_and_cleans_temporary_files(trainer):
    trainer.fit(epochs=1)
    model_info = object()
    staged_paths = []

    def record_log_call(**kwargs):
        staged_paths.extend(kwargs["artifacts"].values())
        assert all(Path(path).is_file() for path in staged_paths)
        assert isinstance(kwargs["python_model"], LSTMPipeline)
        assert kwargs["registered_model_name"] == "test_lstm"
        assert kwargs["signature"].inputs.inputs[0].shape == (-1, -1, 2)
        return model_info

    with patch("src.modeling.sequence.pipeline.mlflow.start_run") as start_run:
        with patch("src.modeling.sequence.pipeline.mlflow.pyfunc.log_model", side_effect=record_log_call):
            actual = LSTMPipeline.log_pipeline(
                model=trainer.model, preprocessor=trainer.preprocessor,
                feature_columns=trainer.prepared_data["train"].feature_columns,
                run_id="training-run", registered_model_name="test_lstm",
                input_example={"X": trainer.data["train"].X.numpy(), "lengths": trainer.data["train"].lengths.numpy()},
            )

    start_run.assert_called_once_with(run_id="training-run")
    assert actual is model_info
    assert not any(Path(path).exists() for path in staged_paths)


def test_save_artifacts_matches_load_context_without_changing_live_model(trainer, tmp_path):
    trainer.fit(epochs=1)
    trainer.model.train()
    original_device = next(trainer.model.parameters()).device
    directory = tmp_path / "staging"

    artifacts = LSTMPipeline._save_artifacts(
        directory, trainer.model, trainer.preprocessor, trainer.prepared_data["train"].feature_columns,
    )

    assert artifacts == {
        "preprocessor": str(directory / "preprocessor.pkl"),
        "weights": str(directory / "weights.pt"),
        "metadata": str(directory / "metadata.json"),
    }
    assert all(Path(path).is_file() for path in artifacts.values())
    assert trainer.model.training
    assert next(trainer.model.parameters()).device == original_device
    with open(artifacts["metadata"]) as file:
        assert json.load(file) == {
            "model_params": {"input_size": 2, "hidden_size": 4, "num_layers": 2, "dropout": 0.2},
            "feature_columns": ["feature_a", "feature_b"],
        }
    state = torch.load(artifacts["weights"], map_location="cpu", weights_only=True)
    assert state.keys() == trainer.model.state_dict().keys()
    for key, tensor in state.items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, trainer.model.state_dict()[key].cpu())

    restored = LSTMPipeline()
    restored.load_context(SimpleNamespace(artifacts=artifacts))
    in_memory = LSTMPipeline(trainer.model, trainer.preprocessor, trainer.prepared_data["train"].feature_columns)
    np.testing.assert_allclose(
        restored.predict_dataset(trainer.data["prediction"]),
        in_memory.predict_dataset(trainer.data["prediction"]),
        rtol=1e-5, atol=1e-6,
    )


def test_load_context_supports_legacy_sklearn_preprocessor(trainer, tmp_path):
    trainer.fit(epochs=1)
    artifacts = LSTMPipeline._save_artifacts(
        tmp_path, trainer.model, trainer.preprocessor, trainer.prepared_data["train"].feature_columns,
    )
    with open(artifacts["preprocessor"], "wb") as file:
        pickle.dump(trainer.preprocessor.pipeline, file)
    pipeline = LSTMPipeline()

    pipeline.load_context(SimpleNamespace(artifacts=artifacts))

    assert isinstance(pipeline.preprocessor, SequencePreprocessor)
    in_memory = LSTMPipeline(trainer.model, trainer.preprocessor, trainer.prepared_data["train"].feature_columns)
    np.testing.assert_allclose(
        pipeline.predict_dataset(trainer.data["prediction"]),
        in_memory.predict_dataset(trainer.data["prediction"]),
        rtol=1e-5, atol=1e-6,
    )


def test_pipeline_rejects_feature_order_mismatch(trainer, fitted_pipeline):
    data = replace(trainer.data["prediction"], feature_columns=["feature_b", "feature_a"])

    with pytest.raises(ValueError, match="Feature columns"):
        fitted_pipeline.predict_dataset(data)


def test_log_model_requires_training(trainer):
    with pytest.raises(ValueError, match="Call fit"):
        trainer.log_model("unused-run")


def test_pipeline_handles_empty_prediction_dataset(fitted_pipeline):
    actual = fitted_pipeline.predict(None, {
        "X": np.empty((0, 0, 2), dtype=np.float32),
        "lengths": np.empty(0, dtype=np.int64),
    })

    assert actual.shape == (0, 0)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("lengths", [np.array([0, 3]), np.array([1, 5]), np.array([1.0, 3.0])])
def test_pipeline_rejects_invalid_lengths(trainer, fitted_pipeline, lengths):
    with pytest.raises(ValueError, match="length"):
        fitted_pipeline.predict(None, {"X": trainer.data["prediction"].X.numpy(), "lengths": lengths})


def test_pipeline_validates_empty_input_structure_before_returning(fitted_pipeline):
    with pytest.raises(ValueError, match="lengths must have shape"):
        fitted_pipeline.predict(None, {
            "X": np.empty((0, 0, 2), dtype=np.float32),
            "lengths": np.array([1], dtype=np.int64),
        })


def test_pipeline_rejects_invalid_input_rank(fitted_pipeline):
    with pytest.raises(ValueError, match="X must have shape"):
        fitted_pipeline.predict(None, {
            "X": np.zeros((2, 2), dtype=np.float32),
            "lengths": np.array([1, 1], dtype=np.int64),
        })


def test_pipeline_retains_model_specific_feature_count_check(trainer, fitted_pipeline):
    fitted_pipeline.model = LSTMModel(input_size=3)

    with pytest.raises(ValueError, match="model input_size"):
        fitted_pipeline.predict(None, {
            "X": trainer.data["prediction"].X.numpy(),
            "lengths": trainer.data["prediction"].lengths.numpy(),
        })
