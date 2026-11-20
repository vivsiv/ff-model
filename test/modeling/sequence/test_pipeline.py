import os
import subprocess
import sys
from dataclasses import replace

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from mlflow.tracking import MlflowClient

from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.lstm import LSTMModel
from src.modeling.sequence.lstm_trainer import LSTMModelTrainer
from src.modeling.utils import load_data_builder_config


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
        builder, batch_size=2, tracking_dir=str(tmp_path / "mlruns"),
    )
    yield trainer
    mlflow.set_tracking_uri(old_tracking_uri)


def test_logged_pipeline_round_trip_includes_config_preprocessing_and_metrics(trainer, tmp_path):
    run_id = trainer.setup_mlflow(extra_params={"seed": 42})
    history = trainer.fit(epochs=2, run_id=run_id)
    test_mse = trainer.evaluate(split="test", run_id=run_id)

    run = mlflow.get_run(run_id)
    assert mlflow.get_experiment(run.info.experiment_id).name == "ppr_points_per_game_rb_wr_sequence"
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
        pipeline.preprocessor.named_steps["imputer"].statistics_,
        trainer.preprocessor.named_steps["imputer"].statistics_,
    )
    data = trainer.data["prediction"]
    inputs = {"X": data.X.numpy(), "lengths": data.lengths.numpy()}
    expected = trainer.get_pipeline().predict_dataset(data)
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


def test_pipeline_rejects_feature_order_mismatch(trainer):
    trainer.fit(epochs=1)
    data = replace(trainer.data["prediction"], feature_columns=["feature_b", "feature_a"])

    with pytest.raises(ValueError, match="Feature columns"):
        trainer.get_pipeline().predict_dataset(data)


def test_pipeline_requires_training_before_logging(trainer):
    with pytest.raises(ValueError, match="Call fit"):
        trainer.get_pipeline()


def test_pipeline_handles_empty_prediction_dataset(trainer):
    trainer.fit(epochs=1)
    pipeline = trainer.get_pipeline()

    actual = pipeline.predict(None, {
        "X": np.empty((0, 0, 2), dtype=np.float32),
        "lengths": np.empty(0, dtype=np.int64),
    })

    assert actual.shape == (0, 0)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("lengths", [np.array([0, 3]), np.array([1, 5]), np.array([1.0, 3.0])])
def test_pipeline_rejects_invalid_lengths(trainer, lengths):
    trainer.fit(epochs=1)

    with pytest.raises(ValueError, match="length"):
        trainer.get_pipeline().predict(None, {"X": trainer.data["prediction"].X.numpy(), "lengths": lengths})
