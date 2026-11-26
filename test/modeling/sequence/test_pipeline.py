import os
import subprocess
import sys
from dataclasses import replace
from importlib.metadata import version as package_version
from pathlib import Path
from unittest.mock import Mock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch

from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.lstm import LSTMModel
from src.modeling.sequence.lstm_trainer import LSTMModelTrainer
from src.modeling.sequence.pipeline import LSTMPipeline
from src.modeling.sequence.preprocessor import SequencePreprocessor
from src.modeling.tabular.data_builder import TabularModelDataBuilder
from src.modeling.tabular.trainer import TabularModelTrainer


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
        identity=pd.DataFrame(columns=["seq_idx", "step_idx"]),
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


@pytest.fixture
def prediction_inputs(trainer):
    data = trainer.data["prediction"]
    return {"X": data.X.numpy(), "lengths": data.lengths.numpy()}


@pytest.fixture
def logged_model_info(trainer, fitted_pipeline, prediction_inputs):
    run_id = trainer.setup_mlflow()

    return LSTMPipeline.log_pipeline(
        model=fitted_pipeline.model, preprocessor=fitted_pipeline.preprocessor,
        feature_columns=fitted_pipeline.feature_columns, run_id=run_id,
        registered_model_name=f"{trainer.base_model_name}_lstm",
        input_example=prediction_inputs,
    )


@pytest.fixture
def logged_pipeline(logged_model_info):
    return mlflow.pyfunc.load_model(logged_model_info.model_uri)


def test_logged_pipeline_restores_in_memory_state(fitted_pipeline, logged_pipeline):
    restored = logged_pipeline.unwrap_python_model()

    assert restored.feature_columns == fitted_pipeline.feature_columns
    assert LSTMPipeline.model_params(restored.model) == LSTMPipeline.model_params(fitted_pipeline.model)
    assert next(restored.model.parameters()).device.type == "cpu"
    assert not restored.model.training

    expected_state = fitted_pipeline.model.state_dict()
    actual_state = restored.model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name].cpu())

    expected_preprocessor = fitted_pipeline.preprocessor.preprocessor.named_steps
    actual_preprocessor = restored.preprocessor.preprocessor.named_steps
    np.testing.assert_allclose(actual_preprocessor["imputer"].statistics_, expected_preprocessor["imputer"].statistics_)
    np.testing.assert_allclose(actual_preprocessor["scaler"].mean_, expected_preprocessor["scaler"].mean_)
    np.testing.assert_allclose(actual_preprocessor["scaler"].scale_, expected_preprocessor["scaler"].scale_)


def test_logged_predictions_match_in_memory_pipeline(trainer, fitted_pipeline, logged_pipeline, prediction_inputs):
    expected = fitted_pipeline.predict(context=None, model_input=prediction_inputs)
    actual = logged_pipeline.predict(prediction_inputs)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert np.isfinite(actual).all()
    assert (actual[~trainer.data["prediction"].real_step_mask.numpy()] == 0.0).all()


def test_in_memory_predictions_match_direct_lstm_on_preprocessed_inputs(trainer, fitted_pipeline, prediction_inputs):
    actual = fitted_pipeline.predict(context=None, model_input=prediction_inputs)
    data = trainer.prepared_data["prediction"]

    with torch.no_grad():
        expected = trainer.model(data.X, data.lengths).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_logged_pipeline_accepts_larger_sequence_dimension_than_training(fitted_pipeline, logged_pipeline, prediction_inputs):
    expected = fitted_pipeline.predict(context=None, model_input=prediction_inputs)
    original_width = prediction_inputs["X"].shape[1]
    wider = np.pad(prediction_inputs["X"], ((0, 0), (0, 3), (0, 0)), constant_values=np.nan)

    actual = logged_pipeline.predict({"X": wider, "lengths": prediction_inputs["lengths"]})

    np.testing.assert_allclose(actual[:, :original_width], expected, rtol=1e-5, atol=1e-6)
    assert (actual[:, original_width:] == 0.0).all()


def test_logged_pipeline_accepts_smaller_sequence_dimension_and_batch_size(fitted_pipeline, logged_pipeline, prediction_inputs):
    expected = fitted_pipeline.predict(context=None, model_input=prediction_inputs)

    actual = logged_pipeline.predict({
        "X": prediction_inputs["X"][:1, :1], "lengths": prediction_inputs["lengths"][:1],
    })

    np.testing.assert_allclose(actual, expected[:1, :1], rtol=1e-5, atol=1e-6)


def test_logged_pipeline_loads_in_readonly_working_directory_in_fresh_process(
    trainer, fitted_pipeline, logged_model_info, prediction_inputs, tmp_path,
):
    expected = fitted_pipeline.predict(context=None, model_input=prediction_inputs)
    arrays_path = tmp_path / "inference.npz"
    np.savez(arrays_path, **prediction_inputs, expected=expected)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    working_directory = tmp_path / "readonly-inference"
    working_directory.mkdir()
    # A directory at the former log-file path also prevents opening that file
    # when tests run as root, which can bypass read-only directory permissions.
    (working_directory / "gold_builder.log").mkdir()
    working_directory.chmod(0o555)
    try:
        result = subprocess.run([
            sys.executable, "-c",
            "import sys, mlflow, numpy as np; "
            "mlflow.set_tracking_uri(sys.argv[1]); "
            "arrays = np.load(sys.argv[2]); "
            f"model = mlflow.pyfunc.load_model('{logged_model_info.model_uri}'); "
            "actual = model.predict({'X': arrays['X'], 'lengths': arrays['lengths']}); "
            "np.testing.assert_allclose(actual, arrays['expected'], rtol=1e-5, atol=1e-6)",
            trainer.tracking_dir, str(arrays_path),
        ], cwd=working_directory, env=environment, capture_output=True, text=True, timeout=60)
    finally:
        working_directory.chmod(0o755)

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


@pytest.mark.parametrize("installed_torch_version", ["2.14.1", "2.14.1+cpu", "2.14.1+cu130"])
def test_log_pipeline_pins_public_torch_version(trainer, installed_torch_version):
    trainer.preprocessor.fit(trainer.data["train"])

    def installed_version(package):
        return installed_torch_version if package == "torch" else package_version(package)

    with (
        patch("src.modeling.sequence.pipeline.version", side_effect=installed_version),
        patch("src.modeling.sequence.pipeline.mlflow.start_run"),
        patch("src.modeling.sequence.pipeline.mlflow.pyfunc.log_model") as log_model,
    ):
        LSTMPipeline.log_pipeline(
            model=trainer.model, preprocessor=trainer.preprocessor,
            feature_columns=trainer.data["train"].feature_columns,
            run_id="training-run", registered_model_name="test_lstm",
            input_example={"X": trainer.data["train"].X.numpy(), "lengths": trainer.data["train"].lengths.numpy()},
        )

    requirements = log_model.call_args.kwargs["pip_requirements"]
    assert [requirement for requirement in requirements if requirement.startswith("torch==")] == ["torch==2.14.1"]


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


@pytest.mark.parametrize("max_steps", [0, 3])
def test_logged_pipeline_handles_empty_input(logged_pipeline, max_steps):
    actual = logged_pipeline.predict({
        "X": np.empty((0, max_steps, 2), dtype=np.float32),
        "lengths": np.empty(0, dtype=np.int64),
    })

    assert actual.shape == (0, max_steps)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("invalid_input", ["missing-lengths", "wrong-X-dtype", "wrong-X-rank"])
def test_logged_pipeline_enforces_input_signature(logged_pipeline, prediction_inputs, invalid_input):
    inputs = dict(prediction_inputs)
    if invalid_input == "missing-lengths":
        del inputs["lengths"]
    elif invalid_input == "wrong-X-dtype":
        inputs["X"] = inputs["X"].astype(np.float64)
    else:
        inputs["X"] = inputs["X"][:, 0]

    with pytest.raises(mlflow.exceptions.MlflowException):
        logged_pipeline.predict(inputs)


def test_logged_pipeline_rejects_mismatched_player_counts(logged_pipeline, prediction_inputs):
    # Each array satisfies its own signature; their batch dimensions must also
    # agree, which is checked when the pipeline constructs a SequenceDataset.
    inputs = {"X": prediction_inputs["X"], "lengths": prediction_inputs["lengths"][:1]}

    with pytest.raises(ValueError, match="lengths must have shape"):
        logged_pipeline.predict(inputs)


def test_pipeline_validates_empty_input_structure_before_returning(fitted_pipeline):
    with pytest.raises(ValueError, match="lengths must have shape"):
        fitted_pipeline.predict(context=None, model_input={
            "X": np.empty((0, 0, 2), dtype=np.float32),
            "lengths": np.array([1], dtype=np.int64),
        })


def test_pipeline_validates_model_input_size_matches_feature_count(trainer, fitted_pipeline):
    fitted_pipeline.model = LSTMModel(input_size=3)

    with pytest.raises(ValueError, match="model input_size"):
        fitted_pipeline.predict(context=None, model_input={
            "X": trainer.data["prediction"].X.numpy(),
            "lengths": trainer.data["prediction"].lengths.numpy(),
        })
