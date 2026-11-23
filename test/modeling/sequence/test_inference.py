from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError
from mlflow.tracking import MlflowClient

from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.inference import SequenceModelInference, main
from src.modeling.sequence.lstm import LSTMModel
from src.modeling.sequence.pipeline import LSTMPipeline
from src.modeling.sequence.preprocessor import SequencePreprocessor
from src.modeling.utils import load_data_builder_config, setup_mlflow_run


@pytest.fixture
def data():
    return SequenceDataset(
        X=torch.tensor([
            [[1.0, 10.0], [float("nan"), float("nan")]],
            [[2.0, 20.0], [3.0, float("nan")]],
        ]),
        lengths=torch.tensor([1, 2]),
        identity=pd.DataFrame({
            "seq_idx": [0, 1, 1], "step_idx": [0, 0, 1],
            "player_id": ["A", "B", "B"], "season": [2020, 2020, 2021],
            "target_season": [2022, 2021, 2022],
        }),
        feature_columns=["feature_a", "feature_b"],
        y=torch.tensor([[2.0, float("nan")], [float("nan"), 4.0]]),
        loss_mask=torch.tensor([[True, False], [False, True]]),
        sample_weight=torch.tensor([[1.0, 0.0], [0.0, 3.0]]),
    )


@pytest.fixture
def builder(data):
    builder = Mock(spec=SequenceModelDataBuilder)
    builder.target = "ppr_points_per_game"
    builder.config = {"target": builder.target, "split": {"prediction_season": 2022}}
    builder.get_data_splits.return_value = {
        "eval": data,
        "test": replace(data, y=torch.full((2, 2), 5.0)),
        "prediction": replace(data, y=torch.zeros_like(data.y), loss_mask=torch.zeros_like(data.loss_mask), sample_weight=None),
    }

    return builder


@pytest.fixture
def inference(data, builder):
    model = LSTMModel(input_size=2, hidden_size=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    pipeline = LSTMPipeline(model, SequencePreprocessor().fit(data), data.feature_columns)

    return SequenceModelInference(pipeline=pipeline, data_builder=builder)


@pytest.fixture
def registry(inference, data, tmp_path):
    """A logged model and synthetic gold snapshot for registry/CLI tests."""
    data_dir = tmp_path / "data"
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True)
    rows = [
        ("A", 2020, 2021, 2.0), ("A", 2021, 2022, 3.0),
        ("A", 2022, 2023, 4.0), ("A", 2023, 2024, np.nan),
        ("B", 2020, 2021, np.nan), ("B", 2021, 2022, 5.0),
        ("B", 2022, 2023, 6.0), ("B", 2023, 2024, np.nan),
    ]
    source = pd.DataFrame({
        "player_id": [row[0] for row in rows],
        "player_display_name": [f"Player {row[0]}" for row in rows],
        "position": ["WR"] * len(rows), "recent_team": ["KC"] * len(rows),
        "season": [row[1] for row in rows], "target_season": [row[2] for row in rows],
        "target": [row[3] for row in rows],
        "feature_a": [float(row[1] - 2019) for row in rows],
        "feature_b": [float((row[1] - 2019) * 10) for row in rows],
    })
    source.to_csv(gold_dir / "ppr_points_per_game__2024.csv", index=False)
    config = {
        "target": "ppr_points_per_game", "positions": ["WR"],
        "split": {"prediction_season": 2024},
        "features": {"mode": "include", "columns": data.feature_columns},
    }
    tracking_dir = str(tmp_path / "mlruns")
    old_tracking_uri = mlflow.get_tracking_uri()
    run_id = setup_mlflow_run(
        "saved_sequence_experiment", "train", tracking_dir=tracking_dir,
        tags={"model_family": "sequence", "model_type": "lstm", "phase": "train"},
    )
    mlflow.log_dict(config, "data_prep_config.json", run_id=run_id)
    LSTMPipeline.log_pipeline(
        model=inference.pipeline.model, preprocessor=inference.pipeline.preprocessor,
        feature_columns=data.feature_columns, run_id=run_id, registered_model_name="test_lstm",
        input_example={"X": data.X.numpy(), "lengths": data.lengths.numpy()},
    )
    yield SimpleNamespace(
        data_dir=str(data_dir), tracking_dir=tracking_dir, name="test_lstm", run_id=run_id,
        config=config, pipeline=inference.pipeline, inputs={"X": data.X.numpy(), "lengths": data.lengths.numpy()},
    )
    mlflow.set_tracking_uri(old_tracking_uri)


def test_inference_requires_pipeline_feature_columns(builder):
    with pytest.raises(ValueError, match="fitted feature columns"):
        SequenceModelInference(pipeline=LSTMPipeline(), data_builder=builder)


def test_predict_split_preserves_player_step_layout(inference, data):
    actual = inference.predict_split(data)

    np.testing.assert_array_equal(actual, np.zeros((2, 2), dtype=np.float32))
    assert actual.dtype == np.float32


def test_predict_split_rejects_feature_order_mismatch(inference, data):
    reordered = replace(data, feature_columns=["feature_b", "feature_a"])

    with pytest.raises(ValueError, match="Feature columns"):
        inference.predict_split(reordered)


def test_evaluate_is_unweighted_and_does_not_update_parameters(inference):
    before = [parameter.detach().clone() for parameter in inference.pipeline.model.parameters()]
    with patch.object(inference.pipeline.preprocessor, "fit", wraps=inference.pipeline.preprocessor.fit) as fit:
        actual = inference.evaluate()

    assert actual == pytest.approx(10.0)
    fit.assert_not_called()
    assert not inference.pipeline.model.training
    assert all(parameter.grad is None for parameter in inference.pipeline.model.parameters())
    for old, new in zip(before, inference.pipeline.model.parameters()):
        torch.testing.assert_close(old, new)


def test_evaluate_scores_steps_not_players(inference, builder, data):
    builder.get_data_splits.return_value["eval"] = replace(
        data, y=torch.tensor([[2.0, 0.0], [3.0, 4.0]]),
        loss_mask=torch.tensor([[True, False], [True, True]]),
    )

    assert inference.evaluate() == pytest.approx(29.0 / 3.0)


def test_evaluate_selects_test_split(inference):
    assert inference.evaluate(split="test") == pytest.approx(25.0)


@pytest.mark.parametrize("include_actual", [False, True])
def test_build_predictions_frame_uses_caller_mask_and_optional_actuals(inference, data, include_actual):
    predictions = np.array([[11.0, 0.0], [22.0, 33.0]], dtype=np.float32)
    step_mask = torch.tensor([[True, False], [True, False]])
    actual_values = torch.tensor([[7.0, 0.0], [8.0, 9.0]])

    actual = inference._build_predictions_frame(
        data, predictions, step_mask=step_mask,
        actual=actual_values if include_actual else None,
    )

    expected = pd.DataFrame({
        "player_id": ["A", "B"],
        "season": [2020, 2020], "target_season": [2022, 2021],
        "predictions": np.array([11.0, 22.0], dtype=np.float32),
    })
    if include_actual:
        expected["actual"] = np.array([7.0, 8.0], dtype=np.float32)
    pd.testing.assert_frame_equal(actual, expected)


def test_predict_exports_only_last_steps_with_identity(inference, data, tmp_path):
    # Metadata order is not tensor order; coordinates must drive alignment.
    inference.data_builder.get_data_splits.return_value["prediction"] = replace(
        data, identity=data.identity.iloc[[2, 0, 1]],
        y=torch.zeros_like(data.y), loss_mask=torch.zeros_like(data.loss_mask), sample_weight=None,
    )
    csv_path = str(tmp_path / "predictions.csv")

    actual = inference.predict(csv_path=csv_path)

    expected = pd.DataFrame({
        "player_id": ["B", "A"],
        "season": [2021, 2020], "target_season": [2022, 2022],
        "predictions": np.array([0.0, 0.0], dtype=np.float32),
    })
    pd.testing.assert_frame_equal(actual, expected)
    expected_csv = pd.DataFrame({
        "player_id": ["B", "A"], "season": [2021, 2020], "target_season": [2022, 2022],
        "ppr_points_per_game": [0.0, 0.0],
    })
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), expected_csv)


def test_evaluate_exports_only_scored_steps_and_actuals(inference, tmp_path):
    csv_path = str(tmp_path / "evaluation.csv")

    assert inference.evaluate(csv_path=csv_path) == pytest.approx(10.0)

    expected = pd.DataFrame({
        "player_id": ["A", "B"], "season": [2020, 2021], "target_season": [2022, 2022],
        "ppr_points_per_game": [0.0, 0.0], "actual": [2.0, 4.0],
    })
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), expected)


def test_csv_rounding_does_not_change_evaluation_metrics(inference, tmp_path):
    with torch.no_grad():
        inference.pipeline.model.head.bias.fill_(0.12345)
    csv_path = str(tmp_path / "evaluation.csv")

    actual = inference.evaluate(csv_path=csv_path)

    expected = float(np.mean((np.array([2.0, 4.0], dtype=np.float32) - np.float32(0.12345)) ** 2))
    assert actual == pytest.approx(expected)
    np.testing.assert_allclose(pd.read_csv(csv_path)["ppr_points_per_game"], [0.12, 0.12])


def test_evaluate_rejects_unscored_prediction_split(inference):
    with pytest.raises(ValueError, match="'prediction' split with no scored steps"):
        inference.evaluate(split="prediction")


@pytest.mark.parametrize("empty", [False, True], ids=["all-false-mask", "empty-tensors"])
@pytest.mark.parametrize("split", ["eval", "test"])
def test_evaluate_rejects_unscored_heldout_split(inference, builder, data, empty, split):
    if empty:
        unscored = replace(
            data, X=torch.empty((0, 0, 2)), lengths=torch.empty(0, dtype=torch.int64),
            y=torch.empty((0, 0)), loss_mask=torch.empty((0, 0), dtype=torch.bool), sample_weight=None,
        )
    else:
        unscored = replace(data, loss_mask=torch.zeros_like(data.loss_mask))
    builder.get_data_splits.return_value[split] = unscored

    with pytest.raises(ValueError, match="no scored steps"):
        inference.evaluate(split=split)


def test_evaluate_requires_targets_and_loss_mask(inference, builder, data):
    builder.get_data_splits.return_value["eval"] = replace(data, y=None, loss_mask=None, sample_weight=None)

    with pytest.raises(ValueError, match="Targets and a loss mask"):
        inference.evaluate()


def test_inference_does_not_fit_an_unfitted_preprocessor(inference):
    inference.pipeline.preprocessor = SequencePreprocessor()

    with pytest.raises(NotFittedError):
        inference.evaluate()


@pytest.mark.parametrize("model_output", [np.nan, np.inf, -np.inf])
def test_evaluate_rejects_nonfinite_model_outputs(inference, model_output):
    with patch.object(inference.pipeline, "predict", return_value=np.full((2, 2), model_output)):
        with pytest.raises(ValueError, match="Non-finite evaluation MSE"):
            inference.evaluate()


def test_evaluate_logs_metrics_at_requested_step(inference):
    with patch("src.modeling.sequence.inference.mlflow.log_metrics") as log_metrics:
        actual = inference.evaluate(split="test", run_id="evaluation-run", step=3)

    assert actual == pytest.approx(25.0)
    log_metrics.assert_called_once_with({"test_mse": 25.0, "test_rmse": 5.0}, step=3, run_id="evaluation-run")


def test_inference_accepts_mlflow_loaded_pipeline(inference, builder, data, tmp_path):
    old_tracking_uri = mlflow.get_tracking_uri()
    try:
        run_id = setup_mlflow_run("sequence_inference", "load", tracking_dir=str(tmp_path / "mlruns"))
        model_info = LSTMPipeline.log_pipeline(
            model=inference.pipeline.model, preprocessor=inference.pipeline.preprocessor,
            feature_columns=data.feature_columns, run_id=run_id,
            registered_model_name="test_lstm",
            input_example={"X": data.X.numpy(), "lengths": data.lengths.numpy()},
        )
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)
        loaded_inference = SequenceModelInference(pipeline=loaded, data_builder=builder)

        np.testing.assert_allclose(loaded_inference.predict_split(data), inference.predict_split(data))
        assert loaded_inference.evaluate() == pytest.approx(10.0)
        with pytest.raises(ValueError, match="Feature columns"):
            loaded_inference.predict_split(replace(data, feature_columns=["feature_b", "feature_a"]))
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_from_registry_reconstructs_saved_builder_and_version(registry):
    inference = SequenceModelInference.from_registry(
        data_dir=registry.data_dir, tracking_dir=registry.tracking_dir,
        registered_model=registry.name, model_version=1,
    )

    assert inference.model_version == 1
    assert inference.source_run_id == registry.run_id
    assert inference.data_builder.config == registry.config
    assert inference.data_builder.split_years.prediction_season == 2024
    assert inference.evaluate(split="eval") == pytest.approx(17.0)
    assert inference.evaluate(split="test") == pytest.approx(26.0)


def test_from_registry_defaults_to_latest_version(registry):
    run_id = setup_mlflow_run("saved_sequence_experiment", "train_v2", tracking_dir=registry.tracking_dir)
    mlflow.log_dict(registry.config, "data_prep_config.json", run_id=run_id)
    with torch.no_grad():
        registry.pipeline.model.head.bias.fill_(2.0)
    LSTMPipeline.log_pipeline(
        model=registry.pipeline.model, preprocessor=registry.pipeline.preprocessor,
        feature_columns=registry.pipeline.feature_columns, run_id=run_id,
        registered_model_name=registry.name, input_example=registry.inputs,
    )

    latest = SequenceModelInference.from_registry(registry.data_dir, registry.tracking_dir, registry.name)
    original = SequenceModelInference.from_registry(registry.data_dir, registry.tracking_dir, registry.name, model_version=1)

    assert latest.model_version == 2
    assert latest.source_run_id == run_id
    assert latest.evaluate(split="eval") == pytest.approx(5.0)
    assert original.evaluate(split="eval") == pytest.approx(17.0)


def test_from_registry_reports_missing_model(registry):
    with pytest.raises(ValueError, match="No versions found"):
        SequenceModelInference.from_registry(registry.data_dir, registry.tracking_dir, "missing_model")


def test_setup_mlflow_links_inference_run_to_source_experiment(registry):
    inference = SequenceModelInference.from_registry(registry.data_dir, registry.tracking_dir, registry.name)

    run_id = inference.setup_mlflow(phase="predict")

    run = mlflow.get_run(run_id)
    assert run.info.experiment_id == mlflow.get_run(registry.run_id).info.experiment_id
    assert run.data.tags["phase"] == "predict"
    assert run.data.tags["model_family"] == "sequence"
    assert run.data.tags["model_type"] == "lstm"
    assert run.data.params["source_run_id"] == registry.run_id
    assert run.data.params["model_version"] == "1"
    assert load_data_builder_config(run_id) == registry.config


def test_setup_mlflow_requires_registry_source(inference):
    with pytest.raises(ValueError, match="from_registry"):
        inference.setup_mlflow(phase="test")


@pytest.mark.parametrize("mode, expected_mse", [("eval", 17.0), ("test", 26.0), ("predict", None)])
def test_main_runs_registered_inference_and_exports_csv(registry, monkeypatch, mode, expected_mse):
    monkeypatch.setattr("sys.argv", [
        "sequence_inference", "--data-dir", registry.data_dir, "--tracking-dir", registry.tracking_dir,
        "--registered-model", registry.name, "--model-version", "1", "--mode", mode,
    ])

    main()

    runs = MlflowClient().search_runs(
        experiment_ids=[mlflow.get_run(registry.run_id).info.experiment_id],
        filter_string=f"tags.phase = '{mode}'",
    )
    assert len(runs) == 1
    run = runs[0]
    outputs = list((Path(registry.data_dir) / "predictions").glob("*.csv"))
    assert len(outputs) == 1
    frame = pd.read_csv(outputs[0])
    assert set(frame["player_id"]) == {"A", "B"}
    assert (frame["ppr_points_per_game"] == 0.0).all()
    assert outputs[0].name in {artifact.path for artifact in MlflowClient().list_artifacts(run.info.run_id)}
    if expected_mse is None:
        assert (frame["target_season"] == 2024).all()
        assert "actual" not in frame
        assert not run.data.metrics
    else:
        assert len(frame) == 2
        assert run.data.metrics[f"{mode}_mse"] == pytest.approx(expected_mse)
        assert run.data.metrics[f"{mode}_rmse"] == pytest.approx(expected_mse ** 0.5)
        assert "actual" in frame


@pytest.mark.parametrize("arguments", [[], ["--registered-model", "test_lstm"], ["--mode", "test"]])
def test_main_requires_model_and_mode(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["sequence_inference", *arguments])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
