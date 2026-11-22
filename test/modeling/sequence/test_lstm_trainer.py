from dataclasses import replace
from unittest.mock import Mock, patch

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.lstm import LSTMModel
from src.modeling.sequence.lstm_trainer import LSTMModelTrainer, main
from src.modeling.sequence.preprocessor import SequencePreprocessor
from src.modeling.utils import load_data_builder_config


def make_data():
    return SequenceDataset(
        X=torch.tensor([[[1.0], [float("nan")]], [[2.0], [3.0]]]),
        lengths=torch.tensor([1, 2]),
        identity=pd.DataFrame(),
        feature_columns=["feature"],
        y=torch.tensor([[2.0, float("nan")], [float("nan"), 4.0]]),
        loss_mask=torch.tensor([[True, False], [False, True]]),
        sample_weight=torch.tensor([[1.0, 0.0], [0.0, 3.0]]),
    )


def zero_model():
    model = LSTMModel(input_size=1, hidden_size=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def make_builder(train=None, eval_data=None):
    builder = Mock(spec=SequenceModelDataBuilder)
    builder.target = "ppr_points_per_game"
    builder.positions = None
    builder.config = {"target": builder.target, "split": {"prediction_season": 2026}}
    builder.get_data_splits.return_value = {
        "train": make_data() if train is None else train,
        "eval": make_data() if eval_data is None else eval_data,
        "test": make_data(),
    }
    return builder


def make_unscored_data(empty):
    data = make_data()
    if empty:
        return replace(
            data,
            X=torch.empty((0, 0, 1)),
            lengths=torch.empty(0, dtype=torch.int64),
            y=torch.empty((0, 0)),
            loss_mask=torch.empty((0, 0), dtype=torch.bool),
            sample_weight=torch.empty((0, 0)),
        )
    return replace(
        data,
        y=torch.zeros_like(data.y),
        loss_mask=torch.zeros_like(data.loss_mask),
        sample_weight=torch.zeros_like(data.sample_weight),
    )


def test_prepared_data_requires_fit_without_triggering_preprocessing():
    preprocessor = SequencePreprocessor()
    trainer = LSTMModelTrainer(zero_model(), make_builder(), preprocessor)

    with patch.object(preprocessor, "fit", wraps=preprocessor.fit) as fit:
        with patch.object(preprocessor, "transform", wraps=preprocessor.transform) as transform:
            with pytest.raises(ValueError, match="Call fit"):
                _ = trainer.prepared_data

    fit.assert_not_called()
    transform.assert_not_called()
    assert trainer._prepared_data is None


def test_prepared_data_property_is_read_only():
    trainer = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor())
    trainer.fit(epochs=1)
    prepared = trainer.prepared_data

    with pytest.raises(AttributeError):
        trainer.prepared_data = {}

    assert trainer.prepared_data is prepared


def test_mlflow_training_run_logs_config_params_and_metrics(tmp_path):
    builder = make_builder()
    trainer = LSTMModelTrainer(
        zero_model(), builder, SequencePreprocessor(), batch_size=2,
        tracking_dir=str(tmp_path / "mlruns"),
    )
    old_tracking_uri = mlflow.get_tracking_uri()
    try:
        run_id = trainer.setup_mlflow(extra_params={"seed": 42})
        history = trainer.fit(epochs=2, run_id=run_id)
        test_mse = trainer.evaluate(split="test", run_id=run_id)

        run = mlflow.get_run(run_id)
        assert mlflow.get_experiment(run.info.experiment_id).name == trainer.base_model_name
        assert run.data.tags["model_family"] == "sequence"
        assert run.data.tags["model_type"] == "lstm"
        assert run.data.tags["phase"] == "train"
        assert run.data.params["hidden_size"] == "2"
        assert run.data.params["num_layers"] == "1"
        assert run.data.params["imputation_strategy"] == "mean"
        assert load_data_builder_config(run_id) == builder.config
        assert run.data.metrics["test_mse"] == pytest.approx(test_mse)
        assert run.data.metrics["test_rmse"] == pytest.approx(test_mse ** 0.5)
        metric_history = MlflowClient().get_metric_history(run_id, "train_mse")
        assert [metric.step for metric in metric_history] == [1, 2]
        assert [metric.value for metric in metric_history] == pytest.approx([item["train_mse"] for item in history])
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)


def test_log_model_requires_training():
    trainer = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor())

    with pytest.raises(ValueError, match="Call fit"):
        trainer.log_model("unused-run")


def test_uses_supplied_preprocessor_for_training():
    preprocessor = SequencePreprocessor()
    trainer = LSTMModelTrainer(zero_model(), make_builder(), preprocessor)

    with patch.object(preprocessor, "fit", wraps=preprocessor.fit) as fit:
        with patch.object(preprocessor, "transform", wraps=preprocessor.transform) as transform:
            trainer.fit(epochs=1)

    assert trainer.preprocessor is preprocessor
    fit.assert_called_once_with(trainer.data["train"])
    assert transform.call_count == len(trainer.data)


def test_fit_updates_parameters_and_returns_epoch_history():
    builder = make_builder()
    trainer = LSTMModelTrainer(zero_model(), builder, SequencePreprocessor(), batch_size=2)
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    history = trainer.fit(epochs=2)

    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert all(key in history[0] for key in ["train_mse", "eval_mse"])
    assert history[0]["train_mse"] == pytest.approx((4.0 + 3.0 * 16.0) / 4.0)

    assert history[1]["epoch"] == 2
    assert all(key in history[1] for key in ["train_mse", "eval_mse"])
    assert history[1]["eval_mse"] < history[0]["eval_mse"]
    for metrics in history:
        assert metrics["train_rmse"] == pytest.approx(metrics["train_mse"] ** 0.5)
        assert metrics["eval_rmse"] == pytest.approx(metrics["eval_mse"] ** 0.5)
    assert any(not torch.equal(old, new) for old, new in zip(before, trainer.model.parameters()))


def test_fit_clears_preloaded_gradients_before_backward():
    clean = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor(), batch_size=2)
    preloaded = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor(), batch_size=2)
    for parameter in preloaded.model.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)

    # One batch per trainer: stale gradients must not affect its gradients or update.
    clean.fit(epochs=1, evaluate_each_epoch=False)
    preloaded.fit(epochs=1, evaluate_each_epoch=False)

    for expected, actual in zip(clean.model.parameters(), preloaded.model.parameters()):
        assert expected.grad is not None
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad)
        torch.testing.assert_close(actual, expected)


def test_evaluate_is_unweighted_and_does_not_update_parameters():
    trainer = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor(), batch_size=1, learning_rate=0.0)
    trainer.fit(epochs=1, evaluate_each_epoch=False)
    trainer.optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    actual = trainer.evaluate()

    assert actual == pytest.approx(10.0)
    assert not trainer.model.training
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    for old, new in zip(before, trainer.model.parameters()):
        torch.testing.assert_close(old, new)


def test_epoch_loss_aggregates_scored_steps_not_batch_means():
    data = replace(
        make_data(),
        y=torch.tensor([[2.0, 0.0], [3.0, 4.0]]),
        loss_mask=torch.tensor([[True, False], [True, True]]),
    )
    trainer = LSTMModelTrainer(zero_model(), make_builder(eval_data=data), SequencePreprocessor(), batch_size=1, learning_rate=0.0)
    trainer.fit(epochs=1, evaluate_each_epoch=False)

    assert trainer.evaluate() == pytest.approx(29.0 / 3.0)


def test_zero_weight_batches_are_skipped():
    data = replace(make_data(), sample_weight=torch.tensor([[0.0, 0.0], [0.0, 3.0]]))
    trainer = LSTMModelTrainer(zero_model(), make_builder(train=data), SequencePreprocessor(), batch_size=1)

    history = trainer.fit(epochs=1)

    assert history[0]["train_mse"] == pytest.approx(16.0)


@pytest.mark.parametrize("sample_weights, expected_gradients", [
    ([0.1, 1.0], [-4.0 * 0.1 / 0.55, -8.0 / 0.55]),
    ([1.0, 1.0], [-4.0, -8.0]),
    ([1.0, 10.0], [-4.0 / 5.5, -8.0 * 10.0 / 5.5]),
])
def test_single_player_batches_preserve_relative_sample_weights(sample_weights, expected_gradients):
    data = replace(make_data(), sample_weight=torch.tensor([[sample_weights[0], 0.0], [0.0, sample_weights[1]]]))
    trainer = LSTMModelTrainer(zero_model(), make_builder(train=data), SequencePreprocessor(), batch_size=1)
    gradients = []

    # Record backward gradients without changing parameters, isolating the loss
    # from Adam's adaptive update scaling. Shuffling may swap the two batches.
    with patch.object(trainer.optimizer, "step", side_effect=lambda: gradients.append(trainer.model.head.bias.grad.item())):
        trainer.fit(epochs=1, evaluate_each_epoch=False)

    assert sorted(gradients) == pytest.approx(sorted(expected_gradients))


@pytest.mark.parametrize("weighted", [True, False])
def test_training_normalization_handles_unequal_scored_lengths_and_final_batch(weighted):
    data = SequenceDataset(
        X=torch.tensor([[[1.0], [float("nan")]], [[2.0], [3.0]], [[4.0], [float("nan")]]]),
        lengths=torch.tensor([1, 2, 1]),
        identity=pd.DataFrame(),
        feature_columns=["feature"],
        y=torch.tensor([[2.0, float("nan")], [3.0, 4.0], [5.0, float("nan")]]),
        loss_mask=torch.tensor([[True, False], [True, True], [True, False]]),
        sample_weight=torch.tensor([[0.1, 0.0], [1.0, 2.0], [0.4, 0.0]]) if weighted else None,
    )
    trainer = LSTMModelTrainer(zero_model(), make_builder(train=data), SequencePreprocessor(), batch_size=2)
    trainer._fit_preprocessing()
    prepared = trainer.prepared_data["train"]
    tensors = trainer._make_loader(prepared, training=True).dataset
    # Fix player order: first batch has 2 players/3 scored steps, final batch 1/1.
    loader = DataLoader(tensors, batch_size=2, shuffle=False)
    gradients = []
    with patch.object(trainer.optimizer, "step", side_effect=lambda: gradients.append(trainer.model.head.bias.grad.item())):
        reported_mse = trainer._run_epoch(loader, training=True)

    expected_gradients = [-9.6, -24.0 / 7.0] if weighted else [-6.75, -7.5]
    assert gradients == pytest.approx(expected_gradients)
    # Weight batch gradients by their player counts to recover the full-split
    # weighted objective's gradient at these unchanged parameters.
    expected_global_gradient = -26.4 / 3.5 if weighted else -7.0
    assert (2.0 * gradients[0] + gradients[1]) / 3.0 == pytest.approx(expected_global_gradient)
    assert reported_mse == pytest.approx(51.4 / 3.5 if weighted else 13.5)


def test_feature_count_must_match_model_input_size():
    trainer = LSTMModelTrainer(LSTMModel(input_size=2), make_builder(), SequencePreprocessor())

    with pytest.raises(ValueError, match="input_size"):
        trainer.fit()


def test_repeated_fit_reuses_optimizer_state():
    trainer = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor())
    trainer.fit(epochs=1)
    optimizer = trainer.optimizer
    prepared = trainer.prepared_data
    step_before = optimizer.state[trainer.model.head.bias]["step"].item()

    with patch.object(trainer.preprocessor, "fit", wraps=trainer.preprocessor.fit) as fit_preprocessor:
        trainer.fit(epochs=1, evaluate_each_epoch=False)
        fit_preprocessor.assert_not_called()

    assert trainer.optimizer is optimizer
    assert trainer.prepared_data is prepared
    assert optimizer.state[trainer.model.head.bias]["step"].item() > step_before


def test_evaluate_selects_test_split():
    builder = make_builder()
    builder.get_data_splits.return_value["test"] = replace(make_data(), y=torch.full((2, 2), 5.0))
    trainer = LSTMModelTrainer(zero_model(), builder, SequencePreprocessor(), learning_rate=0.0)
    trainer.fit(epochs=1, evaluate_each_epoch=False)

    assert trainer.evaluate(split="test") == pytest.approx(25.0)
    with pytest.raises(ValueError, match="Evaluation split"):
        trainer.evaluate(split="prediction")


@pytest.mark.parametrize("empty", [False, True], ids=["all-false-mask", "empty-tensors"])
def test_fit_rejects_unscored_eval_split_when_evaluation_requested(empty):
    eval_data = make_unscored_data(empty)
    trainer = LSTMModelTrainer(zero_model(), make_builder(eval_data=eval_data), SequencePreprocessor())

    with pytest.raises(ValueError, match="no scored steps"):
        trainer.fit(epochs=1)


@pytest.mark.parametrize("empty", [False, True], ids=["all-false-mask", "empty-tensors"])
def test_fit_allows_unscored_eval_split_when_evaluation_disabled(empty):
    trainer = LSTMModelTrainer(zero_model(), make_builder(eval_data=make_unscored_data(empty)), SequencePreprocessor())

    history = trainer.fit(epochs=1, evaluate_each_epoch=False)

    assert "eval_mse" not in history[0]
    assert "eval_rmse" not in history[0]
    assert history[0]["train_rmse"] == pytest.approx(history[0]["train_mse"] ** 0.5)


@pytest.mark.parametrize("empty", [False, True], ids=["all-false-mask", "empty-tensors"])
@pytest.mark.parametrize("split", ["eval", "test"])
def test_evaluate_rejects_unscored_heldout_split(empty, split):
    builder = make_builder()
    builder.get_data_splits.return_value[split] = make_unscored_data(empty)
    trainer = LSTMModelTrainer(zero_model(), builder, SequencePreprocessor())
    trainer.fit(epochs=1, evaluate_each_epoch=False)

    with pytest.raises(ValueError, match="no scored steps"):
        trainer.evaluate(split=split)


def test_evaluate_requires_fitted_preprocessing():
    trainer = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor())

    with pytest.raises(ValueError, match="Call fit"):
        trainer.evaluate()


def test_preprocessing_fits_on_real_training_history_and_preserves_raw_data():
    train = make_data()
    eval_data = make_data()
    eval_data.X[1, 0, 0] = float("nan")
    eval_data.X[1, 1, 0] = 100.0
    builder = make_builder(train=train, eval_data=eval_data)
    prediction = replace(
        make_data(),
        X=torch.empty((0, 0, 1)),
        lengths=torch.empty(0, dtype=torch.int64),
        y=torch.empty((0, 0)),
        loss_mask=torch.empty((0, 0), dtype=torch.bool),
        sample_weight=None,
    )
    builder.get_data_splits.return_value["prediction"] = prediction
    trainer = LSTMModelTrainer(zero_model(), builder, SequencePreprocessor())
    raw_X = {split: data.X.clone() for split, data in trainer.data.items()}
    assert not hasattr(trainer.preprocessor.preprocessor.named_steps["scaler"], "mean_")

    trainer.fit(epochs=1)

    # Real training steps are [1, 2, 3]; the unscored context step 2 is included.
    scaler = trainer.preprocessor.preprocessor.named_steps["scaler"]
    imputer = trainer.preprocessor.preprocessor.named_steps["imputer"]
    assert imputer.statistics_[0] == pytest.approx(2.0)
    assert scaler.mean_[0] == pytest.approx(2.0)
    assert scaler.scale_[0] == pytest.approx((2.0 / 3.0) ** 0.5)
    expected_train = torch.tensor([
        [[-1.0 / scaler.scale_[0]], [float("nan")]],
        [[0.0], [1.0 / scaler.scale_[0]]],
    ], dtype=torch.float32)
    torch.testing.assert_close(trainer.prepared_data["train"].X, expected_train, equal_nan=True)
    expected_eval = torch.tensor([
        [[-1.0 / scaler.scale_[0]], [float("nan")]],
        [[0.0], [98.0 / scaler.scale_[0]]],
    ], dtype=torch.float32)
    torch.testing.assert_close(trainer.prepared_data["eval"].X, expected_eval, equal_nan=True)
    for split, raw in trainer.data.items():
        prepared = trainer.prepared_data[split]
        torch.testing.assert_close(raw.X, raw_X[split], equal_nan=True)
        for field in ("identity", "lengths", "y", "loss_mask", "sample_weight", "feature_columns"):
            assert getattr(prepared, field) is getattr(raw, field)
        torch.testing.assert_close(prepared.X[~raw.real_step_mask], raw.X[~raw.real_step_mask], equal_nan=True)


def test_missing_training_features_are_imputed_before_training():
    data = make_data()
    data.X[1, 0, 0] = float("nan")
    trainer = LSTMModelTrainer(zero_model(), make_builder(train=data), SequencePreprocessor())

    trainer.fit(epochs=1)

    assert trainer.preprocessor.preprocessor.named_steps["imputer"].statistics_[0] == pytest.approx(2.0)
    prepared = trainer.prepared_data["train"]
    assert prepared.X[1, 0, 0] == 0.0
    assert torch.isfinite(prepared.X[prepared.real_step_mask]).all()


def test_entirely_missing_training_features_keep_their_columns():
    train = make_data()
    train.X[:] = float("nan")
    trainer = LSTMModelTrainer(zero_model(), make_builder(train=train), SequencePreprocessor())

    trainer.fit(epochs=1)

    prepared = trainer.prepared_data["train"]
    assert prepared.X.shape == train.X.shape
    torch.testing.assert_close(prepared.X[prepared.real_step_mask], torch.zeros(3, 1))


def test_prepared_features_are_validated_before_training():
    trainer = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor())
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    with patch.object(trainer.preprocessor.preprocessor, "transform", return_value=[[float("nan")]] * 3):
        with pytest.raises(ValueError, match="finite after preprocessing"):
            trainer.fit(epochs=1)

    assert trainer._prepared_data is None
    with pytest.raises(ValueError, match="Call fit"):
        _ = trainer.prepared_data
    for old, new in zip(before, trainer.model.parameters()):
        torch.testing.assert_close(old, new)


@pytest.mark.parametrize("model_output", [float("nan"), float("inf"), -float("inf")])
def test_fit_rejects_nonfinite_model_output_loss(model_output):
    trainer = LSTMModelTrainer(zero_model(), make_builder(), SequencePreprocessor(), batch_size=2)
    model_outputs = torch.full((2, 2), model_output)

    with patch.object(trainer.model, "forward", return_value=model_outputs):
        with pytest.raises(ValueError, match="Non-finite loss encountered"):
            trainer.fit(epochs=1, evaluate_each_epoch=False)


def test_main_uses_default_params_and_infers_input_size(monkeypatch):
    builder = make_builder()
    monkeypatch.setattr("sys.argv", ["lstm_trainer", "--config", "sequence.yaml"])
    with patch.object(SequenceModelDataBuilder, "from_config_file", return_value=builder) as from_config:
        with patch("src.modeling.sequence.lstm_trainer.LSTMModelTrainer") as trainer_class:
            trainer_class.return_value.setup_mlflow.return_value = "training-run"
            main()

    from_config.assert_called_once_with(data_dir="data", config_path="sequence.yaml")
    kwargs = trainer_class.call_args.kwargs
    model = kwargs["model"]
    assert model.lstm.input_size == 1
    assert model.lstm.hidden_size == 64
    assert model.lstm.num_layers == 1
    assert model.lstm.dropout == 0.0
    assert kwargs["data_builder"] is builder
    assert isinstance(kwargs["preprocessor"], SequencePreprocessor)
    assert kwargs["preprocessor"].feature_columns is None
    assert kwargs["batch_size"] == 32
    assert kwargs["learning_rate"] == 1e-3
    assert kwargs["device"] == "cpu"
    assert kwargs["tracking_dir"] == "mlruns"
    trainer_class.return_value.setup_mlflow.assert_called_once_with(
        extra_params={"config_path": "sequence.yaml", "seed": 42, "epochs": 20}
    )
    trainer_class.return_value.fit.assert_called_once_with(epochs=20, run_id="training-run")
    trainer_class.return_value.evaluate.assert_not_called()


def test_main_passes_custom_params_and_evaluates_test_when_requested(monkeypatch):
    builder = make_builder()
    monkeypatch.setattr("sys.argv", [
        "lstm_trainer", "--config", "sequence.yaml", "--data-dir", "custom-data",
        "--hidden-size", "8", "--num-layers", "2", "--dropout", "0.2",
        "--epochs", "3", "--batch-size", "4", "--learning-rate", "0.01",
        "--device", "mps", "--evaluate-test", "--tracking-dir", "custom-mlruns",
    ])
    with patch.object(SequenceModelDataBuilder, "from_config_file", return_value=builder) as from_config:
        with patch("src.modeling.sequence.lstm_trainer.LSTMModelTrainer") as trainer_class:
            trainer_class.return_value.setup_mlflow.return_value = "training-run"
            trainer_class.return_value.evaluate.return_value = 4.0
            with patch("src.modeling.sequence.lstm_trainer.torch.manual_seed") as seed:
                main()

    from_config.assert_called_once_with(data_dir="custom-data", config_path="sequence.yaml")
    seed.assert_called_once_with(42)
    kwargs = trainer_class.call_args.kwargs
    assert kwargs["model"].lstm.hidden_size == 8
    assert kwargs["model"].lstm.num_layers == 2
    assert kwargs["model"].lstm.dropout == 0.2
    assert kwargs["batch_size"] == 4
    assert kwargs["learning_rate"] == 0.01
    assert kwargs["device"] == "mps"
    assert kwargs["tracking_dir"] == "custom-mlruns"
    trainer_class.return_value.fit.assert_called_once_with(epochs=3, run_id="training-run")
    trainer_class.return_value.evaluate.assert_called_once_with(split="test", run_id="training-run")


def test_main_requires_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["lstm_trainer"])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2


def test_main_trains_and_evaluates_with_synthetic_data(monkeypatch, caplog):
    monkeypatch.setattr("sys.argv", [
        "lstm_trainer", "--config", "sequence.yaml", "--epochs", "1",
        "--hidden-size", "2", "--evaluate-test",
    ])
    with patch.object(SequenceModelDataBuilder, "from_config_file", return_value=make_builder()):
        # MLflow persistence is exercised by the separate round-trip tests.
        with patch.object(LSTMModelTrainer, "setup_mlflow", return_value=None):
            with caplog.at_level("INFO", logger="src.modeling.sequence.lstm_trainer"):
                main()

    assert "LSTM epoch metrics" in caplog.text
    assert "LSTM test MSE" in caplog.text
