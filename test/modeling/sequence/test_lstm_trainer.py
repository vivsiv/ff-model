from dataclasses import replace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.lstm import LSTMModel
from src.modeling.sequence.lstm_trainer import LSTMModelTrainer


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
    builder.get_data_splits.return_value = {
        "train": make_data() if train is None else train,
        "eval": make_data() if eval_data is None else eval_data,
        "test": make_data(),
    }
    return builder


def test_fit_updates_parameters_and_returns_epoch_history():
    builder = make_builder()
    trainer = LSTMModelTrainer(zero_model(), builder, batch_size=2)
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    history = trainer.fit(epochs=2)

    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert all(key in history[0] for key in ["train_mse", "eval_mse"])
    assert history[0]["train_mse"] == pytest.approx((4.0 + 3.0 * 16.0) / 4.0)

    assert history[1]["epoch"] == 2
    assert all(key in history[1] for key in ["train_mse", "eval_mse"])
    assert history[1]["eval_mse"] < history[0]["eval_mse"]
    assert any(not torch.equal(old, new) for old, new in zip(before, trainer.model.parameters()))


def test_fit_clears_preloaded_gradients_before_backward():
    clean = LSTMModelTrainer(zero_model(), make_builder(), batch_size=2)
    preloaded = LSTMModelTrainer(zero_model(), make_builder(), batch_size=2)
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
    trainer = LSTMModelTrainer(zero_model(), make_builder(), batch_size=1)
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
    trainer = LSTMModelTrainer(zero_model(), make_builder(eval_data=data), batch_size=1)

    assert trainer.evaluate() == pytest.approx(29.0 / 3.0)


def test_zero_weight_batches_are_skipped():
    data = replace(make_data(), sample_weight=torch.tensor([[0.0, 0.0], [0.0, 3.0]]))
    trainer = LSTMModelTrainer(zero_model(), make_builder(train=data), batch_size=1)

    history = trainer.fit(epochs=1)

    assert history[0]["train_mse"] == pytest.approx(16.0)


def test_feature_count_must_match_model_input_size():
    trainer = LSTMModelTrainer(LSTMModel(input_size=2), make_builder())

    with pytest.raises(ValueError, match="input_size"):
        trainer.fit()


def test_repeated_fit_reuses_optimizer_state():
    trainer = LSTMModelTrainer(zero_model(), make_builder())
    trainer.fit(epochs=1)
    optimizer = trainer.optimizer
    step_before = optimizer.state[trainer.model.head.bias]["step"].item()

    trainer.fit(epochs=1, evaluate_each_epoch=False)

    assert trainer.optimizer is optimizer
    assert optimizer.state[trainer.model.head.bias]["step"].item() > step_before


def test_evaluate_selects_test_split():
    builder = make_builder()
    builder.get_data_splits.return_value["test"] = replace(make_data(), y=torch.full((2, 2), 5.0))
    trainer = LSTMModelTrainer(zero_model(), builder)

    assert trainer.evaluate(split="test") == pytest.approx(25.0)
    with pytest.raises(ValueError, match="Evaluation split"):
        trainer.evaluate(split="prediction")


def test_fit_skips_empty_eval_split():
    eval_data = replace(make_data(), loss_mask=torch.zeros(2, 2, dtype=torch.bool))
    trainer = LSTMModelTrainer(zero_model(), make_builder(eval_data=eval_data))

    history = trainer.fit(epochs=1)

    assert "eval_mse" not in history[0]


def test_evaluate_rejects_empty_eval_split():
    eval_data = replace(make_data(), loss_mask=torch.zeros(2, 2, dtype=torch.bool))
    trainer = LSTMModelTrainer(zero_model(), make_builder(eval_data=eval_data))

    with pytest.raises(ValueError, match="no scored steps"):
        trainer.evaluate()
