import torch
from torch import nn

from src.modeling.sequence.lstm import LSTMModel


def test_forward_skips_nan_padding_and_preserves_sequence_order():
    model = LSTMModel(input_size=2, hidden_size=4)
    X = torch.tensor([
        [[1.0, 2.0], [float("nan"), float("nan")], [float("nan"), float("nan")]],
        [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    ])
    lengths = torch.tensor([1, 3])

    actual = model(X, lengths)
    expected = torch.zeros(2, 3)
    for idx, length in enumerate(lengths.tolist()):
        output, _ = model.lstm(X[idx:idx + 1, :length])
        expected[idx, :length] = model.head(output).squeeze(-1).squeeze(0)

    torch.testing.assert_close(actual, expected)


def test_forward_does_not_read_future_steps():
    model = LSTMModel(input_size=2, hidden_size=4)
    X = torch.randn(1, 3, 2)
    changed = X.clone()
    changed[:, 2] += 100.0

    actual = model(X, torch.tensor([3]))
    modified = model(changed, torch.tensor([3]))

    torch.testing.assert_close(actual[:, :2], modified[:, :2])


def test_backward_with_nan_padding_has_finite_gradients():
    model = LSTMModel(input_size=2, hidden_size=4)
    X = torch.tensor([
        [[1.0, 2.0], [float("nan"), float("nan")]],
        [[3.0, 4.0], [5.0, 6.0]],
    ])
    loss_mask = torch.tensor([[True, False], [True, True]])

    predictions = model(X, torch.tensor([1, 2]))
    predictions[loss_mask].square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_inits_with_correct_params():
    single_layer = LSTMModel(input_size=2, hidden_size=4, dropout=0.2)
    stacked = LSTMModel(input_size=2, hidden_size=4, num_layers=2, dropout=0.2)

    for model in (single_layer, stacked):
        assert not model.lstm.bidirectional

        assert isinstance(model.lstm, nn.LSTM)
        assert model.lstm.input_size == 2
        assert model.lstm.hidden_size == 4
        assert model.lstm.batch_first

        assert isinstance(model.head, nn.Linear)
        assert model.head.in_features == 4
        assert model.head.out_features == 1

    assert single_layer.lstm.num_layers == 1
    assert single_layer.lstm.dropout == 0.0

    assert stacked.lstm.num_layers == 2
    assert stacked.lstm.dropout == 0.2
