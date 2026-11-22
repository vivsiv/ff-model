import pickle
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from src.modeling.sequence.data_builder import SequenceDataset
from src.modeling.sequence.preprocessor import SequencePreprocessor


@pytest.fixture
def data():
    nan = float("nan")
    return SequenceDataset(
        X=torch.tensor([
            [[1.0, 10.0], [nan, nan], [nan, nan]],
            [[2.0, nan], [4.0, 20.0], [nan, nan]],
        ]),
        lengths=torch.tensor([1, 2]),
        identity=pd.DataFrame({"player_id": ["A", "B", "B"], "step_idx": [0, 0, 1]}),
        feature_columns=["a", "b"],
        y=torch.tensor([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
        loss_mask=torch.tensor([[True, False, False], [False, True, False]]),
        sample_weight=torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
    )


def test_fit_uses_all_real_training_steps(data):
    preprocessor = SequencePreprocessor()

    assert preprocessor.fit(data) is preprocessor

    # Feature A includes Player 2, Step 0's value of 2 despite it not contributing to 2.
    # Feature b's missing value is imputed to 15 before fitting the scaler.
    # Padding contributes to neither.
    np.testing.assert_allclose(preprocessor.preprocessor.named_steps["imputer"].statistics_, [7.0 / 3.0, 15.0])
    np.testing.assert_allclose(preprocessor.preprocessor.named_steps["scaler"].mean_, [7.0 / 3.0, 15.0])
    np.testing.assert_allclose(preprocessor.preprocessor.named_steps["scaler"].scale_, [14.0 ** 0.5 / 3.0, (50.0 / 3.0) ** 0.5])
    assert preprocessor.feature_columns == data.feature_columns
    assert preprocessor.feature_columns is not data.feature_columns


def test_transform_preserves_layout_padding_metadata_and_leaves_input_unchanges(data):
    preprocessor = SequencePreprocessor().fit(data)
    before = data.X.clone()

    actual = preprocessor.transform(data)

    scale_a = 14.0 ** 0.5 / 3.0
    scale_b = (50.0 / 3.0) ** 0.5
    nan = float("nan")
    expected = torch.tensor([
        [[(1.0 - 7.0 / 3.0) / scale_a, -5.0 / scale_b], [nan, nan], [nan, nan]],
        [[(2.0 - 7.0 / 3.0) / scale_a, 0.0], [(4.0 - 7.0 / 3.0) / scale_a, 5.0 / scale_b], [nan, nan]],
    ])
    torch.testing.assert_close(actual.X, expected, equal_nan=True)
    torch.testing.assert_close(data.X, before, equal_nan=True)
    assert actual is not data
    assert actual.X is not data.X
    for field in ("lengths", "identity", "feature_columns", "y", "loss_mask", "sample_weight"):
        assert getattr(actual, field) is getattr(data, field)


def test_transform_reuses_training_statistics(data):
    preprocessor = SequencePreprocessor().fit(data)
    heldout = replace(data, X=data.X.clone())
    heldout.X[0, 0] = torch.tensor([100.0, float("nan")])

    actual = preprocessor.transform(heldout)

    scale_a = 14.0 ** 0.5 / 3.0
    torch.testing.assert_close(actual.X[0, 0], torch.tensor([(100.0 - 7.0 / 3.0) / scale_a, 0.0]))
    np.testing.assert_allclose(preprocessor.preprocessor.named_steps["imputer"].statistics_, [7.0 / 3.0, 15.0])


def test_transform_requires_fit(data):
    with pytest.raises(NotFittedError):
        SequencePreprocessor().transform(data)


def test_transform_rejects_reordered_features(data):
    preprocessor = SequencePreprocessor().fit(data)

    with pytest.raises(ValueError, match="Feature columns"):
        preprocessor.transform(replace(data, feature_columns=["b", "a"]))


def empty_data(data):
    return replace(
        data,
        X=torch.empty((0, 0, 2)), lengths=torch.empty(0, dtype=torch.int64),
        identity=data.identity.iloc[:0], y=torch.empty((0, 0)),
        loss_mask=torch.empty((0, 0), dtype=torch.bool), sample_weight=None,
    )


def test_fit_rejects_empty_data(data):
    with pytest.raises(ValueError, match="Real training steps"):
        SequencePreprocessor().fit(empty_data(data))


def test_transform_handles_empty_data_without_calling_sklearn_transform(data):
    preprocessor = SequencePreprocessor().fit(data)
    with patch.object(preprocessor.preprocessor, "transform") as transform:
        actual = preprocessor.transform(empty_data(data))

    transform.assert_not_called()
    assert actual.X.shape == (0, 0, 2)


def test_fit_keeps_entirely_missing_features(data):
    data.X[:, :, 1] = float("nan")
    preprocessor = SequencePreprocessor().fit(data)

    actual = preprocessor.transform(data)

    assert actual.X.shape == data.X.shape
    torch.testing.assert_close(actual.X[actual.real_step_mask][:, 1], torch.zeros(3))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_transform_rejects_nonfinite_prepared_features(data, value):
    preprocessor = SequencePreprocessor().fit(data)
    with patch.object(preprocessor.preprocessor, "transform", return_value=np.full((3, 2), value)):
        with pytest.raises(ValueError, match="finite after preprocessing"):
            preprocessor.transform(data)


def test_pickle_preserves_fitted_preprocessing(data):
    preprocessor = SequencePreprocessor().fit(data)

    restored = pickle.loads(pickle.dumps(preprocessor))

    assert restored.feature_columns == preprocessor.feature_columns
    torch.testing.assert_close(restored.transform(data).X, preprocessor.transform(data).X, equal_nan=True)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_transform_preserves_tensor_device(data, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    preprocessor = SequencePreprocessor().fit(data)
    expected = preprocessor.transform(data)

    actual = preprocessor.transform(replace(data, X=data.X.to(device)))

    assert actual.X.device.type == device
    torch.testing.assert_close(actual.X.cpu(), expected.X, equal_nan=True)
