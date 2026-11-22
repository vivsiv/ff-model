"""Shared preprocessing for sequence-model training and inference."""

from dataclasses import replace

import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.modeling.sequence.data_builder import SequenceDataset

# TODO: use an nn.Module for all this, allowing us to use mlflows pytorch flavor for logging
class SequencePreprocessor:
    """Mean-impute and scale real steps while preserving the sequence layout.

    Fit on the training dataset only, including unscored historical context and
    excluding padding. Reuse the fitted state for every other split. Entirely
    missing training features are filled with zero without dropping columns.
    Targets and sample weights are not transformed.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        self.feature_columns: list[str] | None = None

    def fit(self, data: SequenceDataset) -> "SequencePreprocessor":
        """Learn feature statistics from real training steps, without changing data."""
        inputs = data.X[data.real_step_mask].detach().cpu().numpy()
        if len(inputs) == 0:
            raise ValueError("Real training steps are required to fit preprocessing.")
        self.pipeline.fit(inputs)
        self.feature_columns = list(data.feature_columns)
        return self

    def transform(self, data: SequenceDataset) -> SequenceDataset:
        """Return a prepared copy, preserving padding and all non-feature fields."""
        check_is_fitted(self.pipeline)
        if data.feature_columns != self.feature_columns:
            raise ValueError("Feature columns must match fitted preprocessing in order.")
        real_mask = data.real_step_mask
        X = data.X.clone()
        if real_mask.any():
            inputs = data.X[real_mask].detach().cpu().numpy()
            transformed = self.pipeline.transform(inputs)
            X[real_mask] = torch.as_tensor(transformed, dtype=X.dtype, device=X.device)
        if not torch.isfinite(X[real_mask]).all():
            raise ValueError("Real input steps must be finite after preprocessing.")
        return replace(data, X=X)

    @classmethod
    def from_fitted_pipeline(cls, pipeline: Pipeline, feature_columns: list[str]) -> "SequencePreprocessor":
        """Wrap a fitted sklearn pipeline from an older saved sequence model."""
        check_is_fitted(pipeline)
        preprocessor = cls()
        preprocessor.pipeline = pipeline
        preprocessor.feature_columns = list(feature_columns)
        return preprocessor
