"""MLflow-loadable pipeline for raw padded player histories."""

import json
import pickle
from dataclasses import replace

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from src.modeling.sequence.data_builder import SequenceDataset
from src.modeling.sequence.lstm import LSTMModel


def transform_sequence_dataset(data: SequenceDataset, preprocessor: Pipeline) -> SequenceDataset:
    """Transform real feature rows without changing padding or dataset metadata."""
    real_mask = data.real_step_mask
    X = data.X.clone()
    if real_mask.any():
        inputs = data.X[real_mask].detach().cpu().numpy()
        transformed = preprocessor.transform(inputs)
        X[real_mask] = torch.as_tensor(transformed, dtype=X.dtype, device=X.device)
    if not torch.isfinite(X[real_mask]).all():
        raise ValueError("Real input steps must be finite after preprocessing.")
    return replace(data, X=X)


class LSTMPipeline(mlflow.pyfunc.PythonModel):
    """Apply fitted sklearn preprocessing and a trained LSTM in one prediction call.

    MLflow predict accepts a dictionary with float32 X [N, L, F] and int64 lengths
    [N] NumPy arrays. Features must follow the saved feature_columns order. Outputs
    are float32 [N, L] predictions; padded positions are zero. Loaded models run on
    CPU for portability, regardless of the device used during training.
    """

    def __init__(self, model=None, preprocessor=None, feature_columns=None):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns

    def load_context(self, context):
        """Restore architecture, CPU weights, feature order, and fitted preprocessing."""
        with open(context.artifacts["metadata"]) as file:
            metadata = json.load(file)
        self.feature_columns = metadata["feature_columns"]
        self.model = LSTMModel(**metadata["model_params"])
        self.model.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu", weights_only=True))
        self.model.eval()
        # Like other MLflow models, only load artifacts from trusted sources.
        with open(context.artifacts["preprocessor"], "rb") as file:
            self.preprocessor = pickle.load(file)

    def predict(self, context, model_input, params=None):
        """Predict from raw arrays without fitting preprocessing or using targets."""
        X = torch.as_tensor(model_input["X"], dtype=torch.float32)
        lengths = torch.as_tensor(model_input["lengths"])
        if X.ndim != 3 or X.shape[-1] != self.model.lstm.input_size:
            raise ValueError("X must have shape [N, L, model input_size].")
        if lengths.shape != (X.shape[0],) or lengths.dtype != torch.int64:
            raise ValueError("lengths must be an int64 tensor of shape [N].")
        if ((lengths < 1) | (lengths > X.shape[1])).any():
            raise ValueError("Sequence lengths must be between 1 and L.")
        if X.shape[0] == 0:
            return np.zeros(X.shape[:2], dtype=np.float32)

        data = SequenceDataset(X=X, lengths=lengths, identity=pd.DataFrame(), feature_columns=self.feature_columns)
        prepared = transform_sequence_dataset(data, self.preprocessor)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(prepared.X.to(device), lengths)
        return predictions.cpu().numpy()

    def predict_dataset(self, data: SequenceDataset) -> np.ndarray:
        """Predict a raw SequenceDataset, verifying its feature names and order."""
        if data.feature_columns != self.feature_columns:
            raise ValueError("Feature columns must match the saved pipeline in order.")
        return self.predict(None, {
            "X": data.X.detach().cpu().numpy(),
            "lengths": data.lengths.cpu().numpy(),
        })
