"""MLflow-loadable pipeline for raw padded player histories."""

import json
import pickle
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow.pyfunc
from mlflow.pyfunc import PythonModelContext
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.types.schema import Schema, TensorSpec
import numpy as np
import torch

from src.modeling.sequence.data_builder import SequenceDataset
from src.modeling.sequence.lstm import LSTMModel
from src.modeling.sequence.preprocessor import SequencePreprocessor


class LSTMPipeline(mlflow.pyfunc.PythonModel):
    """Apply fitted sklearn preprocessing and a trained LSTM in one prediction call.

    MLflow predict accepts a dictionary with float32 X [N, L, F] and int64 lengths
    [N] NumPy arrays. Features must follow the saved feature_columns order. Outputs
    are float32 [N, L] predictions; padded positions are zero. Loaded models run on
    CPU for portability, regardless of the device used during training.
    """

    def __init__(
        self,
        model: LSTMModel | None = None,
        preprocessor: SequencePreprocessor | None = None,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns

    @staticmethod
    def model_params(model: LSTMModel) -> dict[str, int | float]:
        """Return architecture arguments used both for tracking and reconstruction."""
        return {
            "input_size": model.lstm.input_size,
            "hidden_size": model.lstm.hidden_size,
            "num_layers": model.lstm.num_layers,
            "dropout": model.lstm.dropout,
        }

    @staticmethod
    def _save_artifacts(
        directory: str | Path,
        model: LSTMModel,
        preprocessor: SequencePreprocessor,
        feature_columns: list[str],
    ) -> dict[str, str]:
        """Stage fitted components in the format consumed by load_context().

        Return logical artifact names mapped to file paths for MLflow logging.
        Saving CPU copies of the state dict leaves the live model's device and
        training mode unchanged. The caller owns the staging directory's lifetime.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "preprocessor": str(directory / "preprocessor.pkl"),
            "weights": str(directory / "weights.pt"),
            "metadata": str(directory / "metadata.json"),
        }
        with open(artifacts["preprocessor"], "wb") as file:
            pickle.dump(preprocessor, file)
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, artifacts["weights"])
        with open(artifacts["metadata"], "w") as file:
            json.dump({"model_params": LSTMPipeline.model_params(model), "feature_columns": feature_columns}, file)
        return artifacts

    @staticmethod
    def log_pipeline(
        *,
        model: LSTMModel,
        preprocessor: SequencePreprocessor,
        feature_columns: list[str],
        run_id: str,
        registered_model_name: str,
        input_example: dict[str, np.ndarray],
    ) -> ModelInfo:
        """Package and register fitted components as one MLflow inference model.

        The caller creates the run and logs its config/metrics. This method owns
        staging, serialization, signature, code/dependencies, and registration.
        Temporary files are removed after MLflow copies them into its model store.
        """
        pip_requirements = [f"{package}=={version(package)}" for package in (
            "mlflow", "cloudpickle", "numpy", "pandas", "scikit-learn", "nflreadpy", "PyYAML",
        )]
        # Build suffixes such as +cpu require a separate index/platform. Pin the
        # public release so PyPI can select the destination platform's standard build.
        torch_version = version("torch").split("+", 1)[0]
        pip_requirements.append(f"torch=={torch_version}")

        with TemporaryDirectory(prefix="lstm-model-") as directory:
            artifacts = LSTMPipeline._save_artifacts(directory, model, preprocessor, feature_columns)
            with mlflow.start_run(run_id=run_id):
                return mlflow.pyfunc.log_model(
                    name="lstm",
                    python_model=LSTMPipeline(),
                    artifacts=artifacts,
                    code_paths=[str(Path(__file__).resolve().parents[2])],
                    signature=LSTMPipeline.build_signature(model.lstm.input_size),
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    pip_requirements=pip_requirements,
                )

    @staticmethod
    def build_signature(input_size: int) -> ModelSignature:
        """Describe raw array inputs and per-step outputs with variable batch/step sizes."""
        return ModelSignature(
            inputs=Schema([
                TensorSpec(np.dtype("float32"), (-1, -1, input_size), "X"),
                TensorSpec(np.dtype("int64"), (-1,), "lengths"),
            ]),
            outputs=Schema([TensorSpec(np.dtype("float32"), (-1, -1))]),
        )

    def load_context(self, context: PythonModelContext) -> None:
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

    def predict(
        self,
        context: PythonModelContext | None,
        model_input,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict from raw arrays without fitting preprocessing or using targets.

        model_input is dict[str, np.ndarray]. Its annotation is intentionally
        omitted because MLflow cannot infer that tensor-dictionary schema from
        type hints; the logged ModelSignature supplies dtype/shape validation.
        """
        X = torch.as_tensor(model_input["X"], dtype=torch.float32)
        lengths = torch.as_tensor(model_input["lengths"])

        data = SequenceDataset(X=X, lengths=lengths, identity=None, feature_columns=self.feature_columns)
        if X.shape[-1] != self.model.lstm.input_size:
            raise ValueError("X must have shape [N, L, model input_size].")
        if X.shape[0] == 0:
            return np.zeros(X.shape[:2], dtype=np.float32)

        prepared = self.preprocessor.transform(data)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(prepared.X.to(device), lengths)

        return predictions.cpu().numpy()
