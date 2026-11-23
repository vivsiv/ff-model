"""Dataset-level prediction and evaluation for fitted sequence pipelines."""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch

from src.modeling.data_builder import EVAL_SPLIT, PREDICTION_SPLIT, TEST_SPLIT
from src.modeling.inference import ModelInference
from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.pipeline import LSTMPipeline

logger = logging.getLogger(__name__)


class SequenceModelInference(ModelInference[LSTMPipeline | mlflow.pyfunc.PyFuncModel, SequenceModelDataBuilder]):
    """Predict raw sequences and score held-out steps without fitting anything.

    Accept an in-memory LSTMPipeline or use from_registry() to load a version and
    reconstruct its saved data_builder config.
    The data builder supplies raw SeqeuenceDataset splits and the pipeline applies saved preprocessing
    and runs the network in evaluation mode.
    """

    model_family = "sequence"

    def __init__(
        self,
        pipeline: LSTMPipeline | mlflow.pyfunc.PyFuncModel,
        data_builder: SequenceModelDataBuilder,
        tracking_dir: str | None = None,
        registered_model: str | None = None,
        model_version: int | None = None,
        source_run_id: str | None = None,
    ) -> None:
        super().__init__(
            pipeline=pipeline, data_builder=data_builder, tracking_dir=tracking_dir,
            registered_model=registered_model, model_version=model_version, source_run_id=source_run_id,
        )
        python_model = pipeline.unwrap_python_model() if isinstance(pipeline, mlflow.pyfunc.PyFuncModel) else pipeline

        if python_model.feature_columns is None:
            raise ValueError("The pipeline must have fitted feature columns before inference.")
        self.feature_columns = list(python_model.feature_columns)

    @staticmethod
    def _build_data_builder(data_dir: str, config: dict[str, Any]) -> SequenceModelDataBuilder:
        return SequenceModelDataBuilder(data_dir=data_dir, config=config)

    def predict_split(self, data: SequenceDataset) -> np.ndarray:
        """Predict raw steps, verifying feature names/order and preserving [N, L]."""
        if data.feature_columns != self.feature_columns:
            raise ValueError("Feature columns must match the saved pipeline in order.")

        inputs = {
            "X": data.X.detach().cpu().numpy(),
            "lengths": data.lengths.cpu().numpy(),
        }
        if isinstance(self.pipeline, mlflow.pyfunc.PyFuncModel):
            return self.pipeline.predict(inputs)

        return self.pipeline.predict(context=None, model_input=inputs)

    def _build_predictions_frame(
        self,
        data: SequenceDataset,
        predictions: np.ndarray,
        step_mask: torch.Tensor,
        actual: torch.Tensor | None = None,
    ) -> pd.DataFrame:
        """Build selected identity rows, gathering predictions and optional actuals by coordinates."""
        sequence_indices = data.identity["seq_idx"].to_numpy(dtype=np.int64)
        step_indices = data.identity["step_idx"].to_numpy(dtype=np.int64)
        selected = step_mask.cpu().numpy()[sequence_indices, step_indices]

        frame = data.identity.loc[selected].drop(columns=["seq_idx", "step_idx"])
        sequence_indices = sequence_indices[selected]
        step_indices = step_indices[selected]
        frame["predictions"] = predictions[sequence_indices, step_indices]

        if not np.isfinite(frame["predictions"]).all():
            raise ValueError("Non-finite predictions encountered.")

        if actual is not None:
            frame["actual"] = actual.detach().cpu().numpy()[sequence_indices, step_indices]
        sort_columns = ["target_season", "predictions"] if "target_season" in frame else ["predictions"]

        return frame.sort_values(sort_columns, ascending=False).reset_index(drop=True)

    def _save_predictions(self, frame: pd.DataFrame, csv_path: str, run_id: str | None) -> None:
        """Write identity and target-named predictions, optionally logging the CSV."""
        output = frame.rename(columns={"predictions": self.data_builder.target})
        output[self.data_builder.target] = output[self.data_builder.target].round(2)
        output.to_csv(csv_path, index=False)

        if run_id is not None:
            mlflow.log_artifact(csv_path, run_id=run_id)

    def predict(self, run_id: str | None = None, csv_path: str | None = None) -> pd.DataFrame:
        """Return each prediction sequence's last-step output with player identity, without scoring."""
        data = self.data_builder.get_data_splits()[PREDICTION_SPLIT]
        predictions = self.predict_split(data)
        frame = self._build_predictions_frame(data, predictions, step_mask=data.last_step_mask)

        if csv_path is not None:
            self._save_predictions(frame, csv_path, run_id)

        return frame

    def evaluate(
        self,
        split: str = EVAL_SPLIT,
        run_id: str | None = None,
        step: int | None = None,
        csv_path: str | None = None,
    ) -> float:
        """Evaluate the model on the provided split, save the predictions, log to mlflow, and return unweighted eval/test MSE.

        Only loss-mask steps are scored; history is retained as input context.

        When csv_path is supplied, export scored identity rows and unrounded
        actuals, with predictions rounded only in the CSV (not in the metric).
        """
        data = self.data_builder.get_data_splits()[split]

        if data.y is None or data.loss_mask is None:
            raise ValueError("Targets and a loss mask are required for evaluation.")
        if not data.loss_mask.any():
            raise ValueError(f"Cannot evaluate an empty '{split}' split with no scored steps.")

        loss_mask = data.loss_mask.cpu().numpy()

        predictions = self.predict_split(data)
        actual = data.y.detach().cpu().numpy()[loss_mask]
        mse = float(np.mean((predictions[loss_mask] - actual) ** 2))

        if not np.isfinite(mse):
            raise ValueError("Non-finite evaluation MSE encountered.")
        if csv_path is not None:
            frame = self._build_predictions_frame(data, predictions, step_mask=data.loss_mask, actual=data.y)
            self._save_predictions(frame, csv_path, run_id)
        if run_id is not None:
            mlflow.log_metrics(
                {f"{split}_mse": mse, f"{split}_rmse": mse ** 0.5},
                step=step, run_id=run_id,
            )

        return mse


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate a registered sequence model or export predictions using its saved data-builder config",
    )
    parser.add_argument("--data-dir", default="data", help="Parent directory for gold/predictions data (default: data)")
    parser.add_argument("--tracking-dir", default="mlruns", help="MLflow tracking/registry directory (default: mlruns)")
    parser.add_argument("--registered-model", required=True, help="Registered model name, e.g. ppr_points_per_game_rb_lstm")
    parser.add_argument("--model-version", type=int, default=None, help="Model version (default: latest)")
    parser.add_argument("--mode", choices=[EVAL_SPLIT, TEST_SPLIT, "predict"], required=True)
    args = parser.parse_args()

    inference = SequenceModelInference.from_registry(
        data_dir=args.data_dir,
        tracking_dir=args.tracking_dir,
        registered_model=args.registered_model,
        model_version=args.model_version,
    )

    run_id = inference.setup_mlflow(phase=args.mode)
    predictions_dir = os.path.join(args.data_dir, "predictions")
    csv_path = SequenceModelInference.build_predictions_csv_path(
        predictions_dir, inference.registered_model,
        inference.model_version, run_id, split=args.mode if args.mode != "predict" else None,
    )
    if args.mode in (EVAL_SPLIT, TEST_SPLIT):
        mse = inference.evaluate(split=args.mode, run_id=run_id, csv_path=csv_path)
        logger.info("%s MSE: %s; RMSE: %s", args.mode, mse, mse ** 0.5)
    else:
        print(inference.predict(run_id=run_id, csv_path=csv_path).to_string(index=False))

    logger.info("MLflow inference run: %s; predictions: %s", run_id, csv_path)


if __name__ == "__main__":
    main()
