import os
import logging
import argparse
from datetime import datetime
from typing import Any, List, Optional

import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline

from src.modeling.data_prep import TabularModelDataPrep
from src.modeling.utils import load_mlflow_model, predict, score, set_mlflow_tracking_uri, setup_mlflow_run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tabular_model_test_set_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_PREP_CONFIG_ARTIFACT_PATH = "data_prep_config.json"


class TabularModelTestSetEvaluator:
    """
    Loads a trained tabular model from mlflow and scores it against its test set.

    Caveat: this assumes gold_dir/{target}__training_set.csv hasn't fundamentally changed
    (rows added/removed/ changed or columns removed/changed) since the model was trained.
    """

    def __init__(
            self,
            data_dir: str,
            tracking_dir: str,
            registered_model: str,
            model_version: Optional[int],
    ):
        self.data_dir = data_dir
        self.tracking_dir = tracking_dir
        self.registered_model = registered_model

        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

        self.pipeline, self.mv = load_mlflow_model(registered_model, model_version, self.tracking_dir)
        self.model_version, self.source_run_id = int(self.mv.version), self.mv.run_id
        self.config = self._load_data_prep_config(self.source_run_id)

        self.data_prep = TabularModelDataPrep(data_dir=self.data_dir, config=self.config)

    def _load_data_prep_config(self, source_run_id: str) -> dict[str, Any]:
        """
        Downloads and parses the data_prep_config.json artifact logged on source_run_id.

        Args:
            source_run_id: The training run to pull the config from.

        Returns:
            The logged TabularModelDataPrep config dict.

        Raises:
            RuntimeError: If the run has no data_prep_config.json artifact -- e.g. it was
                created by fit_model()/param_search() directly, or by a version of
                tabular_models.py predating this config-driven data prep.
        """
        try:
            return mlflow.artifacts.load_dict(f"runs:/{source_run_id}/{DATA_PREP_CONFIG_ARTIFACT_PATH}")
        except mlflow.exceptions.MlflowException as e:
            raise RuntimeError(
                f"Run {source_run_id} has no {DATA_PREP_CONFIG_ARTIFACT_PATH} artifact."
            ) from e

    def _select_pipeline_features(self, pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
        """
        Restricts X to exactly the columns -- in the same order -- the pipeline was fit on.

        Args:
            pipeline: The fit pipeline being evaluated.
            X: Candidate feature set for the test split.

        Returns:
            X restricted to pipeline.feature_names_in_, in that exact order.

        Raises:
            ValueError: If the pipeline needs a column that's no longer in X.
        """
        required_features = list(pipeline.feature_names_in_)
        missing = [col for col in required_features if col not in X.columns]
        if missing:
            raise ValueError(
                f"Training set is missing {len(missing)} column(s) the model was fit on: "
                f"{missing}. It may have been renamed/removed since the model was trained."
            )

        return X[required_features]

    def evaluate(
        self,
        top_ns: Optional[List[int]] = [50, 100, 200],
    ) -> pd.DataFrame:
        """
        Loads a registered model @ version model_version (model_version is latest if not provided).
        Rebuilds the exact train/eval/test split the model was trained on from its
        data prep config, and finally scores it against the test set.

        Logs a new run (tagged phase=test, alongside a source_run_id param pointing back at the
        training run) with the test R^2/RMSE and the full predictions-vs-actual CSV, in the same
        mlflow experiment as the model's training/eval runs (found by looking up which
        experiment source_run_id itself lives in, rather than recomputing the name).

        Args:
            top_ns: For each n, also logs top_{n}_r2/top_{n}_rmse (default: [50, 100, 200]).

        Returns:
            DataFrame of test-set predictions vs actuals (see predict())
        """
        data = self.data_prep.split()
        X_test = self._select_pipeline_features(self.pipeline, data["X_test"])

        set_mlflow_tracking_uri(self.tracking_dir)
        source_experiment_id = mlflow.get_run(self.source_run_id).info.experiment_id
        experiment_name = mlflow.get_experiment(source_experiment_id).name

        run_id = setup_mlflow_run(
            experiment_name=experiment_name,
            run_name=f"{self.registered_model}_v{self.model_version}_test_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            tracking_dir=self.tracking_dir,
            tags={"phase": "test"},
            params={"source_run_id": self.source_run_id, "model_version": self.model_version},
        )
        mlflow.log_dict(self.config, "data_prep_config.json", run_id=run_id)

        csv_path = os.path.join(
            self.predictions_dir, f"{self.registered_model}_v{self.model_version}_test_predictions_{run_id}.csv"
        )

        preds_df = predict(
            pipeline=self.pipeline,
            X=X_test,
            identity=data["identity_test"],
            target=self.data_prep.target,
            run_id=run_id,
            csv_path=csv_path,
            artifact_path="test_predictions",
            y=data["y_test"],
        )
        score(preds_df, run_id, top_ns=top_ns)

        return preds_df


def main():
    parser = argparse.ArgumentParser(
        description="Scores an already-trained, registered model against the test split it was "
                     "trained with, reconstructed from that model's logged data prep config"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Parent directory for the gold/predictions layers, relative to the repo root (default: data)"
    )
    parser.add_argument(
        "--tracking-dir",
        type=str,
        default="mlruns",
        help="Top-level mlruns tracking/registry store directory, not nested under "
             "--data-dir, relative to the repo root (default: mlruns)"
    )
    parser.add_argument(
        "--registered-model",
        type=str,
        required=True,
        help="Name of the model as registered in mlflow to evaluate."
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Specific model version to evaluate. Defaults to the latest version."
    )

    args = parser.parse_args()

    evaluator = TabularModelTestSetEvaluator(
        data_dir=args.data_dir,
        tracking_dir=args.tracking_dir,
        registered_model=args.registered_model,
        model_version=args.model_version)
    test_preds_df = evaluator.evaluate()

    print(test_preds_df)


if __name__ == "__main__":
    main()
