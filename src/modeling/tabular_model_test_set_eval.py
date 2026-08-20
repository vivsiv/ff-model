import logging
import argparse
from datetime import datetime
from typing import Any, List, Optional

import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline

from src.modeling.data_prep import TabularModelDataPrep
from src.modeling.tabular_models import TabularModel
from src.modeling.utils import load_mlflow_model

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
    Loads a trained (and registered) tabular model from mlflow and scores it against its test set.

    The train/eval/test split is fully determined by the TabularModelDataPrep config logged
    (as a "data_prep_config.json" artifact) on the model's training run -- see
    TabularModel.setup_mlflow.

    Caveat: this assumes gold_dir/{target}__training_set.csv hasn't changed (rows added/removed/
    changed or columns removed/changed) since the model was trained.
    New feature columns are fine -- the model's feature_names_in_ params is used to extract the
    exact set of features used to train the model.
    """

    def __init__(
            self,
            data_dir: str,
            tracking_dir: str,
            target: str = "fantasy_points_ppr",
    ):
        self.data_dir = data_dir
        self.tracking_dir = tracking_dir
        self.target = target

    def _load_data_prep_config(self, source_run_id: str) -> dict[str, Any]:
        """
        Downloads and parses the data_prep_config.json artifact logged on source_run_id (see
        TabularModel.setup_mlflow).

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
        Restricts X to exactly the columns -- in the same order -- that pipeline was fit on
        (per sklearn's own pipeline.feature_names_in_), rather than whatever the current gold
        training set happens to contain.

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
        model_type: str,
        model_version: Optional[int] = None,
        top_ns: Optional[List[int]] = [50, 100, 200],
    ) -> pd.DataFrame:
        """
        Loads the model at {target}_{model_type}/versions/{model_version} (version is latest if model_version isn't
        given), then rebuilds the exact train/eval/test split the model was trained on from the
        TabularModelDataPrep config logged on its training run, and finally scores it against
        the test split.

        Logs a new run (tagged phase=test, alongside a source_run_id param pointing back at the
        training run) with the test R^2/RMSE and the full predictions-vs-actual CSV, in the same
        {target}_tabular mlflow experiment as the model's training/eval runs.

        Args:
            model_type: e.g. "random_forest"
            model_version: Specific registered version to evaluate. Defaults to the latest version.
            top_ns: Passed through to TabularModel.eval_model -- for each n, also logs
                top_{n}_r2/top_{n}_rmse (default: [50, 100, 200]). This is purely an
                evaluation-time choice (unlike the data prep config) so it isn't reconstructed
                from the source training run; pass an empty list/None to skip.

        Returns:
            DataFrame of test-set predictions vs actuals (see TabularModel.eval_model)
        """
        pipeline, mv = load_mlflow_model(self.target, model_type, model_version, self.tracking_dir)
        model_version, source_run_id = int(mv.version), mv.run_id
        config = self._load_data_prep_config(source_run_id)

        logger.info(
            f"Reconstructing split for {self.target}_{model_type} v{model_version} from "
            f"source run {source_run_id} using data prep config: {config}"
        )

        data_prep = TabularModelDataPrep(data_dir=self.data_dir, config=config)
        model = TabularModel(data_dir=self.data_dir, tracking_dir=self.tracking_dir, data_prep=data_prep)
        data = model.split_data()
        data["X_test"] = self._select_pipeline_features(pipeline, data["X_test"])

        run_id = model.setup_mlflow(
            model_type,
            extra_params={"source_run_id": source_run_id, "model_version": model_version},
            extra_tags={"phase": "test"},
            run_name=f"{model_type}_v{model_version}_test_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        )

        return model.eval_model(pipeline, data, run_id, split="test", top_ns=top_ns)


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
        "--target",
        type=str,
        default="fantasy_points_ppr",
        help="Which target's training set/registered model to use (default: fantasy_points_ppr)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        help="Registered model type to evaluate, one of: ridge, lasso, random_forest, svr, "
             "gradient_boosting, linear (default: random_forest)"
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Specific model version to evaluate. Defaults to the latest version."
    )

    args = parser.parse_args()

    evaluator = TabularModelTestSetEvaluator(data_dir=args.data_dir, tracking_dir=args.tracking_dir, target=args.target)
    test_preds_df = evaluator.evaluate(model_type=args.model_type, model_version=args.model_version)

    print(test_preds_df)


if __name__ == "__main__":
    main()
