import ast
import logging
import argparse
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import mlflow

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


class TabularModelTestSetEvaluator:
    """
    Loads a trained (and registered) tabular model from mlflow and scores it against its test set.

    The train/eval/test split is fully determined by the params: excluded_features,
    eval_data_years, test_data_years, and num_training_seasons (all of which must be recoverable from mlflow)
    and TabularModel's CLI (main()).

    Caveat: this assumes gold_dir/{target}__training_set.csv hasn't changed (rows added/removed/
    changed) since the model was trained.
        TODO: what if features were added.
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

    def _parse_split_data_params(self, params: dict[str, str]) -> dict[str, Any]:
        """
        Casts the string-valued mlflow params logged by tabular_models.py's CLI (main()) back
        to the types TabularModel.split_data expects.

        Args:
            params: mlflow.entities.Run.data.params from the model's training run

        Returns:
            dict with excluded_features (list[str]), eval_data_years (int), test_data_years
            (int), num_training_seasons (Optional[int])

        Raises:
            KeyError: If the run is missing one of the expected split params -- e.g. it was
                created by fit_model()/param_search() directly rather than main()'s CLI.
        """
        required = ["excluded_features", "eval_data_years", "test_data_years", "num_training_seasons"]
        missing = [key for key in required if key not in params]
        if missing:
            raise KeyError(
                f"Run is missing split param(s) {missing}; it must have been created by "
                "tabular_models.py's CLI (main()), not fit_model()/param_search() directly"
            )

        num_training_seasons = params["num_training_seasons"]

        return {
            "excluded_features": ast.literal_eval(params["excluded_features"]),
            "eval_data_years": int(params["eval_data_years"]),
            "test_data_years": int(params["test_data_years"]),
            "num_training_seasons": None if num_training_seasons == "all" else int(num_training_seasons),
        }

    def evaluate(self, model_type: str, model_version: Optional[int] = None) -> pd.DataFrame:
        """
        Loads the model at {target}_{model_type}/versions/{model_version} (version is latest if model_version isn't
        given), then rebuilds the exact train/eval/test split the model was trained on from params logged to mlflow,
        and finally scores it against the test split.

        Logs a new run (tagged phase=test, alongside a source_run_id param pointing back at the
        training run) with the test R^2/RMSE and the full predictions-vs-actual CSV, in the same
        {target}_tabular mlflow experiment as the model's training/eval runs.

        Args:
            model_type: e.g. "random_forest"
            model_version: Specific registered version to evaluate. Defaults to the latest version.

        Returns:
            DataFrame of test-set predictions vs actuals (see TabularModel.eval_model)
        """
        pipeline, mv = load_mlflow_model(self.target, model_type, model_version, self.tracking_dir)
        model_version, source_run_id = int(mv.version), mv.run_id
        source_run_params = mlflow.get_run(source_run_id).data.params
        split_params = self._parse_split_data_params(source_run_params)

        logger.info(
            f"Reconstructing split for {self.target}_{model_type} v{model_version} from "
            f"source run {source_run_id}: {split_params}"
        )

        model = TabularModel(
            data_dir=self.data_dir,
            tracking_dir=self.tracking_dir,
            target=self.target,
            excluded_features=split_params["excluded_features"],
        )
        data = model.split_data(
            eval_data_years=split_params["eval_data_years"],
            test_data_years=split_params["test_data_years"],
            num_training_seasons=split_params["num_training_seasons"],
        )

        run_id = model.setup_mlflow(
            model_type,
            extra_params={"source_run_id": source_run_id, "model_version": model_version},
            extra_tags={"phase": "test"},
            run_name=f"{model_type}_v{model_version}_test_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        )

        return model.eval_model(pipeline, data, run_id, split="test")


def main():
    parser = argparse.ArgumentParser(
        description="Scores an already-trained, registered model against the test split it was "
                     "trained with, reconstructed from that model's logged split params"
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
