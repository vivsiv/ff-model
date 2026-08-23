import json
import os
import logging
import argparse
from datetime import datetime
from typing import Any, List, Optional

import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from src.modeling.data_prep import TabularModelDataPrep
from src.modeling.utils import predict, score, setup_mlflow_run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tabular_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TabularModel:
    """Fits a Tabular model to data, gathers performance on the eval set, and logs it to mlflow."""

    def __init__(
            self,
            data_dir: str,
            tracking_dir: str,
            data_prep: TabularModelDataPrep,
            model_type: str,
    ):
        """
        Args:
            data_dir: Parent directory in which to log model artifacts
            tracking_dir: mlflow tracking/registry store directory
            data_prep: Prepared data source -- see TabularModelDataPrep.
            model_type: Type of model being fit (i.e.) random forest
        """
        self.data_dir = data_dir
        self.tracking_dir = tracking_dir
        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

        self.data_prep = data_prep
        self.data = data_prep.split()
        self.target = data_prep.target

        self.model_type = model_type

    @property
    def base_model_name(self) -> str:
        positions = [pos.lower() for pos in self.data_prep.positions] if self.data_prep.positions else None
        suffix = f"_{'_'.join(sorted(positions))}" if positions else ""

        return f"{self.target}{suffix}"

    def get_base_model(self, model_type: str) -> Any:
        base_models = {
            'ridge': Ridge(alpha=1000),
            'lasso': Lasso(), # Lasso not performant for ppr_ppg
            'random_forest': RandomForestRegressor(n_jobs=-1, n_estimators=200, min_samples_leaf=8),
            'svr': SVR(),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, min_samples_leaf=24),
            'linear': LinearRegression(),
        }
        return base_models[model_type]

    def create_pipeline(self, model: Any = LinearRegression()) -> Pipeline:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        return pipeline

    def setup_mlflow(
        self,
        run_name: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        extra_params: Optional[dict] = None,
        extra_tags: Optional[dict] = None,
    ) -> str:
        """
        Sets the active mlflow experiment and creates a new run for it, tagged with model_type and phase=train.
        Logs self.data_prep.config as a "data_prep_config.json" artifact.

        Args:
            run_name: Overrides the default "{model_type}_{timestamp}" run name.
            parent_run_id: If given, nests the new run under parent_run_id.
            extra_params: Optional params to log on the run alongside the model's own hyperparameters.
            extra_tags: Optional tags over the default {"model_type": model_type, "phase": "train"}.

        Returns:
            run_id - The resulting mlflow run id.
        """
        run_name = run_name or f"{self.model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        tags = {"model_type": self.model_type, "phase": "train", **(extra_tags or {})}

        run_id = setup_mlflow_run(
            experiment_name=f"{self.base_model_name}_tabular",
            run_name=run_name,
            tracking_dir=self.tracking_dir,
            tags=tags,
            params=extra_params,
            parent_run_id=parent_run_id,
        )
        mlflow.log_dict(self.data_prep.config, "data_prep_config.json", run_id=run_id)

        return run_id

    def fit_model(
        self,
        run_id: str,
        params: Optional[dict] = None) -> Pipeline:
        """
        Gets the base model for model_type, applies params (sklearn defaults if none given),
        fits it on the training split (weighted by data["sample_weight_train"], if present),
        and registers it to mlflow under run_id.

        Args:
            run_id: mlflow run_id from setup_mlflow, so the fit params/model artifact land in
                the same run eval_model logs its metrics/predictions into
            params: Hyperparameters to set on the base model before fitting.
                Uses sklearn's defaults if not given.

        Returns:
            The fit pipeline
        """
        base_model = self.get_base_model(self.model_type)
        base_model.set_params(**(params or {}))

        pipeline = self.create_pipeline(base_model)

        X_train = self.data["X_train"]
        y_train = self.data["y_train"]

        fit_params = {}
        if "sample_weight_train" in self.data:
            fit_params["model__sample_weight"] = self.data["sample_weight_train"]
        pipeline.fit(X_train, y_train, **fit_params)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(base_model.get_params())

            signature = infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                registered_model_name=f"{self.base_model_name}_{self.model_type}",
                name=self.model_type,
                input_example=self.data["X_train"],
                signature=signature)

        return pipeline

    def eval_model(
        self,
        pipeline: Pipeline,
        run_id: str,
        split: str = "eval",
        top_ns: Optional[List[int]] = [50, 100, 200],
    ) -> pd.DataFrame:
        """
        Scores a fitted pipeline against a held-out evaluation data set via predict()/score()
        -- see their docstrings for what gets logged/returned.

        Args:
            pipeline: A fit pipeline.
            run_id: The mlflow run_id to log metrics/artifacts to.
            split: Which evaluation dataset to score against, "eval" or "test" (default: "eval").
            top_ns: For each n in this list, log "top_{n}_r2"/"top_{n}_rmse" (default: [50, 100, 200]).
        """
        csv_path = os.path.join(
            self.predictions_dir, f"{self.base_model_name}_{self.model_type}_{split}_predictions_{run_id}.csv"
        )

        preds_df = predict(
            pipeline=pipeline,
            X=self.data[f"X_{split}"],
            identity=self.data[f"identity_{split}"],
            target=self.target,
            run_id=run_id,
            csv_path=csv_path,
            artifact_path=f"{split}_predictions",
            y=self.data[f"y_{split}"],
        )
        score(preds_df, run_id, top_ns=top_ns)

        return preds_df

    def param_search(
        self,
        param_grid: dict[str, list],
    ) -> pd.DataFrame:
        """
        Runs a one-at-a-time hyperparameter sweep: for each key:value combo in param_grid.

        Creates one parent mlflow run per key and nests one child run per (key, value) combo.

        Args:
            param_grid: e.g. {"n_estimators": [100, 200, 300], "min_samples_split": [2, 4, 8]}
                -- yields 6 total child runs (3 + 3), not a 3x3=9 cartesian grid.

        Returns:
            DataFrame with one row per (param, value) combo with columns: search_param,
            search_value, run_id, eval_rmse, eval_r2, and eval_top_{n}_rmse/eval_top_{n}_r2.
        """
        results = []

        for key, values in param_grid.items():
            parent_run_id = self.setup_mlflow(
                run_name=f"{self.model_type}_{key}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                extra_params={"search_param": key, "search_values": values},
                extra_tags={"phase": f"{key}_search"},
            )

            for value in values:
                params = {key: value}
                child_run_id = self.setup_mlflow(
                    run_name=f"{self.model_type}_{key}_{value}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    parent_run_id=parent_run_id,
                    extra_params={"search_param": key, "search_value": value},
                    extra_tags={"phase": f"{key}_{value}_child"},
                )

                pipeline = self.fit_model(run_id=child_run_id, params=params)
                self.eval_model(pipeline, child_run_id)

                metrics = mlflow.get_run(child_run_id).data.metrics
                result = {
                    "search_param": key,
                    "search_value": value,
                    "run_id": child_run_id,
                    "eval_rmse": metrics.get("rmse"),
                    "eval_r2": metrics.get("r2"),
                }
                for metric_name, metric_value in metrics.items():
                    if metric_name.startswith("top_") and metric_name.endswith(("_r2", "_rmse")):
                        result[f"eval_{metric_name}"] = metric_value
                results.append(result)

        results_df = pd.DataFrame(results)
        logger.info(f"Param search results:\n{results_df.to_string(index=False)}")

        return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Fits and evaluates a tabular model against a gold training set"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Parent directory for model artifacts, relative to the repo root (default: data)"
    )
    parser.add_argument(
        "--tracking-dir",
        type=str,
        default="mlruns",
        help="Top-level mlruns tracking/registry store directory, not nested under "
             "--data-dir, relative to the repo root (default: mlruns)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a TabularModelDataPrep YAML config file (see configs/*.yaml), which "
             "determines the target being modeled plus how its training set is "
             "split/feature-filtered/sample-weighted."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        help="Model type to fit, one of: ridge, lasso, random_forest, svr, hist_gradient_boosting, linear_regression (default: random_forest)"
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        default=None,
        help="JSON dict of {param: [values]} for a one-at-a-time hyperparameter search, e.g. "
             '\'{"n_estimators": [100,200,300], "min_samples_split": [2,4,8]}\'. Iterates through each '
             "key independently, holding every other param at its sklearn default."
    )

    args = parser.parse_args()

    data_prep = TabularModelDataPrep.from_config_file(data_dir=args.data_dir, config_path=args.config)
    model = TabularModel(data_dir=args.data_dir, tracking_dir=args.tracking_dir, data_prep=data_prep, model_type=args.model_type)

    if args.param_grid:
        model.param_search(json.loads(args.param_grid))
    else:
        run_id = model.setup_mlflow(extra_params={"config_path": args.config})
        pipeline = model.fit_model(run_id=run_id)
        model.eval_model(pipeline=pipeline, run_id=run_id)


if __name__ == "__main__":
    main()
