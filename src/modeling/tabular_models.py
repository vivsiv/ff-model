import json
import os
import logging
import argparse
from datetime import datetime
from typing import Any, List, Optional

import pandas as pd
import mlflow
from mlflow.models import infer_signature
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from src.modeling.data_prep import TabularModelDataPrep
from src.modeling.utils import setup_mlflow_run

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
    """Fits/evaluates/logs a Tabular model."""

    def __init__(
            self,
            data_dir: str,
            tracking_dir: str,
            data_prep: TabularModelDataPrep,
    ):
        """
        Args:
            data_dir: Parent directory for the gold/predictions layers
            tracking_dir: mlflow tracking/registry store directory
            data_prep: Prepared data source -- see TabularModelDataPrep.
        """
        self.data_dir = data_dir
        self.tracking_dir = tracking_dir
        self.data_prep = data_prep
        self.target = data_prep.target

        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

    def split_data(self) -> dict[str, pd.DataFrame]:
        """Thin wrapper around self.data_prep.split() -- see TabularModelDataPrep.split."""
        return self.data_prep.split()

    def create_pipeline(self, model: Any = LinearRegression()) -> Pipeline:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        return pipeline

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

    def setup_mlflow(
        self,
        model_type: str,
        extra_params: Optional[dict] = None,
        extra_tags: Optional[dict] = None,
        run_name: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> str:
        """
        Sets the active mlflow experiment to {target}_tabular and creates a new run for
        it, tagged with model_type and phase=train. Logs self.data_prep.config as a
        "data_prep_config.json" artifact.

        Args:
            model_type: e.g. "random_forest"
            extra_params: Optional params to log on the run alongside the model's own
                hyperparameters.
            extra_tags: Optional tags merged over the default {"model_type": model_type,
                "phase": "train"}.
            run_name: Overrides the default "{model_type}_{timestamp}" run name.
            parent_run_id: If given, nests the new run under parent_run_id.

        Returns:
            run_id - The mlflow run to tie training and eval to.
        """
        run_name = run_name or f"{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        tags = {"model_type": model_type, "phase": "train", **(extra_tags or {})}

        run_id = setup_mlflow_run(
            experiment_name=f"{self.target}_tabular",
            run_name=run_name,
            tracking_dir=self.tracking_dir,
            tags=tags,
            params=extra_params,
            parent_run_id=parent_run_id,
        )
        mlflow.log_dict(self.data_prep.config, "data_prep_config.json", run_id=run_id)

        return run_id

    def fit_model(self, data: dict[str, pd.DataFrame], model_type: str, run_id: str, params: Optional[dict] = None) -> Pipeline:
        """
        Gets the base model for model_type, applies params (sklearn defaults if none given),
        fits it on the training split (weighted by data["sample_weight_train"], if present),
        and registers it to mlflow under run_id.

        Args:
            data: Output of split_data
            model_type: One of ridge, lasso, random_forest, svr, hist_gradient_boosting,
                linear_regression
            run_id: mlflow run_id from setup_mlflow, so the fit params/model artifact land in
                the same run eval_model logs its metrics/predictions into
            params: Hyperparameters to set on the base model before fitting.
                Uses sklearn's defaults if not given.

        Returns:
            The fit pipeline
        """
        base_model = self.get_base_model(model_type)
        base_model.set_params(**(params or {}))

        pipeline = self.create_pipeline(base_model)

        fit_params = {}
        if "sample_weight_train" in data:
            fit_params["model__sample_weight"] = data["sample_weight_train"]
        pipeline.fit(data["X_train"], data["y_train"], **fit_params)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(base_model.get_params())

            signature = infer_signature(data["X_train"], data["y_train"])
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                registered_model_name=f"{self.target}_{model_type}",
                name=model_type,
                input_example=data["X_train"],
                signature=signature)

        return pipeline

    def _log_performance_metrics(self, y: pd.Series, y_pred: np.ndarray, n: Optional[int] = None) -> None:
        """
        Logs R^2/RMSE for a slice of the split -- either the n rows with the highest actual
        value (logged as "top_{n}_r2"/"top_{n}_rmse"/"top_{n}_n"), or, if n is None, the
        entire split with no restriction (logged as plain "r2"/"rmse", matching what
        pipeline.score()/mean_squared_error would give directly). Must be called inside an
        active mlflow run.

        Args:
            y: True target values for the split being scored.
            y_pred: Predicted values, aligned positionally with y (as returned by
                pipeline.predict).
            n: Number of rows (highest actual value first) to restrict to. None (default)
                scores the entire split, logged under the unprefixed "r2"/"rmse" names. If
                the split has fewer than n rows, every row is used.
        """
        y_values = np.asarray(y)

        if n is None:
            idx = np.arange(len(y_values))
            prefix, label = "", "Overall"
        else:
            idx = np.argsort(y_values)[::-1][:n]
            prefix, label = f"top_{n}_", f"Top {n}"

        if len(idx) < 2:
            logger.warning(
                f"Only {len(idx)} row(s) available for {label}; skipping {prefix}r2/"
                f"{prefix}rmse (need at least 2 to compute a meaningful R^2)"
            )
            return

        y_slice, y_pred_slice = y_values[idx], y_pred[idx]

        rmse = np.sqrt(mean_squared_error(y_slice, y_pred_slice))
        print(f"{label} RMSE: {rmse} (n={len(idx)})")
        mlflow.log_metric(f"{prefix}rmse", rmse)

        ss_tot = ((y_slice - y_slice.mean()) ** 2).sum()
        if ss_tot > 0:
            r2 = 1 - ((y_slice - y_pred_slice) ** 2).sum() / ss_tot
            print(f"{label} R^2: {r2}")
            mlflow.log_metric(f"{prefix}r2", r2)
        else:
            logger.warning(f"actual has zero variance for {label}; skipping {prefix}r2")

    def eval_model(
        self,
        pipeline: Pipeline,
        data: dict[str, pd.DataFrame],
        run_id: str,
        split: str = "eval",
        top_ns: Optional[List[int]] = [50, 100, 200],
    ) -> pd.DataFrame:
        """
        Scores a fitted pipeline against a held-out split. Logs whole-split R^2/RMSE, plus
        (for each n in top_ns) R^2/RMSE restricted to the n rows with the highest actual value
        (see _log_performance_metrics). Also logs the full dataset of predictions vs actual
        for the split as a CSV artifact.

        Args:
            pipeline: A fit pipeline.
            data: Output of split_data
            run_id: The mlflow run_id to log metrics/artifacts into. When called right after
                fit_model, pass the same run_id so eval metrics land alongside the fit
                params/model artifact.
            split: Which split of data to score against, "eval" or "test" (default: "eval").
                Selects data["X_{split}"]/data["y_{split}"]/data["identity_{split}"], and
                namespaces the predictions CSV filename/artifact path with the split name.
            top_ns: For each n in this list, also logs "top_{n}_r2"/"top_{n}_rmse" (default:
                [50, 100, 200]). Pass an empty list/None to only log whole-split r2/rmse.
        """
        X, y, identity = data[f"X_{split}"], data[f"y_{split}"], data[f"identity_{split}"]

        y_pred = pipeline.predict(X)

        preds_df = identity.copy()
        preds_df["predictions"] = y_pred
        preds_df["actual"] = y
        preds_df = preds_df.sort_values(by=["target_season", "predictions", "actual"], ascending=False)

        with mlflow.start_run(run_id=run_id):
            for n in [None, *(top_ns or [])]:
                self._log_performance_metrics(y, y_pred, n)

            output_df = preds_df.rename(columns={"predictions": self.target})
            output_df[self.target] = output_df[self.target].round(2)

            csv_path = os.path.join(self.predictions_dir, f"{self.target}_{split}_predictions_{run_id}.csv")
            output_df[["player_display_name", "target_season", self.target, "actual"]].to_csv(csv_path, index=False)

            mlflow.log_artifact(csv_path, f"{split}_predictions")

        return preds_df

    def param_search(
        self,
        data: dict[str, pd.DataFrame],
        model_type: str,
        param_grid: dict[str, list],
    ) -> pd.DataFrame:
        """
        Runs a one-at-a-time hyperparameter sweep: for each key:value combo in param_grid.

        Creates one parent mlflow run per key and nests one child run per (key, value) combo.

        Args:
            data: Output of split_data. The same split is reused for every candidate so
                results are directly comparable.
            model_type: One of ridge, lasso, random_forest, svr, hist_gradient_boosting,
                linear_regression
            param_grid: e.g. {"n_estimators": [100, 200, 300], "min_samples_split": [2, 4, 8]}
                -- yields 6 total child runs (3 + 3), not a 3x3=9 cartesian grid.

        Returns:
            DataFrame with one row per (param, value) combo with columns: search_param,
            search_value, run_id, eval_rmse, eval_r2, and eval_top_{n}_rmse/eval_top_{n}_r2
            for whichever top_ns eval_model logged (its own default is used since top_ns
            isn't threaded through here).
        """
        results = []

        for key, values in param_grid.items():
            parent_run_id = self.setup_mlflow(
                model_type,
                extra_params={"search_param": key, "search_values": values},
                extra_tags={"phase": f"{key}_search"},
                run_name=f"{model_type}_{key}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            )

            for value in values:
                params = {key: value}
                child_run_id = self.setup_mlflow(
                    model_type,
                    extra_params={"search_param": key, "search_value": value},
                    extra_tags={"phase": f"{key}_{value}_child"},
                    run_name=f"{model_type}_{key}_{value}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    parent_run_id=parent_run_id,
                )

                pipeline = self.fit_model(data, model_type, child_run_id, params=params)
                self.eval_model(pipeline, data, child_run_id)

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
    model = TabularModel(data_dir=args.data_dir, tracking_dir=args.tracking_dir, data_prep=data_prep)
    data = model.split_data()

    if args.param_grid:
        param_grid = json.loads(args.param_grid)
        model.param_search(data, args.model_type, param_grid)
    else:
        run_id = model.setup_mlflow(args.model_type, extra_params={"config_path": args.config})
        pipeline = model.fit_model(data, args.model_type, run_id)
        model.eval_model(pipeline, data, run_id)


if __name__ == "__main__":
    main()
