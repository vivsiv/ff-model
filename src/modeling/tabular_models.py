import os
import json
import logging
import argparse
from datetime import datetime
from typing import Any, Optional

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

from src.modeling.utils import setup_mlflow_run
from src.processing.column_registry import get_identity_columns
from src.processing.gold import TARGET_COL

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
    """Loads the gold training set, splits it, and fits/evaluates/logs models."""

    def __init__(
            self,
            data_dir: str,
            tracking_dir: str,
            target: str,
            excluded_features: Optional[list[str]] = None,
    ):
        self.data_dir = data_dir
        self.gold_data_dir = os.path.join(data_dir, "gold")
        self.target = target
        self.training_data = self.load_data()

        self.tracking_dir = tracking_dir

        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

        self.excluded_features = excluded_features or []
        self.identity_cols = get_identity_columns("nflverse", "player_stats") + ["target_season"]
        self.feature_cols = [
            col for col in self.training_data.columns
            if col not in self.identity_cols + [TARGET_COL]
            and not self._is_excluded_feature(col)
        ]

        self.identity_df = self.training_data[self.identity_cols]
        self.features_df = self.training_data[self.feature_cols]
        self.target_df = self.training_data[TARGET_COL]

    def _is_excluded_feature(self, col: str) -> bool:
        """Returns True if col matches any entry in self.excluded_features. Entries
        ending in "*" are treated as prefixes (e.g. "f*" matches any column starting
        with "f"); all other entries must match col exactly."""
        for excluded in self.excluded_features:
            if excluded.endswith("*"):
                if col.startswith(excluded[:-1]):
                    return True
            elif col == excluded:
                return True

        return False

    def load_data(self) -> pd.DataFrame:
        filename = f"{self.target}__training_set.csv"
        data = pd.read_csv(os.path.join(self.gold_data_dir, filename))
        logger.info(f"Loaded data: {len(data)} rows")

        return data

    def split_data(
            self,
            eval_data_years: int = 1,
            test_data_years: int = 1,
            num_training_seasons: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Splits chronologically by target_season into train/eval/test. The most recent
        test_data_years worth of seasons become the test set. The most recent
        eval_data_years worth of seasons not already claimed by the test set become the
        eval set. The remaining seasons are training data, the number of seasons used
        for training can be limited with num_training_seasons.

        Args:
            eval_data_years: Number of seasons (immediately preceding the test set) to
                hold out for eval (default: 1)
            test_data_years: Number of most recent seasons to hold out for test (default: 1)
            num_training_seasons: Number of most recent seasons (immediately preceding the
                eval set) to keep for training; older seasons are dropped. (default: None)

        Returns:
            dict with X_train/X_eval/X_test, y_train/y_eval/y_test,
            identity_train/identity_eval/identity_test

        Raises:
            ValueError: If num_training_seasons + eval_data_years + test_data_years exceeds
                the total number of distinct seasons in the training set.
        """
        target_season = self.training_data["target_season"]
        total_seasons = target_season.nunique()

        if num_training_seasons is not None:
            requested_seasons = num_training_seasons + eval_data_years + test_data_years
            if requested_seasons > total_seasons:
                raise ValueError(
                    f"num_training_seasons ({num_training_seasons}) + eval_data_years "
                    f"({eval_data_years}) + test_data_years ({test_data_years}) = "
                    f"{requested_seasons}, which exceeds the {total_seasons} distinct "
                    "season(s) available in the training set"
                )

        max_target_season = target_season.max()
        test_cutoff_season = max_target_season - test_data_years + 1
        eval_cutoff_season = test_cutoff_season - eval_data_years

        is_test = target_season >= test_cutoff_season
        is_eval = (target_season >= eval_cutoff_season) & ~is_test

        if num_training_seasons is None:
            is_train = ~is_eval & ~is_test
        else:
            train_cutoff_season = eval_cutoff_season - num_training_seasons
            is_train = (target_season >= train_cutoff_season) & ~is_eval & ~is_test

        data = {
            "X_train": self.features_df[is_train],
            "X_eval": self.features_df[is_eval],
            "X_test": self.features_df[is_test],
            "y_train": self.target_df[is_train],
            "y_eval": self.target_df[is_eval],
            "y_test": self.target_df[is_test],
            "identity_train": self.identity_df[is_train],
            "identity_eval": self.identity_df[is_eval],
            "identity_test": self.identity_df[is_test],
        }
        return data

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
        Sets the active mlflow experiment to {target}_tabular (shared across all model types
        for this target, so they're directly comparable in mlflow) and creates a new run for
        it, tagged with model_type and phase=train.

        Args:
            model_type: e.g. "random_forest"
            extra_params: Optional params to log on the run alongside the model's own
                hyperparameters, e.g. {"eval_data_years": 1, "test_data_years": 1}
            extra_tags: Optional tags merged over the default {"model_type": model_type,
                "phase": "train"}, e.g. {"phase": "sweep"} to override the default phase tag
            run_name: Overrides the default "{model_type}_{timestamp}" run name
            parent_run_id: If given, nests the new run under parent_run_id (see
                setup_mlflow_run) -- used by grid_search to group a sweep's child runs under
                a single parent run in the mlflow UI.

        Returns:
            run_id - The mlflow run to tie training and eval to.
        """
        run_name = run_name or f"{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        tags = {"model_type": model_type, "phase": "train", **(extra_tags or {})}

        return setup_mlflow_run(
            experiment_name=f"{self.target}_tabular",
            run_name=run_name,
            tracking_dir=self.tracking_dir,
            tags=tags,
            params=extra_params,
            parent_run_id=parent_run_id,
        )

    def fit_model(self, data: dict[str, pd.DataFrame], model_type: str, run_id: str, params: Optional[dict] = None) -> Pipeline:
        """
        Gets the base model for model_type, applies params (sklearn defaults if none given),
        fits it on the training split, and registers it to mlflow under run_id.

        For bagging-based models (e.g. random_forest) that support it, oob_score is turned
        on, so the out-of-bag R^2/RMSE can be logged.

        Args:
            data: Output of split_data
            model_type: One of ridge, lasso, random_forest, svr, hist_gradient_boosting,
                linear_regression
            run_id: mlflow run_id from setup_mlflow, so the fit params/model artifact land in
                the same run eval_model logs its metrics/predictions into
            params: Hyperparameters to set on the base model before fitting, e.g.
                {"n_estimators": 300} for random_forest. Uses sklearn's defaults if not given.

        Returns:
            The fit pipeline
        """
        base_model = self.get_base_model(model_type)
        base_model.set_params(**(params or {}))

        model_params = base_model.get_params()
        if "oob_score" in model_params and model_params.get("bootstrap", True):
            base_model.set_params(oob_score=True)

        pipeline = self.create_pipeline(base_model)
        pipeline.fit(data["X_train"], data["y_train"])

        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(base_model.get_params())

            if getattr(base_model, "oob_score_", None) is not None:
                mlflow.log_metric("oob_r2", base_model.oob_score_)

                oob_pred = base_model.oob_prediction_
                has_oob_pred = ~np.isnan(oob_pred)
                num_no_oob_pred = (~has_oob_pred).sum()
                if num_no_oob_pred > 0:
                    logger.warning(
                        f"{num_no_oob_pred}/{len(oob_pred)} training rows had no out-of-bag "
                        "prediction; excluding them from oob_rmse"
                    )

                if has_oob_pred.any():
                    oob_rmse = np.sqrt(mean_squared_error(data["y_train"][has_oob_pred], oob_pred[has_oob_pred]))
                    print(f"OOB RMSE: {oob_rmse}")
                    mlflow.log_metric("oob_rmse", oob_rmse)
                else:
                    logger.warning("No valid out-of-bag predictions available; skipping oob_rmse")

            signature = infer_signature(data["X_train"], data["y_train"])
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                registered_model_name=f"{self.target}_{model_type}",
                name=model_type,
                input_example=data["X_train"],
                signature=signature)

        return pipeline

    def eval_model(self, pipeline: Pipeline, data: dict[str, pd.DataFrame], run_id: str) -> pd.DataFrame:
        """
        Scores a fitted pipeline against the eval set. Logs R^2, RMSE, and the full dataset of
        predictions vs actual for the eval set..

        Args:
            pipeline: A fit pipeline.
            data: Output of split_data
            run_id: The mlflow run_id returned by fit_model, so eval metrics land in the same
                run as the fit params/model artifact
        """
        y_pred = pipeline.predict(data["X_eval"])

        preds_df = data["identity_eval"].copy()
        preds_df["predictions"] = y_pred
        preds_df["actual"] = data["y_eval"]
        preds_df = preds_df.sort_values(by=["target_season", "predictions", "actual"], ascending=False)

        with mlflow.start_run(run_id=run_id):
            score = pipeline.score(data["X_eval"], data["y_eval"])
            print(f"R^2 score: {score}")
            mlflow.log_metric("r2", score)

            rmse = np.sqrt(mean_squared_error(data["y_eval"], y_pred))
            print(f"RMSE: {rmse}")
            mlflow.log_metric("rmse", rmse)

            output_df = preds_df.rename(columns={"predictions": self.target})
            output_df[self.target] = output_df[self.target].round(2)

            csv_path = os.path.join(self.predictions_dir, f"{self.target}_eval_predictions_{run_id}.csv")
            output_df[["player_display_name", "target_season", self.target, "actual"]].to_csv(csv_path, index=False)

            mlflow.log_artifact(csv_path, "eval_predictions")

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
            search_value, run_id, oob_rmse, eval_rmse, eval_r2.
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
                results.append({
                    "search_param": key,
                    "search_value": value,
                    "run_id": child_run_id,
                    "oob_rmse": metrics.get("oob_rmse"),
                    "eval_rmse": metrics.get("rmse"),
                    "eval_r2": metrics.get("r2"),
                })

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
        "--target",
        type=str,
        default="fantasy_points_ppr",
        help="Which target's training set to load, gold_dir/{target}__training_set.csv (default: fantasy_points_ppr)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        help="Model type to fit, one of: ridge, lasso, random_forest, svr, hist_gradient_boosting, linear_regression (default: random_forest)"
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        nargs="+",
        default=None,
        help="Feature names to exclude from the feature set used to train/eval. Entries "
             "ending in '*' are treated as prefixes, e.g. 'receiving_*' excludes any feature "
             "starting with 'receiving_'. default: none excluded)"
    )
    parser.add_argument(
        "--eval-data-years",
        type=int,
        default=1,
        help="Number of most recent seasons (by target_season) not already claimed by the "
             "test set to hold out for eval (default: 1)"
    )
    parser.add_argument(
        "--test-data-years",
        type=int,
        default=1,
        help="Number of most recent seasons (by target_season) to hold out for test (default: 1)"
    )
    parser.add_argument(
        "--num-training-seasons",
        type=int,
        default=None,
        help="Number of most recent seasons (immediately preceding the eval set) to keep for "
             "training; older seasons are dropped entirely. (default: None)"
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

    model = TabularModel(
        data_dir=args.data_dir,
        tracking_dir=args.tracking_dir,
        target=args.target,
        excluded_features=args.exclude_features,
    )
    data = model.split_data(
        eval_data_years=args.eval_data_years,
        test_data_years=args.test_data_years,
        num_training_seasons=args.num_training_seasons,
    )

    if args.param_grid:
        param_grid = json.loads(args.param_grid)
        model.param_search(data, args.model_type, param_grid)
    else:
        split_params = {
            "excluded_features": args.exclude_features,
            "eval_data_years": args.eval_data_years,
            "test_data_years": args.test_data_years,
            "num_training_seasons": args.num_training_seasons if args.num_training_seasons is not None else "all",
        }
        run_id = model.setup_mlflow(args.model_type, extra_params=split_params)
        pipeline = model.fit_model(data, args.model_type, run_id)
        model.eval_model(pipeline, data, run_id)


if __name__ == "__main__":
    main()
