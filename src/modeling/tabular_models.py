import os
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
from sklearn.ensemble import HistGradientBoostingRegressor

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
            data_dir: str = "../data",
            target: str = "fantasy_points_ppr",
    ):
        self.data_dir = data_dir
        self.gold_data_dir = os.path.join(data_dir, "gold")
        self.target = target
        self.training_data = self.load_data()

        self.tracking_dir = os.path.join(data_dir, "mlruns")
        os.makedirs(self.tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_dir)

        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

        self.identity_cols = get_identity_columns("nflverse", "player_stats") + ["target_season"]
        self.feature_cols = [
            col for col in self.training_data.columns
            if col not in self.identity_cols + [TARGET_COL]
        ]

        self.identity_df = self.training_data[self.identity_cols]
        self.features_df = self.training_data[self.feature_cols]
        self.target_df = self.training_data[TARGET_COL]

    def load_data(self) -> pd.DataFrame:
        filename = f"{self.target}__training_set.csv"
        data = pd.read_csv(os.path.join(self.gold_data_dir, filename))
        logger.info(f"Loaded data: {len(data)} rows")

        return data

    def split_data(self, eval_data_years: int) -> dict[str, pd.DataFrame]:
        """
        Splits chronologically by target_season. The most recent eval_data_years worth of
        seasons become the eval set, everything before that is training data.
        This guarantees the eval set is strictly in the future relative to training.

        Args:
            eval_data_years: Number of seasons to hold out for eval

        Returns:
            dict with X_train/X_test, y_train/y_test, identity_train/identity_test
        """
        max_target_season = self.training_data["target_season"].max()
        eval_cutoff_season = max_target_season - eval_data_years + 1
        is_eval = self.training_data["target_season"] >= eval_cutoff_season

        data = {
            "X_train": self.features_df[~is_eval],
            "X_test": self.features_df[is_eval],
            "y_train": self.target_df[~is_eval],
            "y_test": self.target_df[is_eval],
            "identity_train": self.identity_df[~is_eval],
            "identity_test": self.identity_df[is_eval],
        }
        return data

    def create_pipeline(self, model: Any = LinearRegression()) -> Pipeline:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        return pipeline

    def get_base_model(self, model_type: str) -> Any:
        base_models = {
            'ridge': Ridge(),
            'lasso': Lasso(), # Lasso not performant for ppr_ppg
            'random_forest': RandomForestRegressor(),
            'svr': SVR(),
            'hist_gradient_boosting': HistGradientBoostingRegressor(),
            'linear_regression': LinearRegression(),
        }
        return base_models[model_type]

    def setup_mlflow(self, model_type: str) -> str:
        """
        Sets the active mlflow experiment to {target}_{model_type} and creates a new run for it.

        Args:
            model_type: e.g. "random_forest"

        Returns:
            run_id - The mlflow run to tie training and eval to.
        """
        mlflow.set_experiment(f"{self.target}_{model_type}")
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("model_type", model_type)
            run_id = run.info.run_id

        return run_id

    def fit_model(self, data: dict[str, pd.DataFrame], model_type: str, run_id: str, params: Optional[dict] = None) -> Pipeline:
        """
        Gets the base model for model_type, applies params (sklearn defaults if none given),
        fits it on the training split, and registers it to mlflow under run_id.

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

        pipeline = self.create_pipeline(base_model)
        pipeline.fit(data["X_train"], data["y_train"])

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
        y_pred = pipeline.predict(data["X_test"])

        preds_df = data["identity_test"].copy()
        preds_df["predictions"] = y_pred
        preds_df["actual"] = data["y_test"]
        preds_df = preds_df.sort_values(by=["target_season", "predictions", "actual"], ascending=False)

        with mlflow.start_run(run_id=run_id):
            score = pipeline.score(data["X_test"], data["y_test"])
            print(f"R^2 score: {score}")
            mlflow.log_metric("r2", score)

            rmse = np.sqrt(mean_squared_error(data["y_test"], y_pred))
            print(f"RMSE: {rmse}")
            mlflow.log_metric("rmse", rmse)

            output_df = preds_df.rename(columns={"predictions": self.target})
            output_df[self.target] = output_df[self.target].round(2)

            csv_path = os.path.join(self.predictions_dir, f"{self.target}_eval_predictions_{run_id}.csv")
            output_df[["player_display_name", "target_season", self.target, "actual"]].to_csv(csv_path, index=False)

            mlflow.log_artifact(csv_path, "eval_predictions")

        return preds_df


def main():
    parser = argparse.ArgumentParser(
        description="Fits and evaluates a tabular model against a gold training set"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Parent directory for the gold/mlruns/predictions layers (default: class default)"
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
        "--eval-data-years",
        type=int,
        default=2,
        help="Number of most recent seasons (by target_season) to hold out for eval (default: 2)"
    )

    args = parser.parse_args()

    kwargs = {"data_dir": args.data_dir} if args.data_dir is not None else {}
    model = TabularModel(target=args.target, **kwargs)
    data = model.split_data(eval_data_years=args.eval_data_years)

    run_id = model.setup_mlflow(args.model_type)
    pipeline = model.fit_model(data, args.model_type, run_id)
    model.eval_model(pipeline, data, run_id)


if __name__ == "__main__":
    main()
