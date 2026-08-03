import os
import logging
import argparse
from datetime import datetime

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
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
from sklearn.model_selection import GridSearchCV
from typing import Any, Tuple

from src.processing.column_registry import get_identity_columns
from src.processing.gold import TARGET_COL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FantasyModel:
    """Loads the gold training set, splits it, and builds/tunes/logs models. Doesn't make or
    report predictions itself -- see PredictionReporter for that."""

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

    def split_data(self, eval_data_years: int = 1) -> dict[str, pd.DataFrame]:
        """
        Splits chronologically by target_season, the most recent eval_data_years worth of 
        seasons become the eval set, everything before that is training data.
        This guarantees the eval set is strictly in the future relative to training.

        Args:
            eval_data_years: Number of seasons to hold out for eval (default: 1)

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

    def create_model_grid_search(self) -> GridSearchCV:
        eval_pipeline = self.create_pipeline()
        param_grid = {
            'model': [
                LinearRegression(),
                Ridge(),
                Lasso(),
                RandomForestRegressor(),
                SVR(),
                HistGradientBoostingRegressor(),
            ]
        }

        grid_search = GridSearchCV(
            eval_pipeline,
            param_grid,
            cv=5,
            scoring={
                'r2': 'r2',
                'rmse': 'neg_root_mean_squared_error',
            },
            refit='r2',  # Use R2 for selecting best model
            n_jobs=-1,  # Uses all available cores.
        )

        return grid_search

    def log_grid_search_to_mlflow(self, grid_search: GridSearchCV) -> None:
        mlflow.log_param("input", grid_search.param_grid)
        mlflow.log_param("best", str(grid_search.best_params_))

        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_results_df['mean_test_rmse'] = -cv_results_df['mean_test_rmse']
        cv_results_cols = [f"param_{key}" for key in grid_search.param_grid.keys()] + ['mean_test_r2', 'mean_test_rmse', 'std_test_r2']
        cv_results_log = (
            cv_results_df[cv_results_cols]
            .sort_values(by=['mean_test_r2', 'mean_test_rmse'], ascending=[False, True])
            .to_dict(orient='records')
        )
        mlflow.log_param("results", cv_results_log)

        mlflow.log_metric("r2", grid_search.best_score_)
        mlflow.log_metric("r2_std", grid_search.cv_results_['std_test_r2'][grid_search.best_index_])
        mlflow.log_metric("rmse", -grid_search.cv_results_['mean_test_rmse'][grid_search.best_index_])
        mlflow.log_metric("rmse_std", grid_search.cv_results_['std_test_rmse'][grid_search.best_index_])

    def run_model_eval(self, data: dict[str, pd.DataFrame]) -> GridSearchCV:
        grid_search = self.create_model_grid_search()

        grid_search.fit(data["X_train"], data["y_train"])

        mlflow.set_experiment(self.target)
        run_name = f"model_eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("phase", "eval")

            self.log_grid_search_to_mlflow(grid_search)

        return grid_search

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

    def get_param_grid(self, model_type: str) -> dict[str, list[Any]]:
        master_param_grid = {
            'ridge': {
                'model__alpha': np.logspace(-4, 4, 10),
            },
            'lasso': {
                'model__alpha': np.logspace(-4, 4, 10),
            },
            'random_forest': {
                'model__n_estimators': [200, 300, 400],
                'model__max_depth': [10, 15],
                'model__min_samples_split': [5],
                'model__min_samples_leaf': [2],
            },
            'svr': {
                'model__C': np.logspace(-4, 4, 10),
                'model__kernel': ['linear', 'rbf'],
                'model__gamma': ['scale', 'auto'],
            },
            'hist_gradient_boosting': {
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7, 9],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
            }
        }

        return master_param_grid[model_type]

    def run_model_tuning(self, data: dict[str, pd.DataFrame], model_type: str) -> GridSearchCV:
        pipeline = self.create_pipeline(self.get_base_model(model_type))
        param_grid = self.get_param_grid(model_type)

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring={
                'r2': 'r2',
                'rmse': 'neg_root_mean_squared_error',
            },
            refit='r2',
            n_jobs=-1
        )
        grid_search.fit(data["X_train"], data["y_train"])

        mlflow.set_experiment(self.target)
        run_name = f"{model_type}_tuning_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("phase", "tuning")

            self.log_grid_search_to_mlflow(grid_search)

            signature = infer_signature(data["X_train"], data["y_train"])
            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                registered_model_name=f"{self.target}_{model_type}",
                name=model_type,
                input_example=data["X_train"],
                signature=signature)

        return grid_search

    def load_model(self, model_type: str, model_version: int = None) -> Tuple[Pipeline, int]:
        if model_version is None:
            client = MlflowClient()
            latest_version = client.get_latest_versions(f"{self.target}_{model_type}", stages=["None"])[0].version
            model_version = latest_version
        else:
            model_version = model_version

        pipeline = mlflow.sklearn.load_model(f"models:/{self.target}_{model_type}/{model_version}")

        return pipeline, model_version

    def view_year_test_predictions(self, preds_df: pd.DataFrame, year: int) -> pd.DataFrame:
        return (
            preds_df[preds_df["target_season"] == year]
            .sort_values(by=["predictions", "actual"], ascending=False)
        )

    def eval_model(self, data: dict[str, pd.DataFrame], model_type: str, model_version: int = None, log_year: int = 2024) -> pd.DataFrame:
        """
        Loads a registered model and scores it against the test set: R2/RMSE logged to mlflow,
        plus a per-row predictions-vs-actual csv logged as an mlflow artifact for log_year.
        """
        pipeline, model_version = self.load_model(model_type, model_version)

        y_pred = pipeline.predict(data["X_test"])
        mlflow.set_experiment(self.target)

        preds_df = data["identity_test"].copy()
        preds_df["predictions"] = y_pred
        preds_df["actual"] = data["y_test"]

        with mlflow.start_run(run_name=f"test_{model_type}_v{model_version}"):
            mlflow.set_tag("phase", "test")

            mlflow.log_param("model_name", f"{self.target}_{model_type}_v{model_version}")

            score = pipeline.score(data["X_test"], data["y_test"])
            print(f"R^2 score: {score}")
            mlflow.log_metric("r2", score)

            rmse = np.sqrt(mean_squared_error(data["y_test"], y_pred))
            print(f"RMSE: {rmse}")
            mlflow.log_metric("rmse", rmse)

            log_year_preds_df = self.view_year_test_predictions(preds_df, log_year)
            log_year_preds_df = log_year_preds_df.rename(columns={"predictions": self.target})
            log_year_preds_df[self.target] = log_year_preds_df[self.target].round(2)

            csv_path = os.path.join(self.predictions_dir, f"{self.target}_{log_year}_predictions.csv")
            log_year_preds_df[["player_display_name", "target_season", self.target, "actual"]].to_csv(csv_path, index=False)

            mlflow.log_artifact(csv_path, f"test_predictions_{log_year}")

        return preds_df


def main():
    parser = argparse.ArgumentParser(
        description="Builds/tunes/logs a tabular model against a gold training set"
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
        help="Model type to tune, one of: ridge, lasso, random_forest, svr, hist_gradient_boosting, linear_regression (default: random_forest)"
    )

    args = parser.parse_args()

    model = FantasyModel(target=args.target)
    data = model.split_data()

    model.run_model_eval(data)
    model.run_model_tuning(data, args.model_type)
    model.eval_model(data, args.model_type)


if __name__ == "__main__":
    main()
