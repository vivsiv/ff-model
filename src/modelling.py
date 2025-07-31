import os
import logging
from typing import Tuple
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
from sklearn.model_selection import train_test_split, GridSearchCV
from typing import Any

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

    def __init__(
            self,
            data_dir: str = "../data",
            target_col: str = "ppr_fantasy_points",
            possible_targets: list[str] = [
                "ppr_fantasy_points",
                "standard_fantasy_points",
                "ppr_fantasy_points_per_game",
                "standard_fantasy_points_per_game",
                "value_over_replacement"
            ]
    ):
        self.data_dir = data_dir
        self.gold_data_dir = os.path.join(data_dir, "gold")
        self.training_data, self.live_data = self.load_data()

        self.tracking_dir = os.path.join(data_dir, "mlruns")
        os.makedirs(self.tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_dir)

        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

        self.id_col = "id"
        if target_col not in possible_targets:
            raise ValueError(f"Target column {target_col} not in {possible_targets}")
        self.target_col = target_col

        self.feature_cols = [col for col in self.training_data.columns if col not in [self.id_col] + possible_targets]

        self.train_ids = self.training_data[self.id_col]
        self.train_features = self.training_data[self.feature_cols]
        self.train_target = self.training_data[self.target_col]

        self.live_ids = self.live_data[self.id_col]
        self.live_features = self.live_data[self.feature_cols]

    def load_data(self) -> pd.DataFrame:
        train_data = pd.read_csv(os.path.join(self.gold_data_dir, "training_set.csv"))
        logger.info(f"Loaded training data: {len(train_data)} rows")

        live_data = pd.read_csv(os.path.join(self.gold_data_dir, "live_set.csv"))
        logger.info(f"Loaded live data: {len(live_data)} rows")

        return train_data, live_data

    def split_data(self) -> dict[str, pd.DataFrame]:
        X_train, X_test, y_train, y_test, Id_train, Id_test = train_test_split(
            self.train_features, self.train_target, self.train_ids, test_size=0.2, random_state=42
        )
        data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "Id_train": Id_train,
            "Id_test": Id_test
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

        mlflow.set_experiment(self.target_col)
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

        mlflow.set_experiment(self.target_col)
        run_name = f"{model_type}_tuning_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("phase", "tuning")

            self.log_grid_search_to_mlflow(grid_search)

            signature = infer_signature(data["X_train"], data["y_train"])
            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                registered_model_name=f"{self.target_col}_{model_type}",
                name=model_type,
                input_example=data["X_train"],
                signature=signature)

        return grid_search

    def load_model(self, model_type: str, model_version: int = None) -> Tuple[Pipeline, int]:
        if model_version is None:
            client = MlflowClient()
            latest_version = client.get_latest_versions(f"{self.target_col}_{model_type}", stages=["None"])[0].version
            model_version = latest_version
        else:
            model_version = model_version

        pipeline = mlflow.sklearn.load_model(f"models:/{self.target_col}_{model_type}/{model_version}")

        return pipeline, model_version

    def view_year_test_predictions(self, preds_df: pd.DataFrame, year: int) -> pd.DataFrame:
        preds_df["year"] = preds_df["id"].str.split("_").str[-1].astype(int)
        preds_df = preds_df[preds_df["year"] == year].sort_values(by=["predictions", "actual"], ascending=False)

        return preds_df.drop(columns=["year"])

    def make_test_predictions(self, data: dict[str, pd.DataFrame], model_type: str, model_version: int = None, log_year: int = 2024) -> pd.DataFrame:
        pipeline, model_version = self.load_model(model_type, model_version)

        y_pred = pipeline.predict(data["X_test"])
        mlflow.set_experiment(self.target_col)

        preds_df = pd.DataFrame({
            "id": data["Id_test"],
            "predictions": y_pred,
            "actual": data["y_test"]
        })

        with mlflow.start_run(run_name=f"test_{model_type}_{model_version}"):
            mlflow.set_tag("phase", "test")

            mlflow.log_param("model_name", f"{self.target_col}_{model_type}_v{model_version}")

            score = pipeline.score(data["X_test"], data["y_test"])
            print(f"R^2 score: {score}")
            mlflow.log_metric("r2", score)

            rmse = np.sqrt(mean_squared_error(data["y_test"], y_pred))
            print(f"RMSE: {rmse}")
            mlflow.log_metric("rmse", rmse)

            log_year_preds_df = self.view_year_test_predictions(preds_df, log_year)
            log_year_preds_df["player"] = log_year_preds_df["id"].str.split("_").str[:-1].str.join("_")
            log_year_preds_df["year"] = log_year_preds_df["id"].str.split("_").str[-1].astype(int)
            log_year_preds_df.sort_values(by="predictions", ascending=False, inplace=True)
            log_year_preds_df.rename(columns={"predictions": self.target_col}, inplace=True)
            log_year_preds_df[self.target_col] = log_year_preds_df[self.target_col].round(2)
            log_year_preds_df.drop(columns=["id"], inplace=True)

            csv_path = os.path.join(self.predictions_dir, f"{self.target_col}_{log_year}_predictions.csv")
            log_year_preds_df[["player", "year", self.target_col, "actual"]].to_csv(csv_path, index=False)

            mlflow.log_artifact(csv_path, f"test_predictions_{log_year}")

        return preds_df

    def make_live_predictions(self, data: dict[str, pd.DataFrame], model_type: str, model_version: int = None) -> pd.DataFrame:
        pipeline, model_version = self.load_model(model_type, model_version)

        y_pred = pipeline.predict(self.live_features)

        preds_df = pd.DataFrame({
            "id": self.live_ids,
            "predictions": y_pred,
        })

        preds_df["player"] = preds_df["id"].str.split("_").str[:-1].str.join("_")
        preds_df["position"] = preds_df["id"].str.split("_").str[-1]
        preds_df.sort_values(by="predictions", ascending=False, inplace=True)
        preds_df.rename(columns={"predictions": self.target_col}, inplace=True)
        preds_df[self.target_col] = preds_df[self.target_col].round(2)
        preds_df.drop(columns=["id"], inplace=True)

        csv_path = os.path.join(self.predictions_dir, f"{self.target_col}_live_predictions.csv")
        preds_df[["player", "position", self.target_col]].to_csv(csv_path, index=False)

        with mlflow.start_run(run_name=f"live_{model_type}_v{model_version}"):
            mlflow.set_tag("phase", "live")
            mlflow.log_param("model_name", f"{self.target_col}_{model_type}_v{model_version}")

            mlflow.log_artifact(csv_path, "predictions")

        return preds_df


def main():
    ppr_model = FantasyModel(target_col="ppr_fantasy_points")
    data = ppr_model.split_data()

    ppr_model.run_model_eval(data)

    chosen_model_type = "ridge"

    ppr_model.run_model_tuning(data, chosen_model_type)
    # ppr_model.run_model_tuning(data, "random_forest")
    # ppr_model.run_model_tuning(data, "svr")
    # ppr_model.run_model_tuning(data, "hist_gradient_boosting")

    test_preds_df = ppr_model.make_test_predictions(data, chosen_model_type)

    view_year = 2024
    print(f"Predictions for {ppr_model.target_col} in {view_year}:")
    print(ppr_model.view_year_test_predictions(test_preds_df, view_year))

    live_preds_df = ppr_model.make_live_predictions(data, chosen_model_type)
    print(live_preds_df)


if __name__ == "__main__":
    main()
