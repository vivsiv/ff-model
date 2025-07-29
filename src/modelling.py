import os
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
# from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Any
from datetime import datetime


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
        self.gold_data = self.load_gold_table()

        self.tracking_dir = os.path.join(data_dir, "mlruns")
        os.makedirs(self.tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_dir)

        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

        self.id_col = "id"
        if target_col not in possible_targets:
            raise ValueError(f"Target column {target_col} not in {possible_targets}")
        self.target_col = target_col

        self.feature_cols = [col for col in self.gold_data.columns if col not in [self.id_col] + possible_targets]

        self.Id = self.gold_data[self.id_col]
        self.X = self.gold_data[self.feature_cols]
        self.Y = self.gold_data[self.target_col]

    def load_gold_table(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.gold_data_dir, "final_stats.csv"))

    def split_data(self) -> dict[str, pd.DataFrame]:
        X_train, X_test, y_train, y_test, Id_train, Id_test = train_test_split(
            self.X, self.Y, self.Id, test_size=0.2, random_state=42
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

    def run_model_eval(self, data: dict[str, pd.DataFrame]) -> GridSearchCV:
        grid_search = self.create_model_grid_search()

        grid_search.fit(data["X_train"], data["y_train"])

        mlflow.set_experiment(self.target_col)
        run_name = f"model_eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("phase", "eval")

            mlflow.log_param("param_grid", grid_search.param_grid)

            mlflow.log_metric("r2", grid_search.best_score_)
            mlflow.log_metric("r2_std", grid_search.cv_results_['std_test_r2'][grid_search.best_index_])
            mlflow.log_metric("rmse", -grid_search.cv_results_['mean_test_rmse'][grid_search.best_index_])
            mlflow.log_metric("rmse_std", grid_search.cv_results_['std_test_rmse'][grid_search.best_index_])

            mlflow.log_param("best_params", str(grid_search.best_params_))

            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            cv_results_df['mean_test_rmse'] = -cv_results_df['mean_test_rmse']
            cv_results_log = (
                cv_results_df[['param_model', 'mean_test_r2', 'mean_test_rmse', 'std_test_r2']]
                .sort_values(by=['mean_test_r2', 'mean_test_rmse'], ascending=[False, True])
                .to_dict(orient='records')
            )
            mlflow.log_param("cv_results", cv_results_log)

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
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7, 9],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
            },
            'svr': {
                'model__C': np.logspace(-4, 4, 10),
                'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
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

            mlflow.log_param("param_grid", param_grid)

            mlflow.log_metric("r2", grid_search.best_score_)
            mlflow.log_metric("r2_std", grid_search.cv_results_['std_test_r2'][grid_search.best_index_])
            mlflow.log_metric("rmse", -grid_search.cv_results_['mean_test_rmse'][grid_search.best_index_])
            mlflow.log_metric("rmse_std", grid_search.cv_results_['std_test_rmse'][grid_search.best_index_])

            mlflow.log_param("best_params", str(grid_search.best_params_))

            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            cv_results_cols = [f"param_{key}" for key in param_grid.keys()] + ['mean_test_r2', 'mean_test_rmse', 'std_test_r2']
            cv_results_log = (
                cv_results_df[cv_results_cols]
                .sort_values(by=['mean_test_r2', 'mean_test_rmse'], ascending=[False, True])
                .to_dict(orient='records')
            )
            mlflow.log_param("cv_results", cv_results_log)

            signature = infer_signature(data["X_train"], data["y_train"])
            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                registered_model_name=f"{self.target_col}_{model_type}",
                name=model_type,
                input_example=data["X_train"],
                signature=signature)

        return grid_search

    def make_test_predictions(self, data: dict[str, pd.DataFrame], model_type: str, model_version: int = None) -> pd.DataFrame:
        if model_version is None:
            client = MlflowClient()
            latest_version = client.get_latest_versions(f"{self.target_col}_{model_type}", stages=["None"])[0].version
            model_version = latest_version
        else:
            model_version = model_version

        pipeline = mlflow.sklearn.load_model(f"models:/{self.target_col}_{model_type}/{model_version}")

        y_pred = pipeline.predict(data["X_test"])
        mlflow.set_experiment(self.target_col)

        with mlflow.start_run(run_name=f"test_v{model_version}"):
            mlflow.set_tag("phase", "test")

            mlflow.log_param("model_name", f"{self.target_col}_{model_type}")
            mlflow.log_param("model_version", model_version)

            score = pipeline.score(data["X_test"], data["y_test"])
            print(f"R^2 score: {score}")
            mlflow.log_metric("r2", score)

            rmse = np.sqrt(mean_squared_error(data["y_test"], y_pred))
            print(f"RMSE: {rmse}")
            mlflow.log_metric("rmse", rmse)

            # TODO: log a final model?

        preds_df = pd.DataFrame({
            "id": data["Id_test"],
            "predictions": y_pred,
            "actual": data["y_test"]
        })

        return preds_df

    def view_year_test_predictions(self, preds_df: pd.DataFrame, year: int) -> pd.DataFrame:
        preds_df["year"] = preds_df["id"].str.split("_").str[-1].astype(int)
        preds_df = preds_df[preds_df["year"] == year].sort_values(by=["predictions", "actual"], ascending=False)

        preds_df.to_csv(os.path.join(self.predictions_dir, f"{self.target_col}_{year}_predictions.csv"), index=False)

        return preds_df.drop(columns=["year"])


def main():
    ppr_model = FantasyModel(target_col="ppr_fantasy_points")
    data = ppr_model.split_data()

    ppr_model.run_model_eval(data)

    chosen_model_type = "ridge"

    ppr_model.run_model_tuning(data, chosen_model_type)
    # ppr_model.run_model_tuning(data, "random_forest")
    # ppr_model.run_model_tuning(data, "svr")
    # ppr_model.run_model_tuning(data, "hist_gradient_boosting")

    preds_df = ppr_model.make_test_predictions(data, chosen_model_type)

    view_year = 2024
    print(f"Predictions for {ppr_model.target_col} in {view_year}:")
    print(ppr_model.view_year_test_predictions(preds_df, view_year))


if __name__ == "__main__":
    main()
