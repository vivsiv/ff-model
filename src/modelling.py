import os
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
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
            # ('select', SelectKBest(f_regression, k='all')), # All features performs the best with f_regression.
            # ('select', SelectKBest(mutual_info_regression, k='all')), All features performs the best.
            ('model', model)
        ])

        return pipeline

    def run_model_eval(self, data: dict[str, pd.DataFrame]) -> GridSearchCV:
        eval_pipeline = self.create_pipeline()

        # TODO: incorporate hyper parameters for the better models.
        param_grid = {
            # 'select__k': [100, 150, 'all'], # Lower feature counts perform worse, skew higher
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
        grid_search.fit(data["X_train"], data["y_train"])

        mlflow.set_experiment(self.target_col)
        run_name = f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("phase", "eval")

            # TODO: if feature selection is used, log the features used
            mlflow.log_param("features", self.feature_cols)
            mlflow.log_param("param_grid", grid_search.param_grid)

            mlflow.log_metric("r2", grid_search.best_score_)
            best_rmse = -grid_search.cv_results_['mean_test_rmse'][grid_search.best_index_]
            mlflow.log_metric("rmse", best_rmse)

            mlflow.log_param("best_params", str(grid_search.best_params_))

            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            cv_results_df['mean_test_rmse'] = -cv_results_df['mean_test_rmse']
            cv_results_log = (
                cv_results_df[['param_model', 'mean_test_r2', 'mean_test_rmse', 'std_test_r2']]
                .sort_values(by=['mean_test_r2', 'mean_test_rmse'], ascending=[False, True])
                .to_dict(orient='records')
            )
            mlflow.log_param("cv_results", cv_results_log)

            signature = infer_signature(data["X_train"], data["y_train"])
            best_model = grid_search.best_estimator_.named_steps['model']
            model_name = type(best_model).__name__.lower()

            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                registered_model_name=f"{self.target_col}_{model_name}",
                name=model_name,
                input_example=data["X_train"],
                signature=signature)

        return grid_search

    def make_test_predictions(self, model: Any, data: dict[str, pd.DataFrame]) -> tuple[float, pd.DataFrame]:
        pipeline = self.create_pipeline(model)
        pipeline.fit(data["X_train"], data["y_train"])

        score = pipeline.score(data["X_test"], data["y_test"])

        y_pred = pipeline.predict(data["X_test"])
        preds = pd.DataFrame({
            "id": data["Id_test"],
            "predictions": y_pred,
            "actual": data["y_test"]
        })

        return score, preds

    def view_year_test_predictions(self, preds_df: pd.DataFrame, year: int) -> pd.DataFrame:
        preds_df["year"] = preds_df["id"].str.split("_").str[-2].astype(int)
        preds_df = preds_df[preds_df["year"] == year].sort_values(by=["predictions", "actual"], ascending=False)

        preds_df.to_csv(os.path.join(self.predictions_dir, f"{self.target_col}_{year}_predictions.csv"), index=False)

        return preds_df


def main():
    ppr_model = FantasyModel(target_col="ppr_fantasy_points")
    data = ppr_model.split_data()

    ppr_model.run_model_eval(data)

    # score, preds = ppr_model.make_test_predictions(data)
    # print(f"Test set R^2 Score: {score}")

    # view_year = 2024
    # print(f"Predictions for {ppr_model.target_col} in {view_year}:")
    # print(ppr_model.view_year_test_predictions(preds, view_year))


if __name__ == "__main__":
    main()
