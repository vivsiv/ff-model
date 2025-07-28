import os
import pandas as pd
# import mlflow
# from mlflow.models import infer_signature
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

    def create_grid_search(self, pipeline: Pipeline, data: dict[str, pd.DataFrame]) -> GridSearchCV:
        # TODO: incorporate hyper parameters for the better models.
        param_grid = {
            # 'select__k': [100, 150, 'all'], # Lower feature counts perform worse, skew higher
            'model': [
                LinearRegression(),
                Ridge(),
                Lasso(),
                RandomForestRegressor(),
                SVR(), # Performs poorly compared to other models, no further tuning needed.
                HistGradientBoostingRegressor(),
            ]
        }
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1, # Uses all available cores.
        )
        grid_search.fit(data["X_train"], data["y_train"])
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
    ff_model = FantasyModel()

    # mlflow.set_tracking_uri("http://localhost:8000")
    # mlflow.set_experiment("ppr_total_linear_regression")
    # mlflow.autolog()

    data = ff_model.split_data()

    eval_pipeline = ff_model.create_pipeline()
    grid_search = ff_model.create_grid_search(eval_pipeline)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    print(f"Best estimator: {grid_search.best_estimator_}")

    best_model = grid_search.best_estimator_
    best_model_pipeline = ff_model.create_pipeline(best_model)
    score, preds = ff_model.make_test_predictions(best_model_pipeline, data)

    print(f"Best Model R^2 Score: {score}")

    # with mlflow.start_run():
    #     mlflow.log_param("target", model.target_col)
    #     mlflow.log_param("features", model.feature_cols)
    #     mlflow.log_param("r2_score", score)
    #     # log the model type and params?
    #     mlflow.log_params(estimator.get_params())

    #     signature = infer_signature(data["X_train"], preds)
    #     mlflow.sklearn.log_model(
    #         sk_model=estimator,
    #         registered_model_name="ppr_total_linear_regression",
    #         artifact_path="model",
    #         signature=signature,
    #         input_example=data["X_train"],
    #     )

    view_year = 2024
    print(f"Predictions for {ff_model.target_col} in {view_year}:")
    print(ff_model.view_year_test_predictions(preds, view_year))

    preds.to_csv(os.path.join(ff_model.data_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
