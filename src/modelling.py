import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Any


class FantasyModel:

    def __init__(
            self,
            data_dir: str = "../data",
            target_col: str = "ppr_fantasy_points",
        ):
        self.data_dir = data_dir
        self.gold_data_dir = os.path.join(data_dir, "gold")
        self.gold_data = self.load_gold_table()

        self.id_col = "id"
        possible_targets = [
            "ppr_fantasy_points",
            "standard_fantasy_points",
            "ppr_fantasy_points_per_game",
            "standard_fantasy_points_per_game",
            "value_over_replacement"
        ]
        if target_col not in possible_targets:
            raise ValueError(f"Target column {target_col} not in {possible_targets}")
        self.target_col = target_col

        self.feature_cols = [col for col in self.gold_data.columns if col not in [self.id_col] + possible_targets]

        self.Id = self.gold_data[self.id_col]
        self.X = self.gold_data[self.feature_cols]
        self.Y = self.gold_data[self.target_col]

    def load_gold_table(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.gold_data_dir, "final_data.csv"))

        
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
    
    def create_pipeline(self, model: Any) -> Pipeline:
        preprocessing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])
        model_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', model)
        ])

        return model_pipeline

    def run_pipeline(self, model: Any, data: dict[str, pd.DataFrame]) -> tuple[float, pd.DataFrame]:
        pipeline = self.create_pipeline(model)
        pipeline.fit(data["X_train"], data["y_train"])

        score = pipeline.score(data["X_test"], data["y_test"])

        y_pred = pipeline.predict(data["X_test"])
        preds = pd.DataFrame({
            "id": data["Id_test"],
            "ppr_fantasy_points": y_pred,
        })

        return score, preds

def main():
    model = FantasyModel()
    data = model.split_data()

    estimator = LinearRegression()
    score, preds = model.run_pipeline(estimator, data)

    print(f"R^2 Score: {score}")
    print(preds)

    preds.to_csv(os.path.join(model.data_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
