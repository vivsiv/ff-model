import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Any


class FantasyModel:

    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.gold_data_dir = os.path.join(data_dir, "gold")

    def load_gold_table(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.gold_data_dir, "final_data.csv"))

    def create_pipeline(self, model: Any) -> Pipeline:
        """
        Creates a preprocessing pipeline.
        """
        preprocessing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])
        model_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', model)
        ])
        return model_pipeline


def main():
    model = FantasyModel()
    gold_table = model.load_gold_table()

    metadata_cols = ["player", "year", "team"]
    target_cols = ["ppr_fantasy_points", "standard_fantasy_points", "ppr_fantasy_points_per_game", "standard_fantasy_points_per_game", "value_over_replacement"]
    feature_cols = [col for col in gold_table.columns if col not in metadata_cols + target_cols]

    metadata = gold_table[metadata_cols]
    X = gold_table[feature_cols]
    y = gold_table["ppr_fantasy_points"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = model.create_pipeline(LinearRegression())

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    score = pipeline.score(y_pred, y_test)
    print(f"Score: {score}")

    preds = pd.DataFrame({
        "player": metadata["player"],
        "year": metadata["year"],
        "team": metadata["team"],
        "ppr_fantasy_points": y_pred,
    })

    print(preds)

    preds.to_csv(os.path.join(model.data_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
