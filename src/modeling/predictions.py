import os
import logging
import argparse
from typing import Tuple

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.processing.column_registry import get_identity_columns
from src.processing.gold import TARGET_COL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PredictionReporter:
    """Loads a trained model and reports live predictions for the upcoming season
    (gold_dir/{target}__prediction_set.csv). Doesn't build, tune, or evaluate models itself --
    see FantasyModel for that."""

    def __init__(
            self,
            data_dir: str = "../data",
            target: str = "fantasy_points_ppr",
    ):
        self.data_dir = data_dir
        self.gold_dir = os.path.join(data_dir, "gold")

        # Same role as in FantasyModel: a descriptive label for naming mlflow runs/registered
        # models/output files, and which gold_dir/{target}__prediction_set.csv to load.
        self.target = target

        self.tracking_dir = os.path.join(data_dir, "mlruns")
        os.makedirs(self.tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_dir)

        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

    def load_model(self, model_type: str, model_version: int = None) -> Tuple[Pipeline, int]:
        if model_version is None:
            client = MlflowClient()
            latest_version = client.get_latest_versions(f"{self.target}_{model_type}", stages=["None"])[0].version
            model_version = latest_version
        else:
            model_version = model_version

        pipeline = mlflow.sklearn.load_model(f"models:/{self.target}_{model_type}/{model_version}")

        return pipeline, model_version

    def load_prediction_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads gold_dir/{target}__prediction_set.csv and splits it into identity and feature
        columns, the same way FantasyModel splits the training set.

        Returns:
            (identity_df, features_df)
        """
        filename = f"{self.target}__prediction_set.csv"
        data = pd.read_csv(os.path.join(self.gold_dir, filename))
        logger.info(f"Loaded prediction data: {len(data)} rows")

        identity_cols = get_identity_columns("nflverse", "player_stats") + ["target_season"]
        feature_cols = [col for col in data.columns if col not in identity_cols + [TARGET_COL]]

        return data[identity_cols], data[feature_cols]

    def make_live_predictions(self, model_type: str, model_version: int = None) -> pd.DataFrame:
        identity_df, features_df = self.load_prediction_data()
        pipeline, model_version = self.load_model(model_type, model_version)

        y_pred = pipeline.predict(features_df)

        preds_df = identity_df.copy()
        preds_df["predictions"] = y_pred
        preds_df.sort_values(by="predictions", ascending=False, inplace=True)
        preds_df.rename(columns={"predictions": self.target}, inplace=True)
        preds_df[self.target] = preds_df[self.target].round(2)

        csv_path = os.path.join(self.predictions_dir, f"{self.target}_live_predictions.csv")
        preds_df.to_csv(csv_path, index=False)

        with mlflow.start_run(run_name=f"live_{model_type}_v{model_version}"):
            mlflow.set_tag("phase", "live")
            mlflow.log_param("model_name", f"{self.target}_{model_type}_v{model_version}")

            mlflow.log_artifact(csv_path, "predictions")

        return preds_df


def main():
    parser = argparse.ArgumentParser(
        description="Reports live predictions for the upcoming season from a trained model"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="fantasy_points_ppr",
        help="Which target's prediction set to use, gold_dir/{target}__prediction_set.csv (default: fantasy_points_ppr)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        help="Registered model type to load, one of: ridge, lasso, random_forest, svr, hist_gradient_boosting, linear_regression (default: random_forest)"
    )

    args = parser.parse_args()

    reporter = PredictionReporter(target=args.target)
    live_preds_df = reporter.make_live_predictions(args.model_type)
    print(live_preds_df)


if __name__ == "__main__":
    main()
