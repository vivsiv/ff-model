import os
import logging
import argparse
from typing import Tuple

import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline

from src.modeling.utils import load_mlflow_model, setup_mlflow_run
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
    (gold_dir/{target}__prediction_set.csv)."""

    def __init__(
            self,
            data_dir: str,
            tracking_dir: str,
            target: str = "fantasy_points_ppr",
    ):
        self.data_dir = data_dir
        self.gold_dir = os.path.join(data_dir, "gold")
        self.target = target

        self.tracking_dir = tracking_dir

        self.predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(self.predictions_dir, exist_ok=True)

    def load_model(self, model_type: str, model_version: int = None) -> Tuple[Pipeline, int]:
        pipeline, model_version, _ = load_mlflow_model(self.target, model_type, model_version, tracking_dir=self.tracking_dir)
        return pipeline, model_version

    def load_prediction_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads gold_dir/{target}__prediction_set.csv and splits it into identity and feature columns.

        Returns:
            (identity_df, features_df)
        """
        filename = f"{self.target}__prediction_set.csv"
        data = pd.read_csv(os.path.join(self.gold_dir, filename))
        logger.info(f"Loaded prediction data: {len(data)} rows")

        identity_cols = get_identity_columns("nflverse", "player_stats") + ["target_season"]
        feature_cols = [col for col in data.columns if col not in identity_cols + [TARGET_COL]]

        return data[identity_cols], data[feature_cols]

    def make_predictions(self, model_type: str, model_version: int = None) -> pd.DataFrame:
        identity_df, features_df = self.load_prediction_data()
        pipeline, model_version = self.load_model(model_type, model_version)

        y_pred = pipeline.predict(features_df)

        preds_df = identity_df.copy()
        preds_df["predictions"] = y_pred
        preds_df.sort_values(by="predictions", ascending=False, inplace=True)
        preds_df.rename(columns={"predictions": self.target}, inplace=True)
        preds_df[self.target] = preds_df[self.target].round(2)

        csv_path = os.path.join(self.predictions_dir, f"{self.target}_{model_type}_v{model_version}_predictions.csv")
        preds_df.to_csv(csv_path, index=False)

        run_id = setup_mlflow_run(
            experiment_name=f"{self.target}_tabular",
            run_name=f"{model_type}_v{model_version}_predictions",
            tracking_dir=self.tracking_dir,
            tags={"model_type": model_type, "phase": "predict"},
        )

        with mlflow.start_run(run_id=run_id):
            mlflow.log_param("model_name", f"{self.target}_{model_type}_v{model_version}")
            mlflow.log_artifact(csv_path, "predictions")

        return preds_df


def main():
    parser = argparse.ArgumentParser(
        description="Reports live predictions for the upcoming season from a trained model"
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
        help="Which target's prediction set to use, gold_dir/{target}__prediction_set.csv (default: fantasy_points_ppr)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        help="Registered model type to load, one of: ridge, lasso, random_forest, svr, hist_gradient_boosting, linear_regression (default: random_forest)"
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Specific model version to load for inference"
    )

    args = parser.parse_args()

    reporter = PredictionReporter(data_dir=args.data_dir, tracking_dir=args.tracking_dir, target=args.target)
    live_preds_df = reporter.make_predictions(model_type=args.model_type, model_version=args.model_version)

    print(live_preds_df)


if __name__ == "__main__":
    main()
