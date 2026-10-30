import os
import logging
import argparse
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from src.modeling.data_builder import DATA_BUILDER_CONFIG_ARTIFACT_PATH, EVAL_SPLIT, TEST_SPLIT
from src.modeling.tabular.data_builder import TabularModelDataBuilder
from src.modeling.utils import load_data_builder_config, load_mlflow_model, set_mlflow_tracking_uri, setup_mlflow_run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TabularModelInference:
    """
    Predicts and possibly scores a fitted tabular pipeline against a dataset built from a TabularModelDataBuilder,
    logging results to mlflow. Two methods, split by whether ground truth target values are available to score against:
      - evaluate(run_id, csv_path, split): predicts then scores against the ground truth values in the provided split.
      - predict(run_id, csv_path): predicts against live data.

    Two ways to build:
      - from_fit_pipeline: wraps an already-fit pipeline + in-memory TabularModelDataBuilder.
      - from_registry: loads a registered model from mlflow plus the TabularModelDataBuilder config it was trained with.
        - caveat: assumes the training set hasn't fundamentally changed since the model was trained.
    """

    def __init__(
            self,
            pipeline: Pipeline,
            data_builder: TabularModelDataBuilder,
            tracking_dir: Optional[str] = None,
            registered_model: Optional[str] = None,
            model_version: Optional[int] = None,
            source_run_id: Optional[str] = None,
    ):
        """
        Prefer from_fit_pipeline/from_registry over calling this directly.

        Args:
            pipeline: A fit pipeline to predict with.
            data_builder: The TabularModelDataBuilder the pipeline was (or will be) fit under.
            tracking_dir: mlflow tracking/registry store directory.
            registered_model: The pipeline's registered model name in mlflow.
            model_version: The pipeline's registered version in mlflow
            source_run_id: The mlflow run the pipeline was trained in.
                Used by setup_mlflow() to log a new run into the same experiment, 
                and to look up its data_prep_config.json.
        """
        self.pipeline = pipeline
        self.data_builder = data_builder
        self.tracking_dir = tracking_dir
        self.registered_model = registered_model
        self.model_version = model_version
        self.source_run_id = source_run_id
        self.config = data_builder.config

    @classmethod
    def from_fit_pipeline(cls, pipeline: Pipeline, data_builder: TabularModelDataBuilder) -> "TabularModelInference":
        """
        Wraps an already-fit pipeline + in-memory data_builder.

        Args:
            pipeline: A fit pipeline to predict with.
            data_builder: The TabularModelDataBuilder the pipeline was fit under.

        Returns:
            A TabularModelInference that can evaluate()/predict().
        """
        return cls(pipeline=pipeline, data_builder=data_builder)

    @classmethod
    def from_registry(
        cls,
        data_dir: str,
        tracking_dir: str,
        registered_model: str,
        model_version: Optional[int] = None,
    ) -> "TabularModelInference":
        """
        Loads a registered model and reconstructs the TabularModelDataBuilder it was trained
        with from its logged data_prep_config.json artifact.

        Args:
            data_dir: Parent directory for the gold/predictions layers.
            tracking_dir: mlflow tracking/registry store directory.
            registered_model: Name of the model as registered in mlflow.
            model_version: Specific model version to load. Defaults to the latest version.

        Returns:
            A TabularModelInference ready for evaluate()/predict().
        """
        pipeline, mv = load_mlflow_model(registered_model, model_version, tracking_dir)
        model_version, source_run_id = int(mv.version), mv.run_id
        config = load_data_builder_config(source_run_id)
        data_builder = TabularModelDataBuilder(data_dir=data_dir, config=config)

        return cls(
            pipeline=pipeline,
            data_builder=data_builder,
            tracking_dir=tracking_dir,
            registered_model=registered_model,
            model_version=model_version,
            source_run_id=source_run_id,
        )

    def _select_pipeline_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Restricts X to exactly the columns -- in the same order -- the pipeline was fit on.

        Args:
            X: Candidate feature set (test split, or live prediction set).

        Returns:
            X restricted to self.pipeline.feature_names_in_, in that exact order.

        Raises:
            ValueError: If the pipeline needs a column that's no longer in X.
        """
        required_features = list(self.pipeline.feature_names_in_)
        missing = [col for col in required_features if col not in X.columns]
        if missing:
            raise ValueError(
                f"Training set is missing {len(missing)} column(s) the model was fit on: "
                f"{missing}. It may have been renamed/removed since the model was trained."
            )

        return X[required_features]

    def setup_mlflow(self, phase: str, run_name_suffix: str) -> str:
        """
        Creates a new run (tagged phase, with a source_run_id/model_version param pointing
        back at the training run) in the same mlflow experiment as self.source_run_id, and
        logs self.config onto it as a data_prep_config.json artifact.

        Only valid on a TabularModelInference built via from_registry (needs self.source_run_id/
        registered_model/model_version/tracking_dir).

        Args:
            phase: "eval"/"test"/"predict" -- tagged onto the new run.
            run_name_suffix: Appended to the run name, after the registered model/version.

        Returns:
            run_id - The resulting mlflow run id.
        """
        set_mlflow_tracking_uri(self.tracking_dir)
        source_experiment_id = mlflow.get_run(self.source_run_id).info.experiment_id
        experiment_name = mlflow.get_experiment(source_experiment_id).name

        run_id = setup_mlflow_run(
            experiment_name=experiment_name,
            run_name=f"{self.registered_model}_v{self.model_version}_{run_name_suffix}_"
                     f"{datetime.now().strftime('%Y%m%d%H%M%S')}",
            tracking_dir=self.tracking_dir,
            tags={"phase": phase},
            params={"source_run_id": self.source_run_id, "model_version": self.model_version},
        )
        mlflow.log_dict(self.config, DATA_BUILDER_CONFIG_ARTIFACT_PATH, run_id=run_id)

        return run_id

    @staticmethod
    def build_predictions_csv_path(
        predictions_dir: str,
        registered_model: str,
        model_version: int,
        run_id: str,
        split: Optional[str] = None,
    ) -> str:
        """
        Builds a predictions CSV path under predictions_dir (creating it if it doesn't
        exist), named "{registered_model}_v{model_version}_{split}_predictions_{run_id}.csv"
        (or, if split is omitted, "{registered_model}_v{model_version}_predictions_{run_id}.csv").

        Args:
            predictions_dir: Directory to write the CSV under.
            registered_model: The registered model name to embed in the filename.
            model_version: The registered model version to embed in the filename.
            run_id: The run_id the predictions are being logged into.
            split: "eval"/"test", if scoring against a split. Omit for live predictions.

        Returns:
            The csv path.
        """
        os.makedirs(predictions_dir, exist_ok=True)
        middle = f"{split}_predictions" if split else "predictions"
        return os.path.join(predictions_dir, f"{registered_model}_v{model_version}_{middle}_{run_id}.csv")

    def _predict(
        self,
        X: pd.DataFrame,
        identity: pd.DataFrame,
        run_id: str,
        csv_path: str,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Predicts with self.pipeline on X, and logs the predictions (+ actual, if y is given)
        as a CSV artifact under run_id (at the run's artifact root -- csv_path's own
        filename already says what it is, so there's no need to also nest it under a nested
        artifact directory).

        If y is given: includes an "actual" column, and sorts rows by
        [target_season, predictions, actual]. Pair this with _score to also log R^2/RMSE.

        If y is omitted (e.g. a live prediction set with no ground truth yet): rows are
        sorted by predictions alone.

        Args:
            X: Feature set to predict on.
            identity: Identity columns (e.g. player_display_name, target_season), aligned
                positionally with X, used to build the preds_df.
            run_id: mlflow run_id to log the artifact into (must already exist).
            csv_path: Where to write the predictions CSV on disk.
            y: Optional actual/ground-truth values, aligned positionally with X.

        Returns:
            DataFrame of identity + predictions (+ actual, if y was given), sorted as
            described above.
        """
        target = self.data_builder.target
        y_pred = self.pipeline.predict(X)

        preds_df = identity.copy()
        preds_df["predictions"] = y_pred

        if y is not None:
            preds_df["actual"] = y
            preds_df = preds_df.sort_values(by=["target_season", "predictions", "actual"], ascending=False)
        else:
            preds_df = preds_df.sort_values(by="predictions", ascending=False)

        with mlflow.start_run(run_id=run_id):
            output_df = preds_df.rename(columns={"predictions": target})
            output_df[target] = output_df[target].round(2)

            output_cols = ["player_display_name", "target_season", target] + (["actual"] if y is not None else [])
            output_df[output_cols].to_csv(csv_path, index=False)

            mlflow.log_artifact(csv_path)

        return preds_df

    @staticmethod
    def _score_slice(y: pd.Series, y_pred: np.ndarray, n: Optional[int] = None) -> None:
        """
        Logs R^2/RMSE for a slice of a scored data set -- either the n rows with the highest
        actual value (logged as "top_{n}_r2"/"top_{n}_rmse"), or, if n is None, the entire
        set with no restriction (logged as plain "r2"/"rmse", matching what
        pipeline.score()/mean_squared_error would give directly). Must be called inside an
        active mlflow run.

        Args:
            y: True target values for the data being scored.
            y_pred: Predicted values, aligned positionally with y (as returned by
                pipeline.predict).
            n: Number of rows (highest actual value first) to restrict to. None (default)
                scores every row, logged under the unprefixed "r2"/"rmse" names. If there are
                fewer than n rows, every row is used.
        """
        y_values = np.asarray(y)
        y_pred_values = np.asarray(y_pred)

        if n is None:
            idx = np.arange(len(y_values))
            prefix, label = "", "Overall"
        else:
            idx = np.argsort(y_values)[::-1][:n]
            prefix, label = f"top_{n}_", f"Top {n}"

        if len(idx) < 2:
            logger.warning(
                f"Only {len(idx)} row(s) available for {label}; skipping {prefix}r2/"
                f"{prefix}rmse (need at least 2 to compute a meaningful R^2)"
            )
            return

        y_slice, y_pred_slice = y_values[idx], y_pred_values[idx]

        rmse = np.sqrt(mean_squared_error(y_slice, y_pred_slice))
        print(f"{label} RMSE: {rmse} (n={len(idx)})")
        mlflow.log_metric(f"{prefix}rmse", rmse)

        ss_tot = ((y_slice - y_slice.mean()) ** 2).sum()
        if ss_tot > 0:
            r2 = 1 - ((y_slice - y_pred_slice) ** 2).sum() / ss_tot
            print(f"{label} R^2: {r2}")
            mlflow.log_metric(f"{prefix}r2", r2)
        else:
            logger.warning(f"actual has zero variance for {label}; skipping {prefix}r2")

    def _score(self, preds_df: pd.DataFrame, run_id: str, top_n_rows: Optional[List[int]] = None) -> None:
        """
        Logs overall + top_n R^2/RMSE (see _score_slice) for preds_df's "predictions" vs
        "actual" columns (e.g. as returned by _predict) under run_id.

        Args:
            preds_df: DataFrame with "predictions" and "actual" columns to score against
                each other.
            run_id: mlflow run_id to log metrics into (must already exist).
            top_n_rows: For each n in this list, also logs "top_{n}_r2"/"top_{n}_rmse"
                (restricted to the n rows with the highest actual value). Default: score the
                whole set only.
        """
        y, y_pred = preds_df["actual"], preds_df["predictions"]

        with mlflow.start_run(run_id=run_id):
            for n in [None, *(top_n_rows or [])]:
                self._score_slice(y, y_pred, n)

    def evaluate(
        self,
        run_id: str,
        csv_path: str,
        split: str = TEST_SPLIT,
        top_n_rows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Predicts + Scores self.pipeline against the model's eval or test split and logs the results.

        Args:
            run_id: mlflow run_id to log metrics/artifacts into (must already exist -- e.g.
                from setup_mlflow(), or TabularModelTrainer.eval_model's own already-open run_id).
            csv_path: Where to write the predictions CSV (e.g. from build_predictions_csv_path()).
            split: Name of the dataset split to evaluate against ("eval" or "test", default: "test").
            top_n_rows: For each n, also logs top_{n}_r2/top_{n}_rmse. None (default) skips
                top-n metrics entirely.

        Returns:
            DataFrame of predictions vs actuals (see _predict()).
        """
        data = self.data_builder.build_data_splits()
        X = self._select_pipeline_features(data[f"X_{split}"])

        preds_df = self._predict(
            X=X,
            identity=data[f"identity_{split}"],
            run_id=run_id,
            csv_path=csv_path,
            y=data[f"y_{split}"],
        )
        self._score(preds_df, run_id, top_n_rows=top_n_rows)

        return preds_df

    def predict(self, run_id: str, csv_path: str) -> pd.DataFrame:
        """
        Loads the live (not-yet-played season) gold prediction set and predicts against it.
        There's no ground truth to score against yet, so this never scores -- see evaluate()
        for that.

        Args:
            run_id: mlflow run_id to log the artifact into (must already exist -- e.g. from
                setup_mlflow()).
            csv_path: Where to write the predictions CSV (e.g. from build_predictions_csv_path()).

        Returns:
            DataFrame of live predictions (see _predict()).
        """
        prediction_set = self.data_builder.build_prediction_data()
        X = self._select_pipeline_features(prediction_set["features"])

        return self._predict(
            X=X,
            identity=prediction_set["identity"],
            run_id=run_id,
            csv_path=csv_path,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Scores a registered model against its own eval or test split, or "
                     "produces live predictions for the upcoming season, depending on --mode. "
                     "All reconstruct the model's TabularModelDataBuilder from its logged "
                     "data builder config."
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
        "--registered-model",
        type=str,
        required=True,
        help="Name of the model as registered in mlflow, e.g. ppr_points_per_game_rb_ridge."
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Specific model version to use. Defaults to the latest version."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[EVAL_SPLIT, TEST_SPLIT, "predict"],
        required=True,
        help="'eval'/'test' score the registered model against its own reconstructed "
             "eval/test split (reconstructed from its logged data builder config). 'predict' "
             "produces live predictions for the upcoming season from the gold prediction "
             "set (no scoring -- no ground truth available yet)."
    )
    parser.add_argument(
        "--top-n-rows",
        type=int,
        nargs="*",
        default=[50, 100, 200],
        help="Only used in --mode eval/test. Space-separated top-n values to also log "
             "top_{n}_r2/top_{n}_rmse for, e.g. --top-n-rows 50 100 200. Pass --top-n-rows "
             "with no values to skip top-n metrics entirely. (default: [50, 100, 200])"
    )

    args = parser.parse_args()

    inference = TabularModelInference.from_registry(
        data_dir=args.data_dir,
        tracking_dir=args.tracking_dir,
        registered_model=args.registered_model,
        model_version=args.model_version,
    )
    predictions_dir = os.path.join(args.data_dir, "predictions")

    if args.mode in (EVAL_SPLIT, TEST_SPLIT):
        run_id = inference.setup_mlflow(phase=args.mode, run_name_suffix=args.mode)
        csv_path = TabularModelInference.build_predictions_csv_path(
            predictions_dir, inference.registered_model, inference.model_version, run_id, split=args.mode
        )
        preds_df = inference.evaluate(run_id=run_id, csv_path=csv_path, split=args.mode, top_n_rows=args.top_n_rows)
    else:
        run_id = inference.setup_mlflow(phase="predict", run_name_suffix="predictions")
        csv_path = TabularModelInference.build_predictions_csv_path(
            predictions_dir, inference.registered_model, inference.model_version, run_id
        )
        preds_df = inference.predict(run_id=run_id, csv_path=csv_path)

    print(preds_df)


if __name__ == "__main__":
    main()
