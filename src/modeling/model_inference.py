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

from src.modeling.data_prep import DATA_PREP_CONFIG_ARTIFACT_PATH, TabularModelDataPrep
from src.modeling.utils import load_data_prep_config, load_mlflow_model, set_mlflow_tracking_uri, setup_mlflow_run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelInference:
    """
    Predicts/scores a fit pipeline against a dataset built from a TabularModelDataPrep,
    logging results to mlflow. Two methods, cleanly split by whether ground truth is
    available to score against:
      - evaluate(dataset): scores against one of the pipeline's own train/eval/test splits
        (dataset has ground truth -- always scores).
      - predict(): predicts against the live/upcoming-season gold prediction set (no ground
        truth yet -- never scores).

    Two ways to build one:
      - from_fit_pipeline: wraps an already-fit pipeline + in-memory TabularModelDataPrep,
        no I/O. Used by TabularModel.eval_model to score a just-fit model against one of its
        own train/eval/test splits, in the same run fit_model logged it to -- pass that run's
        run_id/csv_path into evaluate() explicitly.
      - from_registry: loads a registered model plus the TabularModelDataPrep config it was
        trained with (reconstructed from its logged data_prep_config.json artifact), for
        post-hoc use once the model already exists in the mlflow registry. evaluate()/
        predict() can then be called with no run_id/csv_path -- each creates its own new run
        (tagged phase=dataset/predict, alongside a source_run_id param pointing back at the
        training run) into the same mlflow experiment as the model's training/eval runs
        (found by looking up which experiment source_run_id itself lives in, rather than
        recomputing the name -- this also means a positional model's own
        "{target}_{positions}_tabular" experiment is used automatically, not just the
        general model's).

    Caveat: a from_registry-built instance assumes gold_dir/{target}__training_set.csv
    hasn't fundamentally changed (rows added/removed/changed or columns removed/changed)
    since the model was trained.
    """

    def __init__(
            self,
            pipeline: Pipeline,
            data_prep: TabularModelDataPrep,
            predictions_dir: Optional[str] = None,
            tracking_dir: Optional[str] = None,
            registered_model: Optional[str] = None,
            model_version: Optional[int] = None,
            source_run_id: Optional[str] = None,
    ):
        """
        Prefer from_fit_pipeline/from_registry over calling this directly.

        Args:
            pipeline: A fit pipeline to predict with.
            data_prep: The TabularModelDataPrep the pipeline was (or will be) fit under.
            predictions_dir: Where to write predictions CSVs, when evaluate()/predict() are
                called without an explicit csv_path. Required in that case.
            tracking_dir: mlflow tracking/registry store directory, when evaluate()/
                predict() are called without an explicit run_id. Required in that case.
            registered_model: The pipeline's registered model name, if loaded from the
                registry. Used to name csv/run outputs when evaluate()/predict() build their
                own run_id/csv_path.
            model_version: The pipeline's registered version, if loaded from the registry.
                Same use as registered_model.
            source_run_id: The mlflow run the pipeline was trained in, if loaded from the
                registry. Used to log a new run into the same experiment, and to look up its
                data_prep_config.json, when evaluate()/predict() build their own run_id.
        """
        self.pipeline = pipeline
        self.data_prep = data_prep
        self.predictions_dir = predictions_dir
        self.tracking_dir = tracking_dir
        self.registered_model = registered_model
        self.model_version = model_version
        self.source_run_id = source_run_id
        self.config = data_prep.config

    @classmethod
    def from_fit_pipeline(cls, pipeline: Pipeline, data_prep: TabularModelDataPrep) -> "ModelInference":
        """
        Wraps an already-fit pipeline + in-memory data_prep -- no I/O.

        Args:
            pipeline: A fit pipeline to predict with.
            data_prep: The TabularModelDataPrep the pipeline was fit under.

        Returns:
            A ModelInference that can evaluate() the pipeline's own splits (pass an explicit
            run_id/csv_path -- there's no registry metadata to build defaults from). Can't
            predict() -- that requires from_registry's gold prediction set/run metadata.
        """
        return cls(pipeline=pipeline, data_prep=data_prep)

    @classmethod
    def from_registry(
        cls,
        data_dir: str,
        tracking_dir: str,
        registered_model: str,
        model_version: Optional[int] = None,
    ) -> "ModelInference":
        """
        Loads a registered model and reconstructs the TabularModelDataPrep it was trained
        with from its logged data_prep_config.json artifact.

        Args:
            data_dir: Parent directory for the gold/predictions layers.
            tracking_dir: mlflow tracking/registry store directory.
            registered_model: Name of the model as registered in mlflow.
            model_version: Specific model version to load. Defaults to the latest version.

        Returns:
            A ModelInference ready for evaluate()/predict().
        """
        predictions_dir = os.path.join(data_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        pipeline, mv = load_mlflow_model(registered_model, model_version, tracking_dir)
        model_version, source_run_id = int(mv.version), mv.run_id
        config = load_data_prep_config(source_run_id)
        data_prep = TabularModelDataPrep(data_dir=data_dir, config=config)

        return cls(
            pipeline=pipeline,
            data_prep=data_prep,
            predictions_dir=predictions_dir,
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

    def _setup_run(self, phase: str, run_name_suffix: str) -> str:
        """
        Creates a new run (tagged phase, with a source_run_id/model_version param pointing
        back at the training run) in the same mlflow experiment as self.source_run_id, and
        logs self.config onto it as a data_prep_config.json artifact.

        Only valid on a ModelInference built via from_registry.

        Args:
            phase: "test" or "predict" -- tagged onto the new run.
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
        mlflow.log_dict(self.config, DATA_PREP_CONFIG_ARTIFACT_PATH, run_id=run_id)

        return run_id

    def _predict(
        self,
        X: pd.DataFrame,
        identity: pd.DataFrame,
        run_id: str,
        csv_path: str,
        artifact_path: str,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Predicts with self.pipeline on X, and logs the predictions (+ actual, if y is given)
        as a CSV artifact under run_id.

        If y is given: includes an "actual" column, and sorts rows by
        [target_season, predictions, actual]. Pair this with _score to also log R^2/RMSE.

        If y is omitted (e.g. a live prediction set with no ground truth yet): rows are
        sorted by predictions alone.

        Args:
            X: Feature set to predict on.
            identity: Identity columns (e.g. player_display_name, target_season) aligned
                positionally with X, kept in the output for context.
            run_id: mlflow run_id to log the artifact into (must already exist).
            csv_path: Where to write the predictions CSV on disk.
            artifact_path: mlflow artifact path to log csv_path under.
            y: Optional actual/ground-truth values, aligned positionally with X.

        Returns:
            DataFrame of identity + predictions (+ actual, if y was given), sorted as
            described above.
        """
        target = self.data_prep.target
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

            mlflow.log_artifact(csv_path, artifact_path)

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
        dataset: str = "test",
        run_id: Optional[str] = None,
        csv_path: Optional[str] = None,
        top_n_rows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Scores self.pipeline against one of its own train/eval/test splits (from
        self.data_prep.split()) and logs the results. Always scores -- dataset always has
        ground truth (it's part of the training set).

        Args:
            dataset: Which split to evaluate against, "eval" or "test" (default: "test").
            run_id: mlflow run_id to log metrics/artifacts into. If omitted, creates a new
                run (tagged phase=dataset, linked back to self.source_run_id) -- only valid
                on a ModelInference built via from_registry. A from_fit_pipeline instance
                (e.g. TabularModel.eval_model) must pass its own already-open run_id.
            csv_path: Where to write the predictions CSV. If omitted, named
                "{registered_model}_v{model_version}_{dataset}_predictions_{run_id}.csv" --
                only valid via from_registry; a from_fit_pipeline instance must pass its own.
            top_n_rows: For each n, also logs top_{n}_r2/top_{n}_rmse. None (default) skips
                top-n metrics entirely.

        Returns:
            DataFrame of predictions vs actuals (see _predict()).
        """
        data = self.data_prep.split()
        X = self._select_pipeline_features(data[f"X_{dataset}"])

        if run_id is None:
            run_id = self._setup_run(phase=dataset, run_name_suffix=dataset)
        if csv_path is None:
            csv_path = os.path.join(
                self.predictions_dir,
                f"{self.registered_model}_v{self.model_version}_{dataset}_predictions_{run_id}.csv",
            )

        preds_df = self._predict(
            X=X,
            identity=data[f"identity_{dataset}"],
            run_id=run_id,
            csv_path=csv_path,
            artifact_path=f"{dataset}_predictions",
            y=data[f"y_{dataset}"],
        )
        self._score(preds_df, run_id, top_n_rows=top_n_rows)

        return preds_df

    def predict(self, run_id: Optional[str] = None, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Loads the live (not-yet-played season) gold prediction set and predicts against it.
        There's no ground truth to score against yet, so this never scores -- see evaluate()
        for that.

        Args:
            run_id: mlflow run_id to log the artifact into. If omitted, creates a new run
                (tagged phase=predict, linked back to self.source_run_id) -- only valid on a
                ModelInference built via from_registry.
            csv_path: Where to write the predictions CSV. If omitted, named
                "{registered_model}_v{model_version}_predictions_{run_id}.csv" -- only valid
                via from_registry.

        Returns:
            DataFrame of live predictions (see _predict()).
        """
        prediction_set = self.data_prep.load_prediction_set()
        X = self._select_pipeline_features(prediction_set["features"])

        if run_id is None:
            run_id = self._setup_run(phase="predict", run_name_suffix="predictions")
        if csv_path is None:
            csv_path = os.path.join(
                self.predictions_dir, f"{self.registered_model}_v{self.model_version}_predictions_{run_id}.csv"
            )

        return self._predict(
            X=X,
            identity=prediction_set["identity"],
            run_id=run_id,
            csv_path=csv_path,
            artifact_path="predictions",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Scores a registered model against its own eval or test split, or "
                     "produces live predictions for the upcoming season, depending on --mode. "
                     "All reconstruct the model's TabularModelDataPrep from its logged "
                     "data prep config."
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
        choices=["eval", "test", "predict"],
        required=True,
        help="'eval'/'test' score the registered model against its own reconstructed "
             "eval/test split (reconstructed from its logged data prep config). 'predict' "
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

    inference = ModelInference.from_registry(
        data_dir=args.data_dir,
        tracking_dir=args.tracking_dir,
        registered_model=args.registered_model,
        model_version=args.model_version,
    )

    if args.mode in ("eval", "test"):
        preds_df = inference.evaluate(dataset=args.mode, top_n_rows=args.top_n_rows)
    else:
        preds_df = inference.predict()

    print(preds_df)


if __name__ == "__main__":
    main()
