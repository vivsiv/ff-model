import os
import logging
from typing import List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def set_mlflow_tracking_uri(tracking_dir: str = "../mlruns") -> None:
    """
    Points mlflow at the file-based tracking/registry store at tracking_dir.

    This needs to be called (with a tracking_dir relative to wherever the caller is running
    from) before any mlflow.* calls that read the tracking/registry store, 
    otherwise mlflow falls back to a default ./mlruns relative to the current working directory.

    Args:
        tracking_dir: The mlruns tracking/registry store directory (default: "../mlruns")
    """
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir)


def setup_mlflow_run(
    experiment_name: str,
    run_name: str,
    tracking_dir: str = "../mlruns",
    tags: Optional[dict] = None,
    params: Optional[dict] = None,
    parent_run_id: Optional[str] = None,
) -> str:
    """
    Points mlflow at tracking_dir, sets/creates the active experiment, and creates a new run
    under it (tagging it with tags and logging params if given), then immediately ends it.

    Downstream code reopens the run by run_id (mlflow.start_run(run_id=run_id)) to log
    additional params/metrics/artifacts into it.

    Args:
        experiment_name: e.g. "{target}_{model_type}"
        run_name: e.g. "{model_type}_{timestamp}"
        tracking_dir: The mlruns tracking/registry store directory (default: "../mlruns")
        tags: Optional tags to set on the run, e.g. {"model_type": "random_forest"}
        params: Optional params to log on the run, e.g. {"eval_data_years": 1}
        parent_run_id: If given, links this run as a child of parent_run_id -- it'll show up
            nested/collapsible under the parent in the mlflow UI's run list, while still being
            a completely normal run otherwise (own params/metrics/artifacts, still selectable
            for comparison views). Does not require the parent run to be active. (default:
            None, a top-level run)

    Returns:
        run_id - the created run's ID
    """
    set_mlflow_tracking_uri(tracking_dir)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, parent_run_id=parent_run_id) as run:
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        run_id = run.info.run_id

    return run_id


def load_mlflow_model(
    registered_model: str,
    model_version: Optional[int] = None,
    tracking_dir: str = "../mlruns",
) -> Tuple[Pipeline, ModelVersion]:
    """
    Loads a registered {target}_{model_type} model from the mlflow model registry rooted at
    tracking_dir.

    Args:
        registered_model: registered model to load.
        model_version: Specific registered version to load. Defaults to the latest version.
        tracking_dir: The mlruns tracking/registry store directory (default: "../mlruns")

    Returns:
        (pipeline, model_version) - the loaded pipeline, and the registry's ModelVersion entry
        for it (.version, .run_id -- the mlflow run that trained/logged it, e.g. to look up the
        hyperparams or, for models registered via tabular_models.py's CLI, the train/eval/test
        split params logged on that run -- plus whatever else the caller needs off it).
    """
    set_mlflow_tracking_uri(tracking_dir)

    client = MlflowClient()
    if model_version is None:
        model_version = client.get_latest_versions(registered_model, stages=["None"])[0]
    else:
        model_version = client.get_model_version(registered_model, str(model_version))

    pipeline = mlflow.sklearn.load_model(f"models:/{registered_model}/{model_version.version}")

    return pipeline, model_version


def _score_slice(y: pd.Series, y_pred: np.ndarray, n: Optional[int] = None) -> None:
    """
    Logs R^2/RMSE for a slice of a scored data set -- either the n rows with the highest
    actual value (logged as "top_{n}_r2"/"top_{n}_rmse"), or, if n is None, the entire set
    with no restriction (logged as plain "r2"/"rmse", matching what pipeline.score()/
    mean_squared_error would give directly). Must be called inside an active mlflow run.

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


def score(preds_df: pd.DataFrame, run_id: str, top_ns: Optional[List[int]] = None) -> None:
    """
    Logs overall + top_n R^2/RMSE (see _score_slice) for preds_df's "predictions" vs "actual"
    columns (e.g. as returned by predict()) under run_id.

    Args:
        preds_df: DataFrame with "predictions" and "actual" columns to score against
            each other.
        run_id: mlflow run_id to log metrics into (must already exist, e.g. from
            setup_mlflow_run).
        top_ns: For each n in this list, also logs "top_{n}_r2"/"top_{n}_rmse" (restricted to
            the n rows with the highest actual value). Default: score the whole set only.
    """
    y, y_pred = preds_df["actual"], preds_df["predictions"]

    with mlflow.start_run(run_id=run_id):
        for n in [None, *(top_ns or [])]:
            _score_slice(y, y_pred, n)


def predict(
    pipeline: Pipeline,
    X: pd.DataFrame,
    identity: pd.DataFrame,
    target: str,
    run_id: str,
    csv_path: str,
    artifact_path: str,
    y: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Predicts with pipeline on X, and logs the predictions (+ actual, if y is given) as a CSV
    artifact under run_id.

    If y is given: includes an "actual" column, and sorts rows by
    [target_season, predictions, actual]. Pair this with score() to also log R^2/RMSE.

    If y is omitted (e.g. a live prediction set with no ground truth yet): rows are sorted by
    predictions alone.

    Args:
        pipeline: A fit pipeline to predict with.
        X: Feature set to predict on.
        identity: Identity columns (e.g. player_display_name, target_season) aligned
            positionally with X, kept in the output for context.
        target: Prediction target name -- the "predictions" column is renamed to this before
            being written to csv_path (the returned frame keeps the raw "predictions" name).
        run_id: mlflow run_id to log the artifact into (must already exist, e.g. from
            setup_mlflow_run).
        csv_path: Where to write the predictions CSV on disk.
        artifact_path: mlflow artifact path to log csv_path under.
        y: Optional actual/ground-truth values, aligned positionally with X.

    Returns:
        DataFrame of identity + predictions (+ actual, if y was given), sorted as described
        above.
    """
    y_pred = pipeline.predict(X)

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
