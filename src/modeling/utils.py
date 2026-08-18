import os
from typing import Optional, Tuple

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline


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
    target: str,
    model_type: str,
    model_version: Optional[int] = None,
    tracking_dir: str = "../mlruns",
) -> Tuple[Pipeline, ModelVersion]:
    """
    Loads a registered {target}_{model_type} model from the mlflow model registry rooted at
    tracking_dir.

    Args:
        target: e.g. "fantasy_points_ppr"
        model_type: e.g. "random_forest"
        model_version: Specific registered version to load. Defaults to the latest version.
        tracking_dir: The mlruns tracking/registry store directory (default: "../mlruns")

    Returns:
        (pipeline, model_version) - the loaded pipeline, and the registry's ModelVersion entry
        for it (.version, .run_id -- the mlflow run that trained/logged it, e.g. to look up the
        hyperparams or, for models registered via tabular_models.py's CLI, the train/eval/test
        split params logged on that run -- plus whatever else the caller needs off it).
    """
    set_mlflow_tracking_uri(tracking_dir)

    registered_name = f"{target}_{model_type}"
    client = MlflowClient()
    if model_version is None:
        model_version = client.get_latest_versions(registered_name, stages=["None"])[0]
    else:
        model_version = client.get_model_version(registered_name, str(model_version))

    pipeline = mlflow.sklearn.load_model(f"models:/{registered_name}/{model_version.version}")

    return pipeline, model_version
