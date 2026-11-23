import os
import logging
from typing import Any, Optional, Tuple

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.modeling.data_builder import DATA_BUILDER_CONFIG_ARTIFACT_PATH

logger = logging.getLogger(__name__)


def build_base_model_name(target: str, positions: list[str] | None) -> str:
    """Build a target name with an optional sorted, lowercase position suffix."""
    normalized = sorted(pos.lower() for pos in positions or [])
    suffix = f"_{'_'.join(normalized)}" if normalized else ""
    return f"{target}{suffix}"


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
        experiment_name: e.g. "{target}" or "{target}_{positions}", shared across model families.
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
) -> Tuple[Pipeline | mlflow.pyfunc.PyFuncModel, ModelVersion]:
    """
    Loads a registered {target}_{model_type} model from the mlflow model registry rooted at
    tracking_dir. Inspect the saved flavors to select the loader: prefer native
    sklearn (preserving estimator attributes), otherwise use the pyfunc interface.

    Args:
        registered_model: registered model to load.
        model_version: Specific registered version to load. Defaults to the highest numeric version.
        tracking_dir: The mlruns tracking/registry store directory (default: "../mlruns")

    Returns:
        (pipeline, model_version) - the loaded pipeline, and the registry's ModelVersion entry
        for it (.version, .run_id -- the mlflow run that trained/logged it, e.g. to look up the
        hyperparams or, for models registered via tabular/trainer.py's CLI, the train/eval/test
        split params logged on that run -- plus whatever else the caller needs off it).
    """
    set_mlflow_tracking_uri(tracking_dir)

    client = MlflowClient()
    if model_version is None:
        versions = client.search_model_versions(
            f"name = '{registered_model}'", max_results=1, order_by=["version_number DESC"],
        )
        if not versions:
            raise ValueError(f"No versions found for registered model '{registered_model}'.")
        version = versions[0]
    else:
        version = client.get_model_version(registered_model, str(model_version))

    model_uri = f"models:/{registered_model}/{version.version}"
    flavors = mlflow.models.Model.load(model_uri).flavors
    if "sklearn" in flavors:
        pipeline = mlflow.sklearn.load_model(model_uri)
    elif "python_function" in flavors:
        pipeline = mlflow.pyfunc.load_model(model_uri)
    else:
        raise ValueError(f"No supported sklearn or pyfunc flavor found for '{model_uri}'.")

    return pipeline, version


def load_data_builder_config(source_run_id: str) -> dict[str, Any]:
    """
    Downloads and parses the ModelDataBuilder config (see
    data_builder.DATA_BUILDER_CONFIG_ARTIFACT_PATH) logged as an mlflow artifact on source_run_id.

    Args:
        source_run_id: The training run to pull the config from.

    Returns:
        The logged ModelDataBuilder config dict.

    Raises:
        RuntimeError: If the run has no data_prep_config.json artifact -- e.g. it was
            created by fit_model()/param_search() directly, or by a version of
            tabular/trainer.py predating config-driven data building.
    """
    try:
        return mlflow.artifacts.load_dict(f"runs:/{source_run_id}/{DATA_BUILDER_CONFIG_ARTIFACT_PATH}")
    except mlflow.exceptions.MlflowException as e:
        raise RuntimeError(
            f"Run {source_run_id} has no {DATA_BUILDER_CONFIG_ARTIFACT_PATH} artifact."
        ) from e
