"""Shared bookkeeping and MLflow infrastructure for model inference."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Generic, Self, TypeVar

import mlflow

from src.modeling.data_builder import DATA_BUILDER_CONFIG_ARTIFACT_PATH, ModelDataBuilder
from src.modeling.utils import load_data_builder_config, load_mlflow_model, set_mlflow_tracking_uri, setup_mlflow_run

PipelineType = TypeVar("PipelineType")
DataBuilderType = TypeVar("DataBuilderType", bound=ModelDataBuilder)


class ModelInference(Generic[PipelineType, DataBuilderType]):
    """Share inference metadata without imposing a prediction/scoring interface.

    Subclasses supply model_family, builder construction, and prediction/evaluation methods.
    Generic types preserve the concrete pipeline and builder types in subclasses.
    """

    model_family: ClassVar[str | None] = None

    def __init__(
        self,
        pipeline: PipelineType,
        data_builder: DataBuilderType,
        tracking_dir: str | None = None,
        registered_model: str | None = None,
        model_version: int | None = None,
        source_run_id: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.data_builder = data_builder
        self.tracking_dir = tracking_dir
        self.registered_model = registered_model
        self.model_version = model_version
        self.source_run_id = source_run_id
        self.config = data_builder.config

    @classmethod
    def from_registry(
        cls,
        data_dir: str,
        tracking_dir: str,
        registered_model: str,
        model_version: int | None = None,
    ) -> Self:
        """Load a registered pipeline and build the correct inference subclass.

        The shared loader selects the MLflow flavor. The subclass hook constructs
        its builder from saved config, and its constructor handles model-specific state.
        """
        pipeline, version = load_mlflow_model(registered_model, model_version, tracking_dir)
        if not version.run_id:
            raise ValueError("The registered model version has no source training run.")
        config = load_data_builder_config(version.run_id)
        data_builder = cls._build_data_builder(data_dir, config)
        if config.get("split", {}).get("prediction_season") is None:
            logging.getLogger(cls.__module__).warning(
                "Legacy model %s v%s has no saved prediction_season; temporarily using "
                "current roster season %s. Migrate retained models to an explicit season "
                "during artifact cleanup.",
                registered_model, version.version, data_builder.split_years.prediction_season,
            )

        return cls(
            pipeline=pipeline, data_builder=data_builder, tracking_dir=tracking_dir,
            registered_model=registered_model, model_version=int(version.version), source_run_id=version.run_id,
        )

    @staticmethod
    def _build_data_builder(data_dir: str, config: dict[str, Any]) -> DataBuilderType:
        """Subclass hook for explicit builder construction, without storing its type."""
        raise NotImplementedError("Inference subclasses must implement _build_data_builder().")

    def setup_mlflow(self, phase: str, run_name_suffix: str | None = None) -> str:
        """Create a linked inference run in the source model's original experiment.

        Preserve legacy experiment names, copy the source model_type tag, and log
        the builder config. Omitted run-name suffixes default to the phase.
        """
        if any(value is None for value in (
            self.tracking_dir, self.registered_model, self.model_version, self.source_run_id,
        )):
            raise ValueError("Load inference with from_registry() before creating an inference run.")

        set_mlflow_tracking_uri(self.tracking_dir)
        source_run = mlflow.get_run(self.source_run_id)
        experiment_name = mlflow.get_experiment(source_run.info.experiment_id).name
        tags = {"phase": phase}
        model_family = self.model_family or source_run.data.tags.get("model_family")
        if model_family:
            tags["model_family"] = model_family
        if model_type := source_run.data.tags.get("model_type"):
            tags["model_type"] = model_type
        suffix = phase if run_name_suffix is None else run_name_suffix

        run_id = setup_mlflow_run(
            experiment_name=experiment_name,
            run_name=f"{self.registered_model}_v{self.model_version}_{suffix}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            tracking_dir=self.tracking_dir, tags=tags,
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
        split: str | None = None,
    ) -> str:
        """Create a version/run-specific CSV path under the predictions directory."""
        directory = Path(predictions_dir)
        directory.mkdir(parents=True, exist_ok=True)
        middle = f"{split}_predictions" if split else "predictions"

        return str(directory / f"{registered_model}_v{model_version}_{middle}_{run_id}.csv")
