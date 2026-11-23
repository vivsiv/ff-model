import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
import mlflow.sklearn
import pytest

from src.modeling.utils import build_base_model_name, load_data_builder_config, load_mlflow_model, setup_mlflow_run


@pytest.mark.parametrize("positions, expected", [
    (None, "ppr_points_per_game"),
    ([], "ppr_points_per_game"),
    (["RB"], "ppr_points_per_game_rb"),
    (["WR", "RB"], "ppr_points_per_game_rb_wr"),
    (["wr", "Rb", "TE"], "ppr_points_per_game_rb_te_wr"),
])
def test_build_base_model_name(positions, expected):
    original = None if positions is None else positions.copy()

    assert build_base_model_name("ppr_points_per_game", positions) == expected
    assert positions == original


@pytest.mark.parametrize("flavors, loader", [
    ({"sklearn": {}, "python_function": {}}, "sklearn"),
    ({"sklearn": {}}, "sklearn"),
    ({"python_function": {}}, "pyfunc"),
])
def test_load_mlflow_model_selects_loader_from_saved_flavors(flavors, loader):
    version = SimpleNamespace(version="3", run_id="training-run")
    client = Mock()
    client.get_model_version.return_value = version
    with (
        patch("src.modeling.utils.set_mlflow_tracking_uri") as set_tracking,
        patch("src.modeling.utils.MlflowClient", return_value=client),
        patch("src.modeling.utils.mlflow.models.Model.load", return_value=SimpleNamespace(flavors=flavors)) as metadata,
        patch("src.modeling.utils.mlflow.sklearn.load_model") as sklearn_loader,
        patch("src.modeling.utils.mlflow.pyfunc.load_model") as pyfunc_loader,
    ):
        pipeline, actual_version = load_mlflow_model("registered_model", model_version=3, tracking_dir="mlruns")

    model_uri = "models:/registered_model/3"
    set_tracking.assert_called_once_with("mlruns")
    client.get_model_version.assert_called_once_with("registered_model", "3")
    metadata.assert_called_once_with(model_uri)
    selected, unused = (sklearn_loader, pyfunc_loader) if loader == "sklearn" else (pyfunc_loader, sklearn_loader)
    selected.assert_called_once_with(model_uri)
    unused.assert_not_called()
    assert pipeline is selected.return_value
    assert actual_version is version


def test_load_mlflow_model_rejects_unsupported_flavors():
    client = Mock()
    client.get_model_version.return_value = SimpleNamespace(version="1")
    with (
        patch("src.modeling.utils.set_mlflow_tracking_uri"),
        patch("src.modeling.utils.MlflowClient", return_value=client),
        patch("src.modeling.utils.mlflow.models.Model.load", return_value=SimpleNamespace(flavors={"custom": {}})),
    ):
        with pytest.raises(ValueError, match="No supported sklearn or pyfunc flavor"):
            load_mlflow_model("registered_model", model_version=1)


class TestLoadDataBuilderConfig:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.tracking_dir = os.path.join(cls.test_dir, "mlruns")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_load_data_builder_config__returns_the_config_logged_on_the_given_run(self):
        run_id = setup_mlflow_run(
            experiment_name="test_experiment", run_name="run", tracking_dir=self.tracking_dir
        )
        config = {"target": "ppr_points_per_game", "positions": ["RB"]}
        with mlflow.start_run(run_id=run_id):
            mlflow.log_dict(config, "data_prep_config.json")

        assert load_data_builder_config(run_id) == config

    def test_load_data_builder_config__raises_if_the_run_has_no_config_artifact(self):
        # e.g. a run created by fit_model()/param_search() directly rather than via
        # setup_mlflow() -- it never gets a data_prep_config.json artifact logged onto it.
        run_id = setup_mlflow_run(
            experiment_name="test_experiment", run_name="bare_run", tracking_dir=self.tracking_dir
        )

        with pytest.raises(RuntimeError, match="data_prep_config"):
            load_data_builder_config(run_id)
