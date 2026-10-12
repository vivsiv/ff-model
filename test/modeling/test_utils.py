import os
import shutil
import tempfile

import mlflow
import pytest

from src.modeling.utils import load_data_prep_config, setup_mlflow_run


class TestLoadDataPrepConfig:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.tracking_dir = os.path.join(cls.test_dir, "mlruns")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_load_data_prep_config__returns_the_config_logged_on_the_given_run(self):
        run_id = setup_mlflow_run(
            experiment_name="test_experiment", run_name="run", tracking_dir=self.tracking_dir
        )
        config = {"target": "ppr_points_per_game", "positions": ["RB"]}
        with mlflow.start_run(run_id=run_id):
            mlflow.log_dict(config, "data_prep_config.json")

        assert load_data_prep_config(run_id) == config

    def test_load_data_prep_config__raises_if_the_run_has_no_data_prep_config_artifact(self):
        # e.g. a run created by fit_model()/param_search() directly rather than via
        # setup_mlflow() -- it never gets a data_prep_config.json artifact logged onto it.
        run_id = setup_mlflow_run(
            experiment_name="test_experiment", run_name="bare_run", tracking_dir=self.tracking_dir
        )

        with pytest.raises(RuntimeError, match="data_prep_config"):
            load_data_prep_config(run_id)
