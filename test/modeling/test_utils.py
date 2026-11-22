import os
import shutil
import tempfile

import mlflow
import pytest

from src.modeling.utils import build_base_model_name, load_data_builder_config, setup_mlflow_run


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
