import numpy as np
import pandas as pd
import os
import pytest
import shutil
import tempfile
import mlflow
from unittest.mock import patch
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from src.modeling.data_prep import TabularModelDataPrep
from src.modeling.tabular_models import TabularModel
from src.modeling.utils import set_mlflow_tracking_uri
from src.processing.column_registry import get_identity_columns


class _RecordingModel(BaseEstimator, RegressorMixin):
    """Minimal fake estimator that just records the sample_weight it was fit with."""

    def __init__(self):
        self.received_sample_weight = None

    def fit(self, X, y, sample_weight=None):
        self.received_sample_weight = sample_weight
        return self

    def predict(self, X):
        return np.zeros(len(X))


class TestTabularModel:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.gold_dir = os.path.join(cls.test_dir, "gold")
        cls.tracking_dir = os.path.join(cls.test_dir, "mlruns")

        os.makedirs(cls.gold_dir)

        n = 10

        identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
        identity_data["target_season"] = [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024]

        training_data = pd.DataFrame({
            **identity_data,
            'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            'f3': [12, 0, 8, 12, 0, 8, 12, 0, 8, 12],
            'target': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        })
        training_data.to_csv(os.path.join(cls.gold_dir, "target_1__training_set.csv"), index=False)

        cls.data_prep = TabularModelDataPrep(data_dir=cls.test_dir, config={"target": "target_1"})
        cls.model = TabularModel(
            data_dir=cls.test_dir, tracking_dir=cls.tracking_dir, data_prep=cls.data_prep, model_type="ridge"
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def _experiment_name_for_run(self, run_id: str) -> str:
        run = mlflow.get_run(run_id)
        return mlflow.get_experiment(run.info.experiment_id).name

    def _latest_registered_run_id(self, registered_name: str) -> str:
        set_mlflow_tracking_uri(self.tracking_dir)
        return MlflowClient().get_latest_versions(registered_name, stages=["None"])[0].run_id

    def test_init__wires_up_target_model_type_and_predictions_dir(self):
        model = TabularModel(
            data_dir=self.test_dir, tracking_dir=self.tracking_dir, data_prep=self.data_prep, model_type="ridge"
        )

        assert model.target == "target_1"
        assert model.model_type == "ridge"
        assert model.data_prep is self.data_prep
        assert os.path.isdir(os.path.join(self.test_dir, "predictions"))

    def test_init__eagerly_splits_data_from_data_prep(self):
        expected = self.data_prep.split()

        pd.testing.assert_frame_equal(self.model.data["X_train"], expected["X_train"])
        pd.testing.assert_series_equal(self.model.data["y_train"], expected["y_train"])
        assert set(self.model.data["identity_test"]["target_season"]) == {2024}

    def test_base_model_name__defaults_to_target_when_positions_not_set(self):
        assert self.model.base_model_name == "target_1"

    def test_base_model_name__includes_single_lowercased_position_when_data_prep_has_one(self):
        data_prep = TabularModelDataPrep(data_dir=self.test_dir, config={"target": "target_1", "positions": ["RB"]})
        model = TabularModel(
            data_dir=self.test_dir, tracking_dir=self.tracking_dir, data_prep=data_prep, model_type="ridge"
        )

        assert model.base_model_name == "target_1_rb"

    def test_base_model_name__sorts_multiple_lowercased_positions_regardless_of_config_order(self):
        data_prep = TabularModelDataPrep(
            data_dir=self.test_dir, config={"target": "target_1", "positions": ["WR", "RB"]}
        )
        model = TabularModel(
            data_dir=self.test_dir, tracking_dir=self.tracking_dir, data_prep=data_prep, model_type="ridge"
        )

        assert model.base_model_name == "target_1_rb_wr"

    def test_setup_mlflow__logs_data_prep_config_as_an_artifact(self):
        run_id = self.model.setup_mlflow()

        logged_config = mlflow.artifacts.load_dict(f"runs:/{run_id}/data_prep_config.json")
        assert logged_config == self.data_prep.config
        assert self._experiment_name_for_run(run_id) == "target_1_tabular"

    def test_setup_mlflow__tags_the_run_with_model_type_and_phase_train(self):
        run_id = self.model.setup_mlflow()

        tags = mlflow.get_run(run_id).data.tags
        assert tags["model_type"] == "ridge"
        assert tags["phase"] == "train"

    def test_setup_mlflow__uses_position_specific_experiment_name_when_data_prep_has_one(self):
        data_prep = TabularModelDataPrep(data_dir=self.test_dir, config={"target": "target_1", "positions": ["RB"]})
        model = TabularModel(
            data_dir=self.test_dir, tracking_dir=self.tracking_dir, data_prep=data_prep, model_type="ridge"
        )

        run_id = model.setup_mlflow()

        assert self._experiment_name_for_run(run_id) == "target_1_rb_tabular"

    def test_setup_mlflow__sorts_multiple_positions_regardless_of_config_order(self):
        data_prep = TabularModelDataPrep(
            data_dir=self.test_dir, config={"target": "target_1", "positions": ["WR", "RB"]}
        )
        model = TabularModel(
            data_dir=self.test_dir, tracking_dir=self.tracking_dir, data_prep=data_prep, model_type="ridge"
        )

        run_id = model.setup_mlflow()

        assert self._experiment_name_for_run(run_id) == "target_1_rb_wr_tabular"

    def test_fit_model__registers_under_the_base_model_name_and_model_type(self):
        run_id = self.model.setup_mlflow()
        self.model.fit_model(run_id=run_id)

        assert self._latest_registered_run_id("target_1_ridge") == run_id

    def test_fit_model__registers_under_a_position_specific_name_when_positions_are_set(self):
        # A positional model gets its own registered name (base_model_name + model_type,
        # positions included), distinct from the general model's -- not just a new version
        # under the same name.
        identity_data = {col: [f"{col}_{i}" for i in range(10)] for col in get_identity_columns("nflverse", "player_stats")}
        identity_data["position"] = ["RB", "WR"] * 5
        identity_data["target_season"] = [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024]
        training_data = pd.DataFrame({
            **identity_data,
            "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        })
        training_data.to_csv(os.path.join(self.gold_dir, "target_4__training_set.csv"), index=False)

        data_prep = TabularModelDataPrep(data_dir=self.test_dir, config={"target": "target_4", "positions": ["RB"]})
        model = TabularModel(
            data_dir=self.test_dir, tracking_dir=self.tracking_dir, data_prep=data_prep, model_type="ridge"
        )
        run_id = model.setup_mlflow()
        model.fit_model(run_id=run_id)

        assert self._latest_registered_run_id("target_4_rb_ridge") == run_id

    def test_fit_model__passes_sample_weight_train_to_the_underlying_model(self):
        fake_model = _RecordingModel()

        with patch.object(self.model, "get_base_model", return_value=fake_model):
            run_id = self.model.setup_mlflow()
            self.model.fit_model(run_id=run_id)

        assert fake_model.received_sample_weight is not None
        np.testing.assert_array_equal(fake_model.received_sample_weight, self.model.data["sample_weight_train"])

    def test_eval_model__logs_top_n_and_whole_split_metrics_by_default(self):
        # Eval split (target_season 2023) has targets [16, 17] -- fewer than 50/100/200, so
        # all three cap at the full eval split size (2).
        run_id = self.model.setup_mlflow()
        pipeline = self.model.fit_model(run_id=run_id)
        self.model.eval_model(pipeline, run_id)

        run = mlflow.get_run(run_id)
        assert "r2" in run.data.metrics
        assert "rmse" in run.data.metrics
        for n in (50, 100, 200):
            assert f"top_{n}_r2" in run.data.metrics
            assert f"top_{n}_rmse" in run.data.metrics

    def test_eval_model__empty_top_ns_skips_top_n_metrics_entirely(self):
        run_id = self.model.setup_mlflow()
        pipeline = self.model.fit_model(run_id=run_id)
        self.model.eval_model(pipeline, run_id, top_ns=[])

        run = mlflow.get_run(run_id)
        assert not any(key.startswith("top_") for key in run.data.metrics)
        # whole-split r2/rmse are always logged regardless of top_ns
        assert "r2" in run.data.metrics
        assert "rmse" in run.data.metrics

    def test_eval_model__writes_a_predictions_csv_named_with_base_model_name_type_and_split(self):
        run_id = self.model.setup_mlflow()
        pipeline = self.model.fit_model(run_id=run_id)
        self.model.eval_model(pipeline, run_id)

        expected_path = os.path.join(self.test_dir, "predictions", f"target_1_ridge_eval_predictions_{run_id}.csv")
        assert os.path.exists(expected_path)

    def test_eval_model__can_score_the_test_split_too(self):
        run_id = self.model.setup_mlflow()
        pipeline = self.model.fit_model(run_id=run_id)
        preds_df = self.model.eval_model(pipeline, run_id, split="test")

        assert set(preds_df["target_season"]) == {2024}

    def test_param_search__is_one_at_a_time_not_a_full_cartesian_grid(self):
        # 3 alpha values + 2 fit_intercept values = 5 runs, not the 3x2=6 a cartesian grid
        # would produce
        results = self.model.param_search(
            param_grid={"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]},
        )

        assert len(results) == 5
        assert results["search_param"].value_counts().to_dict() == {"alpha": 3, "fit_intercept": 2}
        assert list(results["search_value"]) == [0.1, 1.0, 10.0, True, False]
        assert results["run_id"].nunique() == 5
        assert results["eval_rmse"].notna().all()
        assert results["eval_r2"].notna().all()
        assert results["eval_top_100_rmse"].notna().all()
        assert results["eval_top_100_r2"].notna().all()
        assert results["eval_top_200_rmse"].notna().all()
        assert results["eval_top_200_r2"].notna().all()

    def test_param_search__nests_child_runs_under_one_parent_run_per_search_key(self):
        results = self.model.param_search(
            param_grid={"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]},
        )

        parent_ids_by_key = {}
        for _, row in results.iterrows():
            run = mlflow.get_run(row["run_id"])
            assert run.data.tags["phase"] == f"{row['search_param']}_{row['search_value']}_child"
            assert run.data.tags["model_type"] == "ridge"
            parent_ids_by_key.setdefault(row["search_param"], set()).add(run.data.tags["mlflow.parentRunId"])

        # every child for a given key shares exactly one parent run...
        assert all(len(ids) == 1 for ids in parent_ids_by_key.values())
        # ...and different keys get different parent runs
        assert parent_ids_by_key["alpha"] != parent_ids_by_key["fit_intercept"]

        for key, parent_ids in parent_ids_by_key.items():
            parent_run = mlflow.get_run(next(iter(parent_ids)))
            assert parent_run.data.tags["phase"] == f"{key}_search"
            assert parent_run.data.params["search_param"] == key

    def test_param_search__holds_non_swept_params_at_sklearn_defaults(self):
        results = self.model.param_search(param_grid={"alpha": [0.1, 1.0]})

        default_fit_intercept = Ridge().get_params()["fit_intercept"]
        for _, row in results.iterrows():
            run = mlflow.get_run(row["run_id"])
            assert run.data.params["fit_intercept"] == str(default_fit_intercept)
