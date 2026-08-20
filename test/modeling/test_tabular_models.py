import numpy as np
import pandas as pd
import os
import pytest
import shutil
import tempfile
import mlflow
from unittest.mock import patch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from src.modeling.data_prep import TabularModelDataPrep
from src.modeling.tabular_models import TabularModel
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
        cls.model = TabularModel(data_dir=cls.test_dir, tracking_dir=cls.tracking_dir, data_prep=cls.data_prep)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_init__wires_up_target_and_predictions_dir_from_data_prep(self):
        model = TabularModel(data_dir=self.test_dir, tracking_dir=self.tracking_dir, data_prep=self.data_prep)

        assert model.target == "target_1"
        assert model.data_prep is self.data_prep
        assert os.path.isdir(os.path.join(self.test_dir, "predictions"))

    def test_split_data__delegates_to_data_prep_split(self):
        data = self.model.split_data()
        expected = self.data_prep.split()

        pd.testing.assert_frame_equal(data["X_train"], expected["X_train"])
        pd.testing.assert_series_equal(data["y_train"], expected["y_train"])
        assert set(data["identity_test"]["target_season"]) == {2024}

    def test_setup_mlflow__logs_data_prep_config_as_an_artifact(self):
        run_id = self.model.setup_mlflow("ridge")

        logged_config = mlflow.artifacts.load_dict(f"runs:/{run_id}/data_prep_config.json")
        assert logged_config == self.data_prep.config

    def test_fit_model__passes_sample_weight_train_to_the_underlying_model(self):
        data = self.model.split_data()
        fake_model = _RecordingModel()

        with patch.object(self.model, "get_base_model", return_value=fake_model):
            run_id = self.model.setup_mlflow("fake")
            self.model.fit_model(data, "fake", run_id)

        assert fake_model.received_sample_weight is not None
        np.testing.assert_array_equal(fake_model.received_sample_weight, data["sample_weight_train"])

    def test_param_search__is_one_at_a_time_not_a_full_cartesian_grid(self):
        # 3 alpha values + 2 fit_intercept values = 5 runs, not the 3x2=6 a cartesian grid
        # would produce
        data = self.model.split_data()
        results = self.model.param_search(
            data,
            model_type="ridge",
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
        data = self.model.split_data()
        results = self.model.param_search(
            data,
            model_type="ridge",
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
        data = self.model.split_data()
        results = self.model.param_search(
            data,
            model_type="ridge",
            param_grid={"alpha": [0.1, 1.0]},
        )

        default_fit_intercept = Ridge().get_params()["fit_intercept"]
        for _, row in results.iterrows():
            run = mlflow.get_run(row["run_id"])
            assert run.data.params["fit_intercept"] == str(default_fit_intercept)

    def test_log_performance_metrics__none_scores_the_whole_split_under_unprefixed_names(self):
        y = pd.Series([1, 2, 3, 10, 20, 30])
        y_pred = np.array([1.0, 2.0, 3.0, 8.0, 22.0, 33.0])

        run_id = self.model.setup_mlflow("ridge")
        with mlflow.start_run(run_id=run_id):
            self.model._log_performance_metrics(y, y_pred, n=None)

        metrics = mlflow.get_run(run_id).data.metrics
        y_arr, y_pred_arr = np.asarray(y, dtype=float), y_pred
        expected_rmse = np.sqrt(np.mean((y_arr - y_pred_arr) ** 2))
        expected_r2 = 1 - np.sum((y_arr - y_pred_arr) ** 2) / np.sum((y_arr - y_arr.mean()) ** 2)

        assert "n" not in metrics
        assert metrics["rmse"] == pytest.approx(expected_rmse)
        assert metrics["r2"] == pytest.approx(expected_r2)

    def test_log_performance_metrics__restricts_r2_and_rmse_to_the_top_n_rows_by_actual(self):
        # Perfect predictions for the bottom 3 (by actual), imperfect for the top 3 -- if
        # top_3 didn't actually restrict to the top-3-by-actual rows, results would come out
        # as a perfect 0 RMSE / 1 R^2.
        y = pd.Series([1, 2, 3, 10, 20, 30])
        y_pred = np.array([1.0, 2.0, 3.0, 8.0, 22.0, 33.0])

        run_id = self.model.setup_mlflow("ridge")
        with mlflow.start_run(run_id=run_id):
            self.model._log_performance_metrics(y, y_pred, n=3)

        metrics = mlflow.get_run(run_id).data.metrics

        y_top, y_pred_top = np.array([10.0, 20.0, 30.0]), np.array([8.0, 22.0, 33.0])
        expected_rmse = np.sqrt(np.mean((y_top - y_pred_top) ** 2))
        expected_r2 = 1 - np.sum((y_top - y_pred_top) ** 2) / np.sum((y_top - y_top.mean()) ** 2)

        assert metrics["top_3_rmse"] == pytest.approx(expected_rmse)
        assert metrics["top_3_r2"] == pytest.approx(expected_r2)

    def test_log_performance_metrics__caps_at_available_rows_when_fewer_than_n(self):
        y = pd.Series([1, 2, 3, 10, 20, 30])
        y_pred = np.array([1.0, 2.0, 3.0, 8.0, 22.0, 33.0])

        run_id = self.model.setup_mlflow("ridge")
        with mlflow.start_run(run_id=run_id):
            self.model._log_performance_metrics(y, y_pred, n=100)

        metrics = mlflow.get_run(run_id).data.metrics
        assert "top_100_r2" in metrics
        assert "top_100_rmse" in metrics

    def test_log_performance_metrics__skips_r2_and_rmse_when_fewer_than_two_rows_available(self):
        y = pd.Series([10])
        y_pred = np.array([8.0])

        run_id = self.model.setup_mlflow("ridge")
        with mlflow.start_run(run_id=run_id):
            self.model._log_performance_metrics(y, y_pred, n=5)

        metrics = mlflow.get_run(run_id).data.metrics
        assert "top_5_rmse" not in metrics
        assert "top_5_r2" not in metrics

    def test_eval_model__logs_top_n_and_whole_split_metrics_by_default(self):
        # Eval split (target_season 2023) has targets [16, 17] -- fewer than 50/100/200, so
        # all three cap at the full eval split size (2).
        data = self.model.split_data()
        run_id = self.model.setup_mlflow("ridge")
        pipeline = self.model.fit_model(data, "ridge", run_id)
        self.model.eval_model(pipeline, data, run_id)

        run = mlflow.get_run(run_id)
        assert "r2" in run.data.metrics
        assert "rmse" in run.data.metrics
        for n in (50, 100, 200):
            assert f"top_{n}_r2" in run.data.metrics
            assert f"top_{n}_rmse" in run.data.metrics

    def test_eval_model__empty_top_ns_skips_top_n_metrics_entirely(self):
        data = self.model.split_data()
        run_id = self.model.setup_mlflow("ridge")
        pipeline = self.model.fit_model(data, "ridge", run_id)
        self.model.eval_model(pipeline, data, run_id, top_ns=[])

        run = mlflow.get_run(run_id)
        assert not any(key.startswith("top_") for key in run.data.metrics)
        # whole-split r2/rmse are always logged regardless of top_ns
        assert "r2" in run.data.metrics
        assert "rmse" in run.data.metrics
