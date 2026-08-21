import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import mlflow

from src.modeling.utils import predict, score, setup_mlflow_run


class _FakePipeline:
    """Minimal fake pipeline that returns a fixed set of predictions regardless of X."""

    def __init__(self, predictions):
        self._predictions = np.array(predictions, dtype=float)

    def predict(self, X):
        return self._predictions


class _SharedMlflowTestCase:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.tracking_dir = os.path.join(cls.test_dir, "mlruns")
        cls.predictions_dir = os.path.join(cls.test_dir, "predictions")
        os.makedirs(cls.predictions_dir, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def _new_run(self, experiment_name: str = "test_experiment") -> str:
        return setup_mlflow_run(
            experiment_name=experiment_name, run_name="run", tracking_dir=self.tracking_dir
        )


class TestScore(_SharedMlflowTestCase):
    def test_score__none_top_n_scores_the_whole_set_under_unprefixed_names(self):
        preds_df = pd.DataFrame({
            "predictions": [1.0, 2.0, 3.0, 8.0, 22.0, 33.0],
            "actual": [1, 2, 3, 10, 20, 30],
        })
        run_id = self._new_run()

        score(preds_df, run_id)

        metrics = mlflow.get_run(run_id).data.metrics
        y = preds_df["actual"].to_numpy(dtype=float)
        y_pred = preds_df["predictions"].to_numpy()
        expected_rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        expected_r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

        assert "n" not in metrics
        assert metrics["rmse"] == pytest.approx(expected_rmse)
        assert metrics["r2"] == pytest.approx(expected_r2)

    def test_score__restricts_r2_and_rmse_to_the_top_n_rows_by_actual(self):
        # Perfect predictions for the bottom 3 (by actual), imperfect for the top 3 -- if
        # top_3 didn't actually restrict to the top-3-by-actual rows, results would come out
        # as a perfect 0 RMSE / 1 R^2.
        preds_df = pd.DataFrame({
            "predictions": [1.0, 2.0, 3.0, 8.0, 22.0, 33.0],
            "actual": [1, 2, 3, 10, 20, 30],
        })
        run_id = self._new_run()

        score(preds_df, run_id, top_ns=[3])

        metrics = mlflow.get_run(run_id).data.metrics
        y_top, y_pred_top = np.array([10.0, 20.0, 30.0]), np.array([8.0, 22.0, 33.0])
        expected_rmse = np.sqrt(np.mean((y_top - y_pred_top) ** 2))
        expected_r2 = 1 - np.sum((y_top - y_pred_top) ** 2) / np.sum((y_top - y_top.mean()) ** 2)

        assert metrics["top_3_rmse"] == pytest.approx(expected_rmse)
        assert metrics["top_3_r2"] == pytest.approx(expected_r2)

    def test_score__caps_at_available_rows_when_fewer_than_n(self):
        preds_df = pd.DataFrame({
            "predictions": [1.0, 2.0, 3.0, 8.0, 22.0, 33.0],
            "actual": [1, 2, 3, 10, 20, 30],
        })
        run_id = self._new_run()

        score(preds_df, run_id, top_ns=[100])

        metrics = mlflow.get_run(run_id).data.metrics
        assert "top_100_r2" in metrics
        assert "top_100_rmse" in metrics

    def test_score__skips_r2_and_rmse_when_fewer_than_two_rows_available(self):
        preds_df = pd.DataFrame({"predictions": [8.0], "actual": [10]})
        run_id = self._new_run()

        score(preds_df, run_id, top_ns=[5])

        metrics = mlflow.get_run(run_id).data.metrics
        assert "top_5_rmse" not in metrics
        assert "top_5_r2" not in metrics

    def test_score__no_top_ns_only_logs_the_overall_metrics(self):
        preds_df = pd.DataFrame({
            "predictions": [1.0, 2.0, 3.0],
            "actual": [1, 2, 3],
        })
        run_id = self._new_run()

        score(preds_df, run_id)

        metrics = mlflow.get_run(run_id).data.metrics
        assert not any(key.startswith("top_") for key in metrics)
        assert "r2" in metrics
        assert "rmse" in metrics

    def test_score__does_not_misinterpret_a_reordered_index_as_labels(self):
        # Regression test: predict() sorts its returned frame, giving it a non-default
        # index -- score() must treat top-n selection positionally, not by pandas label.
        preds_df = pd.DataFrame(
            {"predictions": [33.0, 22.0, 8.0, 3.0, 2.0, 1.0], "actual": [30, 20, 10, 3, 2, 1]},
            index=[5, 4, 3, 2, 1, 0],
        )
        run_id = self._new_run()

        score(preds_df, run_id, top_ns=[3])

        metrics = mlflow.get_run(run_id).data.metrics
        y_top, y_pred_top = np.array([30.0, 20.0, 10.0]), np.array([33.0, 22.0, 8.0])
        expected_rmse = np.sqrt(np.mean((y_top - y_pred_top) ** 2))
        assert metrics["top_3_rmse"] == pytest.approx(expected_rmse)


class TestPredict(_SharedMlflowTestCase):
    def test_predict__sorts_by_predictions_descending_when_no_actual_given(self):
        identity = pd.DataFrame({
            "player_display_name": ["a", "b", "c"],
            "target_season": [2024, 2024, 2024],
        })
        pipeline = _FakePipeline([1.0, 3.0, 2.0])
        run_id = self._new_run()
        csv_path = os.path.join(self.predictions_dir, "no_actual.csv")

        preds_df = predict(
            pipeline=pipeline,
            X=pd.DataFrame({"f1": [1, 2, 3]}),
            identity=identity,
            target="ppr",
            run_id=run_id,
            csv_path=csv_path,
            artifact_path="predictions",
        )

        assert list(preds_df["predictions"]) == [3.0, 2.0, 1.0]
        assert "actual" not in preds_df.columns

    def test_predict__includes_actual_and_sorts_by_target_season_predictions_actual_when_given(self):
        identity = pd.DataFrame({
            "player_display_name": ["a", "b", "c"],
            "target_season": [2023, 2024, 2024],
        })
        pipeline = _FakePipeline([1.0, 3.0, 2.0])
        y = pd.Series([10, 30, 20])
        run_id = self._new_run()
        csv_path = os.path.join(self.predictions_dir, "with_actual.csv")

        preds_df = predict(
            pipeline=pipeline,
            X=pd.DataFrame({"f1": [1, 2, 3]}),
            identity=identity,
            target="ppr",
            run_id=run_id,
            csv_path=csv_path,
            artifact_path="predictions",
            y=y,
        )

        # target_season 2024 rows come first (descending), then ordered by predictions/actual
        assert list(preds_df["player_display_name"]) == ["b", "c", "a"]
        assert list(preds_df["actual"]) == [30, 20, 10]

    def test_predict__writes_csv_with_rounded_renamed_target_column(self):
        identity = pd.DataFrame({"player_display_name": ["a"], "target_season": [2024]})
        pipeline = _FakePipeline([1.23456])
        run_id = self._new_run()
        csv_path = os.path.join(self.predictions_dir, "rounded.csv")

        predict(
            pipeline=pipeline,
            X=pd.DataFrame({"f1": [1]}),
            identity=identity,
            target="ppr",
            run_id=run_id,
            csv_path=csv_path,
            artifact_path="predictions",
        )

        written = pd.read_csv(csv_path)
        assert list(written.columns) == ["player_display_name", "target_season", "ppr"]
        assert written["ppr"].iloc[0] == pytest.approx(1.23)

    def test_predict__csv_includes_actual_column_only_when_y_is_given(self):
        identity = pd.DataFrame({"player_display_name": ["a"], "target_season": [2024]})
        run_id = self._new_run()

        with_y_path = os.path.join(self.predictions_dir, "with_y.csv")
        predict(
            pipeline=_FakePipeline([1.0]),
            X=pd.DataFrame({"f1": [1]}),
            identity=identity,
            target="ppr",
            run_id=run_id,
            csv_path=with_y_path,
            artifact_path="predictions",
            y=pd.Series([5]),
        )
        assert "actual" in pd.read_csv(with_y_path).columns

        without_y_path = os.path.join(self.predictions_dir, "without_y.csv")
        predict(
            pipeline=_FakePipeline([1.0]),
            X=pd.DataFrame({"f1": [1]}),
            identity=identity,
            target="ppr",
            run_id=run_id,
            csv_path=without_y_path,
            artifact_path="predictions",
        )
        assert "actual" not in pd.read_csv(without_y_path).columns

    def test_predict__logs_the_csv_as_an_mlflow_artifact(self):
        identity = pd.DataFrame({"player_display_name": ["a"], "target_season": [2024]})
        run_id = self._new_run()
        csv_path = os.path.join(self.predictions_dir, "artifact.csv")

        predict(
            pipeline=_FakePipeline([1.0]),
            X=pd.DataFrame({"f1": [1]}),
            identity=identity,
            target="ppr",
            run_id=run_id,
            csv_path=csv_path,
            artifact_path="predictions",
        )

        mlflow.set_tracking_uri(self.tracking_dir)
        downloaded_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="predictions")
        assert os.path.exists(downloaded_path)
        assert "artifact.csv" in os.listdir(downloaded_path)
