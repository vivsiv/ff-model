import numpy as np
import pandas as pd
import os
import pytest
import shutil
import tempfile
from typing import Tuple

import mlflow
from sklearn.pipeline import Pipeline

from src.modeling.tabular.data_prep import TabularModelDataPrep
from src.modeling.tabular.trainer import TabularModelTrainer
from src.modeling.tabular.inference import TabularModelInference
from src.modeling.utils import setup_mlflow_run
from src.processing.column_registry import get_identity_columns


class _FakePipeline:
    """Minimal fake pipeline that returns a fixed set of predictions regardless of X."""

    def __init__(self, predictions):
        self._predictions = np.array(predictions, dtype=float)

    def predict(self, X):
        return self._predictions


class _FakeDataPrep:
    """Minimal stand-in for a TabularModelDataPrep -- just enough for TabularModelInference's
    _predict/_score (which only need .target and .config off it)."""

    def __init__(self, target: str = "ppr"):
        self.target = target
        self.config = {"target": target}


def _build_training_data(feature_cols: dict[str, list]) -> pd.DataFrame:
    n = 10
    identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
    identity_data["target_season"] = [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024]

    return pd.DataFrame({
        **identity_data,
        **feature_cols,
        'target': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    })


def _build_prediction_data(feature_cols: dict[str, list], n: int = 2) -> pd.DataFrame:
    identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
    identity_data["target_season"] = [2025] * n

    return pd.DataFrame({
        **identity_data,
        **{col: values[:n] for col, values in feature_cols.items()},
        'target': [None] * n,
    })


def _train_and_register_ridge_model(
    gold_dir: str, tracking_dir: str, target: str, training_data: pd.DataFrame
) -> Tuple[Pipeline, str]:
    """Trains+registers a ridge model on training_data (eval/test years=1, no exclusions),
    logging its data prep config the same way tabular/trainer.py's CLI (main()) does.

    Returns (pipeline, source_run_id)."""
    training_data.to_csv(os.path.join(gold_dir, f"{target}__training_set.csv"), index=False)

    data_prep = TabularModelDataPrep(data_dir=os.path.dirname(gold_dir), config={"target": target})
    model = TabularModelTrainer(
        data_dir=os.path.dirname(gold_dir), tracking_dir=tracking_dir, data_prep=data_prep, model_type="ridge"
    )
    run_id = model.setup_mlflow()
    pipeline = model.fit_model(run_id=run_id)
    model.eval_model(pipeline, run_id)

    return pipeline, run_id


class TestTabularModelInference:
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

        prediction_identity = {
            col: [f"{col}_pred_{i}" for i in range(2)] for col in get_identity_columns("nflverse", "player_stats")
        }
        prediction_identity["target_season"] = [2025, 2025]
        prediction_data = pd.DataFrame({
            **prediction_identity,
            'f1': [21, 22],
            'f2': [200, 150],
            'f3': [1, 2],
            'target': [None, None],
        })
        prediction_data.to_csv(os.path.join(cls.gold_dir, "target_1__prediction_set.csv"), index=False)

        # Simulate what tabular/trainer.py's CLI (main()) does when training+registering a
        # model: split (per a data prep config), fit, log the config + eval, all under one run.
        cls.config = {
            "target": "target_1",
            "features": {"mode": "exclude", "columns": ["f3"]},
            "split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 2},
        }
        cls.data_prep = TabularModelDataPrep(data_dir=cls.test_dir, config=cls.config)
        model = TabularModelTrainer(
            data_dir=cls.test_dir, tracking_dir=cls.tracking_dir, data_prep=cls.data_prep, model_type="ridge"
        )
        cls.data = model.data
        cls.train_run_id = model.setup_mlflow()
        cls.pipeline = model.fit_model(run_id=cls.train_run_id)
        model.eval_model(cls.pipeline, cls.train_run_id)

        cls.inference = TabularModelInference.from_registry(
            data_dir=cls.test_dir, tracking_dir=cls.tracking_dir, registered_model="target_1_ridge", model_version=1
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_init__does_not_build_a_tabular_model(self):
        # TabularModelInference has everything it needs (pipeline, reconstructed data_prep) to run
        # predict()/score() directly -- it shouldn't need to build a TabularModelTrainer at all.
        assert not hasattr(self.inference, "model")

    # -- from_fit_pipeline / evaluate (used directly by TabularModelTrainer.eval_model) ----------

    def test_from_fit_pipeline__wraps_pipeline_and_data_prep_without_loading_anything(self):
        inference = TabularModelInference.from_fit_pipeline(self.pipeline, self.data_prep)

        assert inference.pipeline is self.pipeline
        assert inference.data_prep is self.data_prep
        assert inference.registered_model is None
        assert inference.model_version is None
        assert inference.source_run_id is None

    def test_evaluate__predicts_and_scores_a_fit_pipelines_own_split(self):
        # from_fit_pipeline has no registry metadata to build run_id/csv_path defaults from,
        # so both must be passed explicitly -- this is exactly what TabularModelTrainer.eval_model
        # does, logging into the same run fit_model already opened.
        inference = TabularModelInference.from_fit_pipeline(self.pipeline, self.data_prep)
        run_id = setup_mlflow_run(
            experiment_name="target_1_tabular", run_name="direct_evaluate", tracking_dir=self.tracking_dir
        )
        csv_path = os.path.join(self.test_dir, "predictions", "direct_evaluate.csv")

        preds_df = inference.evaluate(split="test", run_id=run_id, csv_path=csv_path)

        assert set(preds_df["target_season"]) == {2024}
        run = mlflow.get_run(run_id)
        assert "r2" in run.data.metrics
        assert "rmse" in run.data.metrics

    def test_predict__also_works_from_a_fit_pipeline_given_explicit_run_id_and_csv_path(self):
        # predict() only needs registry metadata to build run_id/csv_path defaults --
        # prep_prediction_data() itself has no dependency on the registry, so a
        # from_fit_pipeline instance can predict() too as long as both are passed explicitly.
        inference = TabularModelInference.from_fit_pipeline(self.pipeline, self.data_prep)
        run_id = setup_mlflow_run(
            experiment_name="target_1_tabular", run_name="direct_predict", tracking_dir=self.tracking_dir
        )
        csv_path = os.path.join(self.test_dir, "predictions", "direct_predict.csv")

        preds_df = inference.predict(run_id=run_id, csv_path=csv_path)

        assert set(preds_df["target_season"]) == {2025}
        assert "actual" not in preds_df.columns
        run = mlflow.get_run(run_id)
        assert "r2" not in run.data.metrics
        assert "rmse" not in run.data.metrics

    def test_init__loads_the_pipeline_and_source_run_from_the_registered_model(self):
        assert self.inference.model_version == 1
        assert self.inference.source_run_id == self.train_run_id
        assert self.inference.config == self.config

    # -- setup_mlflow / build_predictions_csv_path -------------

    def test_setup_mlflow__tags_phase_and_links_back_to_the_source_run(self):
        run_id = self.inference.setup_mlflow(phase="test", run_name_suffix="test")

        run = mlflow.get_run(run_id)
        assert run.data.tags["phase"] == "test"
        assert run.data.params["source_run_id"] == self.train_run_id
        assert run.data.params["model_version"] == "1"

    def test_setup_mlflow__logs_into_the_same_experiment_as_the_source_run(self):
        run_id = self.inference.setup_mlflow(phase="test", run_name_suffix="test")

        run = mlflow.get_run(run_id)
        experiment = mlflow.get_experiment(run.info.experiment_id)
        assert experiment.name == "target_1_tabular"

    def test_setup_mlflow__also_logs_the_reconstructed_data_prep_config(self):
        run_id = self.inference.setup_mlflow(phase="test", run_name_suffix="test")

        logged_config = mlflow.artifacts.load_dict(f"runs:/{run_id}/data_prep_config.json")
        assert logged_config == self.config

    def test_setup_mlflow__uses_the_positional_models_own_experiment_when_positions_are_set(self):
        # A positional model's training run lives in its own "{target}_{positions}_tabular"
        # experiment (see TabularModelTrainer.base_model_name) -- setup_mlflow() must log into that
        # same experiment (found by looking up which experiment source_run_id itself lives
        # in), not the general one.
        test_dir = tempfile.mkdtemp()
        try:
            gold_dir = os.path.join(test_dir, "gold")
            tracking_dir = os.path.join(test_dir, "mlruns")
            os.makedirs(gold_dir)

            training_data = _build_training_data({"f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
            training_data["position"] = ["RB", "WR"] * 5
            training_data.to_csv(os.path.join(gold_dir, "target_3__training_set.csv"), index=False)

            data_prep = TabularModelDataPrep(
                data_dir=test_dir, config={"target": "target_3", "positions": ["RB"]}
            )
            model = TabularModelTrainer(
                data_dir=test_dir, tracking_dir=tracking_dir, data_prep=data_prep, model_type="ridge"
            )
            train_run_id = model.setup_mlflow()
            model.fit_model(run_id=train_run_id)

            inference = TabularModelInference.from_registry(
                data_dir=test_dir, tracking_dir=tracking_dir, registered_model="target_3_rb_ridge", model_version=1
            )
            run_id = inference.setup_mlflow(phase="test", run_name_suffix="test")

            run = mlflow.get_run(run_id)
            experiment = mlflow.get_experiment(run.info.experiment_id)
            assert experiment.name == "target_3_rb_tabular"
            assert run.data.params["source_run_id"] == train_run_id
        finally:
            shutil.rmtree(test_dir)

    def test_build_predictions_csv_path__names_the_path_with_registered_model_version_and_split(self):
        path = self.inference.build_predictions_csv_path(
            os.path.join(self.test_dir, "predictions"), "target_1_ridge", 1, "abc123", split="test"
        )
        assert path == os.path.join(self.test_dir, "predictions", "target_1_ridge_v1_test_predictions_abc123.csv")

    def test_build_predictions_csv_path__omits_the_split_from_the_name_when_not_given(self):
        path = self.inference.build_predictions_csv_path(
            os.path.join(self.test_dir, "predictions"), "target_1_ridge", 1, "abc123"
        )
        assert path == os.path.join(self.test_dir, "predictions", "target_1_ridge_v1_predictions_abc123.csv")

    # -- evaluate ------------------------------------------------------------------------

    def test_evaluate__reconstructs_the_same_test_split_the_model_was_trained_with(self):
        run_id = self.inference.setup_mlflow(phase="test", run_name_suffix="test")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id, split="test")
        preds_df = self.inference.evaluate(run_id=run_id, csv_path=csv_path, split="test")

        # test split held out 2024 (most recent season), per the logged config
        assert set(preds_df["target_season"]) == {2024}
        assert len(preds_df) == len(self.data["X_test"])

    def test_evaluate__predictions_match_the_source_pipeline_predicting_on_X_test(self):
        run_id = self.inference.setup_mlflow(phase="test", run_name_suffix="test")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id, split="test")
        preds_df = self.inference.evaluate(run_id=run_id, csv_path=csv_path, split="test")

        expected = pd.Series(
            self.pipeline.predict(self.data["X_test"]), index=self.data["X_test"].index
        )
        actual = preds_df["predictions"]

        pd.testing.assert_series_equal(
            expected.sort_index(), actual.sort_index(), check_names=False
        )

    def test_evaluate__logs_r2_and_rmse_metrics_into_the_given_run(self):
        run_id = self.inference.setup_mlflow(phase="test", run_name_suffix="test")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id, split="test")
        self.inference.evaluate(run_id=run_id, csv_path=csv_path, split="test")

        run = mlflow.get_run(run_id)
        assert "r2" in run.data.metrics
        assert "rmse" in run.data.metrics

    def test_evaluate__writes_a_predictions_csv_at_the_given_path(self):
        run_id = self.inference.setup_mlflow(phase="test", run_name_suffix="test")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id, split="test")
        self.inference.evaluate(run_id=run_id, csv_path=csv_path, split="test")

        assert os.path.exists(csv_path)

    def test_evaluate__split_eval_scores_the_eval_split_instead_of_test(self):
        run_id = self.inference.setup_mlflow(phase="eval", run_name_suffix="eval")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id, split="eval")
        preds_df = self.inference.evaluate(run_id=run_id, csv_path=csv_path, split="eval")

        # eval split held out 2023 (the year before the held-out test year), per the logged config
        assert set(preds_df["target_season"]) == {2023}
        assert len(preds_df) == len(self.data["X_eval"])

        run = mlflow.get_run(run_id)
        assert "r2" in run.data.metrics
        assert "rmse" in run.data.metrics
        assert os.path.exists(csv_path)

    def test_evaluate__ignores_feature_columns_added_to_the_training_set_after_training(self):
        # Simulates incremental feature growth: a column is added to the gold training set on
        # disk after the model was trained/registered, without retraining.
        test_dir = tempfile.mkdtemp()
        try:
            gold_dir = os.path.join(test_dir, "gold")
            tracking_dir = os.path.join(test_dir, "mlruns")
            os.makedirs(gold_dir)

            training_data = _build_training_data({
                'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            })
            pipeline, _ = _train_and_register_ridge_model(gold_dir, tracking_dir, "target_2", training_data)

            # a new feature column shows up in the gold layer after training
            training_data_with_new_col = training_data.copy()
            training_data_with_new_col["f3_new"] = 1
            training_data_with_new_col.to_csv(os.path.join(gold_dir, "target_2__training_set.csv"), index=False)

            inference = TabularModelInference.from_registry(
                data_dir=test_dir, tracking_dir=tracking_dir, registered_model="target_2_ridge", model_version=1
            )
            run_id = inference.setup_mlflow(phase="test", run_name_suffix="test")
            csv_path = inference.build_predictions_csv_path(os.path.join(test_dir, "predictions"), inference.registered_model, inference.model_version, run_id, split="test")
            preds_df = inference.evaluate(run_id=run_id, csv_path=csv_path, split="test")

            test_rows = training_data[training_data["target_season"] == 2024]
            expected = pipeline.predict(test_rows[["f1", "f2"]])
            pd.testing.assert_series_equal(
                pd.Series(expected, index=test_rows.index).sort_index(),
                preds_df["predictions"].sort_index(),
                check_names=False,
            )
        finally:
            shutil.rmtree(test_dir)

    def test_evaluate__raises_a_clear_error_if_a_feature_column_the_model_needs_is_gone(self):
        # Simulates a training-set column being renamed/removed after the model was trained --
        # unlike new columns, this can't be recovered from and should fail loudly.
        test_dir = tempfile.mkdtemp()
        try:
            gold_dir = os.path.join(test_dir, "gold")
            tracking_dir = os.path.join(test_dir, "mlruns")
            os.makedirs(gold_dir)

            training_data = _build_training_data({
                'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            })
            _train_and_register_ridge_model(gold_dir, tracking_dir, "target_2", training_data)

            training_data_missing_col = training_data.drop(columns=["f1"])
            training_data_missing_col.to_csv(os.path.join(gold_dir, "target_2__training_set.csv"), index=False)

            inference = TabularModelInference.from_registry(
                data_dir=test_dir, tracking_dir=tracking_dir, registered_model="target_2_ridge", model_version=1
            )
            run_id = inference.setup_mlflow(phase="test", run_name_suffix="test")
            csv_path = inference.build_predictions_csv_path(os.path.join(test_dir, "predictions"), inference.registered_model, inference.model_version, run_id, split="test")
            with pytest.raises(ValueError, match="f1"):
                inference.evaluate(run_id=run_id, csv_path=csv_path, split="test")
        finally:
            shutil.rmtree(test_dir)

    def test_evaluate__respects_position_filtering_for_a_positional_model(self):
        test_dir = tempfile.mkdtemp()
        try:
            gold_dir = os.path.join(test_dir, "gold")
            tracking_dir = os.path.join(test_dir, "mlruns")
            os.makedirs(gold_dir)

            training_data = _build_training_data({"f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
            training_data["position"] = ["RB", "WR"] * 5
            training_data.to_csv(os.path.join(gold_dir, "target_3__training_set.csv"), index=False)

            data_prep = TabularModelDataPrep(
                data_dir=test_dir, config={"target": "target_3", "positions": ["RB"]}
            )
            model = TabularModelTrainer(
                data_dir=test_dir, tracking_dir=tracking_dir, data_prep=data_prep, model_type="ridge"
            )
            train_run_id = model.setup_mlflow()
            model.fit_model(run_id=train_run_id)

            inference = TabularModelInference.from_registry(
                data_dir=test_dir, tracking_dir=tracking_dir, registered_model="target_3_rb_ridge", model_version=1
            )
            run_id = inference.setup_mlflow(phase="test", run_name_suffix="test")
            csv_path = inference.build_predictions_csv_path(os.path.join(test_dir, "predictions"), inference.registered_model, inference.model_version, run_id, split="test")
            preds_df = inference.evaluate(run_id=run_id, csv_path=csv_path, split="test")

            assert set(preds_df["position"]) == {"RB"}
        finally:
            shutil.rmtree(test_dir)

    # -- predict -----------------------------------------------------------------------

    def test_predict__predicts_against_the_live_prediction_set(self):
        run_id = self.inference.setup_mlflow(phase="predict", run_name_suffix="predictions")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id)
        preds_df = self.inference.predict(run_id=run_id, csv_path=csv_path)

        assert set(preds_df["target_season"]) == {2025}
        assert len(preds_df) == 2

    def test_predict__predictions_match_the_source_pipeline_predicting_on_resolved_features(self):
        run_id = self.inference.setup_mlflow(phase="predict", run_name_suffix="predictions")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id)
        preds_df = self.inference.predict(run_id=run_id, csv_path=csv_path)

        prediction_set = self.data_prep.prep_prediction_data()
        expected = pd.Series(
            self.pipeline.predict(prediction_set["features"][["f1", "f2"]]),
            index=prediction_set["features"].index,
        )

        pd.testing.assert_series_equal(
            expected.sort_index(), preds_df["predictions"].sort_index(), check_names=False
        )

    def test_predict__does_not_log_score_metrics(self):
        # There's no ground truth for the live prediction set, so no r2/rmse should be logged.
        run_id = self.inference.setup_mlflow(phase="predict", run_name_suffix="predictions")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id)
        self.inference.predict(run_id=run_id, csv_path=csv_path)

        run = mlflow.get_run(run_id)
        assert "r2" not in run.data.metrics
        assert "rmse" not in run.data.metrics

    def test_predict__writes_a_predictions_csv_at_the_given_path(self):
        run_id = self.inference.setup_mlflow(phase="predict", run_name_suffix="predictions")
        csv_path = self.inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), self.inference.registered_model, self.inference.model_version, run_id)
        self.inference.predict(run_id=run_id, csv_path=csv_path)

        assert os.path.exists(csv_path)

    def test_predict__respects_position_filtering(self):
        # A positional model's live prediction set is filtered down to just its own
        # position(s) too, same as the training data -- this is also what fixes the bug
        # where a positional model couldn't be used for live predictions at all (its
        # registered name encodes the position, and the prediction set must match the
        # columns/rows it was trained on).
        test_dir = tempfile.mkdtemp()
        try:
            gold_dir = os.path.join(test_dir, "gold")
            tracking_dir = os.path.join(test_dir, "mlruns")
            os.makedirs(gold_dir)

            training_data = _build_training_data({"f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
            training_data["position"] = ["RB", "WR"] * 5
            training_data.to_csv(os.path.join(gold_dir, "target_3__training_set.csv"), index=False)

            prediction_data = _build_prediction_data({"f1": [1, 2, 3, 4]}, n=4)
            prediction_data["position"] = ["RB", "WR", "RB", "WR"]
            prediction_data.to_csv(os.path.join(gold_dir, "target_3__prediction_set.csv"), index=False)

            data_prep = TabularModelDataPrep(
                data_dir=test_dir, config={"target": "target_3", "positions": ["RB"]}
            )
            model = TabularModelTrainer(
                data_dir=test_dir, tracking_dir=tracking_dir, data_prep=data_prep, model_type="ridge"
            )
            train_run_id = model.setup_mlflow()
            model.fit_model(run_id=train_run_id)

            inference = TabularModelInference.from_registry(
                data_dir=test_dir, tracking_dir=tracking_dir, registered_model="target_3_rb_ridge", model_version=1
            )
            run_id = inference.setup_mlflow(phase="predict", run_name_suffix="predictions")
            csv_path = inference.build_predictions_csv_path(os.path.join(test_dir, "predictions"), inference.registered_model, inference.model_version, run_id)
            preds_df = inference.predict(run_id=run_id, csv_path=csv_path)

            assert set(preds_df["position"]) == {"RB"}
            assert len(preds_df) == 2
        finally:
            shutil.rmtree(test_dir)

    def test_predict__never_needs_the_training_set_on_disk(self):
        # The whole point of TabularModelInference.predict() calling data_prep.prep_prediction_data()
        # directly (never data_prep.prep_training_data()): a TabularModelInference built purely to
        # make live predictions from an already-trained model should never need to load (or
        # even have) a training set on disk.
        training_set_path = os.path.join(self.gold_dir, "target_1__training_set.csv")
        moved_path = training_set_path + ".moved"
        os.rename(training_set_path, moved_path)
        try:
            inference = TabularModelInference.from_registry(
                data_dir=self.test_dir, tracking_dir=self.tracking_dir, registered_model="target_1_ridge", model_version=1
            )
            run_id = inference.setup_mlflow(phase="predict", run_name_suffix="predictions")
            csv_path = inference.build_predictions_csv_path(os.path.join(self.test_dir, "predictions"), inference.registered_model, inference.model_version, run_id)
            preds_df = inference.predict(run_id=run_id, csv_path=csv_path)

            assert len(preds_df) == 2
            assert not hasattr(inference.data_prep, "training_data")
        finally:
            os.rename(moved_path, training_set_path)


class TestTabularModelInferencePredictAndScore:
    """Tests _predict/_score directly (via a TabularModelInference.from_fit_pipeline built on a
    _FakePipeline/_FakeDataPrep) -- these are the low-level methods evaluate()/predict()
    call, exercised in isolation rather than through a real TabularModelTrainer."""

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

    def _inference(self, pipeline=None, target: str = "ppr") -> TabularModelInference:
        return TabularModelInference.from_fit_pipeline(
            pipeline or _FakePipeline([1.0]), _FakeDataPrep(target=target)
        )

    # -- _score ----------------------------------------------------------------------------

    def test_score__none_top_n_scores_the_whole_set_under_unprefixed_names(self):
        preds_df = pd.DataFrame({
            "predictions": [1.0, 2.0, 3.0, 8.0, 22.0, 33.0],
            "actual": [1, 2, 3, 10, 20, 30],
        })
        run_id = self._new_run()

        self._inference()._score(preds_df, run_id)

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

        self._inference()._score(preds_df, run_id, top_n_rows=[3])

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

        self._inference()._score(preds_df, run_id, top_n_rows=[100])

        metrics = mlflow.get_run(run_id).data.metrics
        assert "top_100_r2" in metrics
        assert "top_100_rmse" in metrics

    def test_score__skips_r2_and_rmse_when_fewer_than_two_rows_available(self):
        preds_df = pd.DataFrame({"predictions": [8.0], "actual": [10]})
        run_id = self._new_run()

        self._inference()._score(preds_df, run_id, top_n_rows=[5])

        metrics = mlflow.get_run(run_id).data.metrics
        assert "top_5_rmse" not in metrics
        assert "top_5_r2" not in metrics

    def test_score__no_top_n_rows_only_logs_the_overall_metrics(self):
        preds_df = pd.DataFrame({
            "predictions": [1.0, 2.0, 3.0],
            "actual": [1, 2, 3],
        })
        run_id = self._new_run()

        self._inference()._score(preds_df, run_id)

        metrics = mlflow.get_run(run_id).data.metrics
        assert not any(key.startswith("top_") for key in metrics)
        assert "r2" in metrics
        assert "rmse" in metrics

    def test_score__does_not_misinterpret_a_reordered_index_as_labels(self):
        # Regression test: _predict sorts its returned frame, giving it a non-default index
        # -- _score must treat top-n selection positionally, not by pandas label.
        preds_df = pd.DataFrame(
            {"predictions": [33.0, 22.0, 8.0, 3.0, 2.0, 1.0], "actual": [30, 20, 10, 3, 2, 1]},
            index=[5, 4, 3, 2, 1, 0],
        )
        run_id = self._new_run()

        self._inference()._score(preds_df, run_id, top_n_rows=[3])

        metrics = mlflow.get_run(run_id).data.metrics
        y_top, y_pred_top = np.array([30.0, 20.0, 10.0]), np.array([33.0, 22.0, 8.0])
        expected_rmse = np.sqrt(np.mean((y_top - y_pred_top) ** 2))
        assert metrics["top_3_rmse"] == pytest.approx(expected_rmse)

    # -- _predict ----------------------------------------------------------------------------

    def test_predict__sorts_by_predictions_descending_when_no_actual_given(self):
        identity = pd.DataFrame({
            "player_display_name": ["a", "b", "c"],
            "target_season": [2024, 2024, 2024],
        })
        run_id = self._new_run()
        csv_path = os.path.join(self.predictions_dir, "no_actual.csv")

        preds_df = self._inference(_FakePipeline([1.0, 3.0, 2.0]))._predict(
            X=pd.DataFrame({"f1": [1, 2, 3]}),
            identity=identity,
            run_id=run_id,
            csv_path=csv_path,
        )

        assert list(preds_df["predictions"]) == [3.0, 2.0, 1.0]
        assert "actual" not in preds_df.columns

    def test_predict__includes_actual_and_sorts_by_target_season_predictions_actual_when_given(self):
        identity = pd.DataFrame({
            "player_display_name": ["a", "b", "c"],
            "target_season": [2023, 2024, 2024],
        })
        y = pd.Series([10, 30, 20])
        run_id = self._new_run()
        csv_path = os.path.join(self.predictions_dir, "with_actual.csv")

        preds_df = self._inference(_FakePipeline([1.0, 3.0, 2.0]))._predict(
            X=pd.DataFrame({"f1": [1, 2, 3]}),
            identity=identity,
            run_id=run_id,
            csv_path=csv_path,
            y=y,
        )

        # target_season 2024 rows come first (descending), then ordered by predictions/actual
        assert list(preds_df["player_display_name"]) == ["b", "c", "a"]
        assert list(preds_df["actual"]) == [30, 20, 10]

    def test_predict__writes_csv_with_rounded_renamed_target_column(self):
        identity = pd.DataFrame({"player_display_name": ["a"], "target_season": [2024]})
        run_id = self._new_run()
        csv_path = os.path.join(self.predictions_dir, "rounded.csv")

        self._inference(_FakePipeline([1.23456]))._predict(
            X=pd.DataFrame({"f1": [1]}),
            identity=identity,
            run_id=run_id,
            csv_path=csv_path,
        )

        written = pd.read_csv(csv_path)
        assert list(written.columns) == ["player_display_name", "target_season", "ppr"]
        assert written["ppr"].iloc[0] == pytest.approx(1.23)

    def test_predict__csv_includes_actual_column_only_when_y_is_given(self):
        identity = pd.DataFrame({"player_display_name": ["a"], "target_season": [2024]})
        run_id = self._new_run()

        with_y_path = os.path.join(self.predictions_dir, "with_y.csv")
        self._inference(_FakePipeline([1.0]))._predict(
            X=pd.DataFrame({"f1": [1]}),
            identity=identity,
            run_id=run_id,
            csv_path=with_y_path,
            y=pd.Series([5]),
        )
        assert "actual" in pd.read_csv(with_y_path).columns

        without_y_path = os.path.join(self.predictions_dir, "without_y.csv")
        self._inference(_FakePipeline([1.0]))._predict(
            X=pd.DataFrame({"f1": [1]}),
            identity=identity,
            run_id=run_id,
            csv_path=without_y_path,
        )
        assert "actual" not in pd.read_csv(without_y_path).columns

    def test_predict__logs_the_csv_as_an_mlflow_artifact(self):
        identity = pd.DataFrame({"player_display_name": ["a"], "target_season": [2024]})
        run_id = self._new_run()
        csv_path = os.path.join(self.predictions_dir, "artifact.csv")

        self._inference(_FakePipeline([1.0]))._predict(
            X=pd.DataFrame({"f1": [1]}),
            identity=identity,
            run_id=run_id,
            csv_path=csv_path,
        )

        mlflow.set_tracking_uri(self.tracking_dir)
        downloaded_path = mlflow.artifacts.download_artifacts(run_id=run_id)
        assert os.path.exists(downloaded_path)
        assert "artifact.csv" in os.listdir(downloaded_path)
