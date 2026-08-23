import pandas as pd
import os
import pytest
import shutil
import tempfile
from typing import Tuple

import mlflow
from sklearn.pipeline import Pipeline

from src.modeling.data_prep import TabularModelDataPrep
from src.modeling.tabular_models import TabularModel
from src.modeling.tabular_model_test_set_eval import TabularModelTestSetEvaluator
from src.modeling.utils import setup_mlflow_run
from src.processing.column_registry import get_identity_columns


def _build_training_data(feature_cols: dict[str, list]) -> pd.DataFrame:
    n = 10
    identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
    identity_data["target_season"] = [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024]

    return pd.DataFrame({
        **identity_data,
        **feature_cols,
        'target': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    })


def _train_and_register_ridge_model(
    gold_dir: str, tracking_dir: str, target: str, training_data: pd.DataFrame
) -> Tuple[Pipeline, str]:
    """Trains+registers a ridge model on training_data (eval/test years=1, no exclusions),
    logging its data prep config the same way tabular_models.py's CLI (main()) does.

    Returns (pipeline, source_run_id)."""
    training_data.to_csv(os.path.join(gold_dir, f"{target}__training_set.csv"), index=False)

    data_prep = TabularModelDataPrep(data_dir=os.path.dirname(gold_dir), config={"target": target})
    model = TabularModel(
        data_dir=os.path.dirname(gold_dir), tracking_dir=tracking_dir, data_prep=data_prep, model_type="ridge"
    )
    run_id = model.setup_mlflow()
    pipeline = model.fit_model(run_id=run_id)
    model.eval_model(pipeline, run_id)

    return pipeline, run_id


class TestTabularModelTestSetEvaluator:
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

        # Simulate what tabular_models.py's CLI (main()) does when training+registering a
        # model: split (per a data prep config), fit, log the config + eval, all under one run.
        cls.config = {
            "target": "target_1",
            "features": {"mode": "exclude", "columns": ["f3"]},
            "split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 2},
        }
        cls.data_prep = TabularModelDataPrep(data_dir=cls.test_dir, config=cls.config)
        model = TabularModel(
            data_dir=cls.test_dir, tracking_dir=cls.tracking_dir, data_prep=cls.data_prep, model_type="ridge"
        )
        cls.data = model.data
        cls.train_run_id = model.setup_mlflow()
        cls.pipeline = model.fit_model(run_id=cls.train_run_id)
        model.eval_model(cls.pipeline, cls.train_run_id)

        cls.evaluator = TabularModelTestSetEvaluator(
            data_dir=cls.test_dir, tracking_dir=cls.tracking_dir, registered_model="target_1_ridge", model_version=1
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_init__does_not_build_a_tabular_model(self):
        # The evaluator has everything it needs (pipeline, reconstructed data_prep) to run
        # predict()/score() directly -- it shouldn't need to build a TabularModel at all.
        assert not hasattr(self.evaluator, "model")

    def test_init__loads_the_pipeline_and_source_run_from_the_registered_model(self):
        assert self.evaluator.model_version == 1
        assert self.evaluator.source_run_id == self.train_run_id
        assert self.evaluator.config == self.config

    def test_evaluate__reconstructs_the_same_test_split_the_model_was_trained_with(self):
        preds_df = self.evaluator.evaluate()

        # test split held out 2024 (most recent season), per the logged config
        assert set(preds_df["target_season"]) == {2024}
        assert len(preds_df) == len(self.data["X_test"])

    def test_evaluate__predictions_match_the_source_pipeline_predicting_on_X_test(self):
        preds_df = self.evaluator.evaluate()

        expected = pd.Series(
            self.pipeline.predict(self.data["X_test"]), index=self.data["X_test"].index
        )
        actual = preds_df["predictions"]

        pd.testing.assert_series_equal(
            expected.sort_index(), actual.sort_index(), check_names=False
        )

    def test_evaluate__logs_a_new_run_tagged_phase_test_linked_to_the_source_run(self):
        self.evaluator.evaluate()

        mlflow.set_tracking_uri(self.tracking_dir)
        experiment = mlflow.get_experiment_by_name("target_1_tabular")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.phase = 'test'",
            output_format="pandas",
        )

        assert len(runs) >= 1
        latest_test_run = runs.iloc[0]
        assert latest_test_run["params.source_run_id"] == self.train_run_id
        assert latest_test_run["params.model_version"] == "1"
        assert "metrics.r2" in latest_test_run
        assert "metrics.rmse" in latest_test_run

    def test_evaluate__test_run_also_logs_the_reconstructed_data_prep_config(self):
        self.evaluator.evaluate()

        mlflow.set_tracking_uri(self.tracking_dir)
        experiment = mlflow.get_experiment_by_name("target_1_tabular")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.phase = 'test'",
            output_format="pandas",
        )
        latest_test_run_id = runs.iloc[0]["run_id"]

        logged_config = mlflow.artifacts.load_dict(f"runs:/{latest_test_run_id}/data_prep_config.json")
        assert logged_config == self.config

    def test_evaluate__writes_a_predictions_csv_named_with_registered_model_and_version(self):
        self.evaluator.evaluate()

        predictions_dir = os.path.join(self.test_dir, "predictions")
        matching = [f for f in os.listdir(predictions_dir) if f.startswith("target_1_ridge_v1_test_predictions_")]
        assert len(matching) >= 1

    def test_load_data_prep_config__returns_the_config_logged_on_the_given_run(self):
        config = self.evaluator._load_data_prep_config(self.train_run_id)
        assert config == self.config

    def test_evaluate__raises_if_the_source_run_has_no_data_prep_config_artifact(self):
        # A run created via fit_model directly (e.g. param_search's children) never gets
        # setup_mlflow's data_prep_config.json artifact logged onto it by itself -- only a
        # run created via setup_mlflow does, and here we skip that entirely.
        data_prep = TabularModelDataPrep(data_dir=self.test_dir, config={"target": "target_1"})
        model = TabularModel(
            data_dir=self.test_dir, tracking_dir=self.tracking_dir, data_prep=data_prep, model_type="lasso"
        )

        run_id = setup_mlflow_run(
            experiment_name="target_1_tabular", run_name="bare_run", tracking_dir=self.tracking_dir,
        )
        model.fit_model(run_id=run_id)

        with pytest.raises(RuntimeError, match="data_prep_config"):
            self.evaluator._load_data_prep_config(run_id)

    def test_evaluate__ignores_feature_columns_added_to_the_training_set_after_training(self):
        # Simulates Phase 2's incremental feature growth: a column is added to the gold
        # training set on disk after the model was trained/registered, without retraining.
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

            evaluator = TabularModelTestSetEvaluator(
                data_dir=test_dir, tracking_dir=tracking_dir, registered_model="target_2_ridge", model_version=1
            )
            preds_df = evaluator.evaluate()

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

            evaluator = TabularModelTestSetEvaluator(
                data_dir=test_dir, tracking_dir=tracking_dir, registered_model="target_2_ridge", model_version=1
            )
            with pytest.raises(ValueError, match="f1"):
                evaluator.evaluate()
        finally:
            shutil.rmtree(test_dir)

    def test_evaluate__uses_the_positional_models_own_experiment_when_positions_are_set(self):
        # A positional model's training run lives in its own "{target}_{positions}_tabular"
        # experiment (see TabularModel.base_model_name) -- evaluate() must log the test run
        # into that same experiment (found by looking up which experiment source_run_id
        # itself lives in), not the general one.
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
            model = TabularModel(
                data_dir=test_dir, tracking_dir=tracking_dir, data_prep=data_prep, model_type="ridge"
            )
            train_run_id = model.setup_mlflow()
            model.fit_model(run_id=train_run_id)

            evaluator = TabularModelTestSetEvaluator(
                data_dir=test_dir, tracking_dir=tracking_dir, registered_model="target_3_rb_ridge", model_version=1
            )
            preds_df = evaluator.evaluate()

            assert set(preds_df["position"]) == {"RB"}

            mlflow.set_tracking_uri(tracking_dir)
            experiment = mlflow.get_experiment_by_name("target_3_rb_tabular")
            assert experiment is not None
            test_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.phase = 'test'",
                output_format="pandas",
            )
            assert test_runs.iloc[0]["params.source_run_id"] == train_run_id
        finally:
            shutil.rmtree(test_dir)
