import pandas as pd
import os
import pytest
import shutil
import tempfile
import mlflow

from src.modeling.tabular_models import TabularModel
from src.modeling.tabular_model_test_set_eval import TabularModelTestSetEvaluator
from src.processing.column_registry import get_identity_columns


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
        # model: split, fit, log split params + eval, all under one run.
        cls.split_params = {
            "excluded_features": ["f3"],
            "eval_data_years": 1,
            "test_data_years": 1,
            "num_training_seasons": 2,
        }
        model = TabularModel(
            data_dir=cls.test_dir,
            tracking_dir=cls.tracking_dir,
            target="target_1",
            excluded_features=cls.split_params["excluded_features"],
        )
        cls.data = model.split_data(
            eval_data_years=cls.split_params["eval_data_years"],
            test_data_years=cls.split_params["test_data_years"],
            num_training_seasons=cls.split_params["num_training_seasons"],
        )
        cls.train_run_id = model.setup_mlflow("ridge", extra_params=cls.split_params)
        cls.pipeline = model.fit_model(cls.data, "ridge", cls.train_run_id)
        model.eval_model(cls.pipeline, cls.data, cls.train_run_id)

        cls.evaluator = TabularModelTestSetEvaluator(data_dir=cls.test_dir, tracking_dir=cls.tracking_dir, target="target_1")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_evaluate__reconstructs_the_same_test_split_the_model_was_trained_with(self):
        preds_df = self.evaluator.evaluate(model_type="ridge")

        # test split held out 2024 (most recent season), per the logged split params
        assert set(preds_df["target_season"]) == {2024}
        assert len(preds_df) == len(self.data["X_test"])

    def test_evaluate__predictions_match_the_source_pipeline_predicting_on_X_test(self):
        preds_df = self.evaluator.evaluate(model_type="ridge")

        expected = pd.Series(
            self.pipeline.predict(self.data["X_test"]), index=self.data["X_test"].index
        )
        actual = preds_df["predictions"]

        pd.testing.assert_series_equal(
            expected.sort_index(), actual.sort_index(), check_names=False
        )

    def test_evaluate__logs_a_new_run_tagged_phase_test_linked_to_the_source_run(self):
        self.evaluator.evaluate(model_type="ridge")

        mlflow.set_tracking_uri(self.tracking_dir)
        experiment = mlflow.get_experiment_by_name("target_1_tabular")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.phase = 'test'",
            output_format="pandas",
        )

        assert len(runs) >= 1
        latest_test_run = runs.iloc[0]
        assert latest_test_run["tags.model_type"] == "ridge"
        assert latest_test_run["params.source_run_id"] == self.train_run_id
        assert "metrics.r2" in latest_test_run
        assert "metrics.rmse" in latest_test_run

    def test_evaluate__raises_if_the_source_run_is_missing_split_params(self):
        # A run created via fit_model directly (e.g. param_search's children) never logs
        # excluded_features/eval_data_years/test_data_years/num_training_seasons -- only
        # tabular_models.py's CLI (main()) does.
        model = TabularModel(data_dir=self.test_dir, tracking_dir=self.tracking_dir, target="target_1")
        data = model.split_data()
        run_id = model.setup_mlflow("lasso")
        model.fit_model(data, "lasso", run_id)

        with pytest.raises(KeyError):
            self.evaluator.evaluate(model_type="lasso")

    def test_parse_split_data_params__casts_logged_string_params_back_to_expected_types(self):
        parsed = self.evaluator._parse_split_data_params({
            "excluded_features": "['fantasy_points*', 'ppr_points_per_game*']",
            "eval_data_years": "1",
            "test_data_years": "1",
            "num_training_seasons": "all",
        })

        assert parsed == {
            "excluded_features": ["fantasy_points*", "ppr_points_per_game*"],
            "eval_data_years": 1,
            "test_data_years": 1,
            "num_training_seasons": None,
        }

    def test_parse_split_data_params__raises_on_missing_keys(self):
        with pytest.raises(KeyError):
            self.evaluator._parse_split_data_params({"eval_data_years": "1"})
