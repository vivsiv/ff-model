import pandas as pd
import os
import pytest
import shutil
import tempfile

from src.modeling.tabular_models import TabularModel
from src.processing.column_registry import get_identity_columns


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

        cls.model = TabularModel(data_dir=cls.test_dir, tracking_dir=cls.tracking_dir, target="target_1")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_initial_datasets(self):
        model = TabularModel(data_dir=self.test_dir, tracking_dir=self.tracking_dir, target="target_1")

        expected_features = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            'f3': [12, 0, 8, 12, 0, 8, 12, 0, 8, 12]
        })
        expected_target = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], name="target")

        assert model.features_df.equals(expected_features)
        assert model.target_df.equals(expected_target)
        assert list(model.identity_df.columns) == get_identity_columns("nflverse", "player_stats") + ["target_season"]
        assert len(model.identity_df) == 10
        assert model.target == "target_1"

    def test_initial_datasets__excludes_features_by_exact_match_and_prefix(self):
        model = TabularModel(
            data_dir=self.test_dir,
            tracking_dir=self.tracking_dir,
            target="target_1",
            excluded_features=["f1", "f3"],
        )

        expected_features = pd.DataFrame({
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
        })

        assert model.features_df.equals(expected_features)
        assert model.feature_cols == ["f2"]

    def test_initial_datasets__excludes_features_by_prefix_when_denoted_with_a_star(self):
        model = TabularModel(
            data_dir=self.test_dir,
            tracking_dir=self.tracking_dir,
            target="target_1",
            excluded_features=["f*"],  # trailing "*" should catch f1, f2, and f3
        )

        assert model.feature_cols == []

    def test_initial_datasets__does_not_treat_features_as_prefixes_without_a_star(self):
        model = TabularModel(
            data_dir=self.test_dir,
            tracking_dir=self.tracking_dir,
            target="target_1",
            excluded_features=["f"],  # no trailing "*", so this shouldn't match f1/f2/f3
        )

        assert model.feature_cols == ["f1", "f2", "f3"]

    def test_split_data__holds_out_most_recent_season_for_test_and_the_one_before_for_eval(self):
        data = self.model.split_data(eval_data_years=1, test_data_years=1)

        identity_col_count = len(get_identity_columns("nflverse", "player_stats")) + 1  # + target_season

        assert data['X_train'].shape == (6, 3)
        assert data['X_eval'].shape == (2, 3)
        assert data['X_test'].shape == (2, 3)
        assert data['y_train'].shape == (6,)
        assert data['y_eval'].shape == (2,)
        assert data['y_test'].shape == (2,)
        assert data['identity_train'].shape == (6, identity_col_count)
        assert data['identity_eval'].shape == (2, identity_col_count)
        assert data['identity_test'].shape == (2, identity_col_count)

        # test must be exactly the most recent season (2024), eval the one before it (2023),
        # nothing older
        assert set(data['identity_test']['target_season']) == {2024}
        assert set(data['identity_eval']['target_season']) == {2023}
        assert set(data['identity_train']['target_season']) == {2020, 2021, 2022}

    def test_split_data__eval_data_years_controls_how_much_is_held_out(self):
        data = self.model.split_data(eval_data_years=2, test_data_years=1)

        assert data['X_train'].shape == (4, 3)
        assert data['X_eval'].shape == (4, 3)
        assert data['X_test'].shape == (2, 3)
        assert set(data['identity_test']['target_season']) == {2024}
        assert set(data['identity_eval']['target_season']) == {2022, 2023}
        assert set(data['identity_train']['target_season']) == {2020, 2021}

    def test_split_data__test_data_years_controls_how_much_is_held_out(self):
        data = self.model.split_data(eval_data_years=1, test_data_years=2)

        assert data['X_train'].shape == (4, 3)
        assert data['X_eval'].shape == (2, 3)
        assert data['X_test'].shape == (4, 3)
        assert set(data['identity_test']['target_season']) == {2023, 2024}
        assert set(data['identity_eval']['target_season']) == {2022}
        assert set(data['identity_train']['target_season']) == {2020, 2021}

    def test_split_data__uses_defaults_of_one_year_each(self):
        data = self.model.split_data()

        assert set(data['identity_test']['target_season']) == {2024}
        assert set(data['identity_eval']['target_season']) == {2023}
        assert set(data['identity_train']['target_season']) == {2020, 2021, 2022}

    def test_split_data__num_training_seasons_limits_training_to_most_recent_n_seasons(self):
        data = self.model.split_data(eval_data_years=1, test_data_years=1, num_training_seasons=2)

        assert data['X_train'].shape == (4, 3)
        assert set(data['identity_test']['target_season']) == {2024}
        assert set(data['identity_eval']['target_season']) == {2023}
        # only the 2 most recent training seasons (2021, 2022) are kept; 2020 is dropped
        assert set(data['identity_train']['target_season']) == {2021, 2022}

    def test_split_data__num_training_seasons_none_keeps_every_older_season(self):
        data = self.model.split_data(eval_data_years=1, test_data_years=1, num_training_seasons=None)

        assert set(data['identity_train']['target_season']) == {2020, 2021, 2022}

    def test_split_data__raises_if_requested_seasons_exceed_available_seasons(self):
        # 5 distinct seasons available (2020-2024); requesting 3 + 1 + 1 = 5 is fine...
        self.model.split_data(eval_data_years=1, test_data_years=1, num_training_seasons=3)

        # ...but 4 + 1 + 1 = 6 exceeds the 5 available
        with pytest.raises(ValueError):
            self.model.split_data(eval_data_years=1, test_data_years=1, num_training_seasons=4)

    def test_split_data_is_deterministic(self):
        data1 = self.model.split_data(eval_data_years=1, test_data_years=1)
        data2 = self.model.split_data(eval_data_years=1, test_data_years=1)

        pd.testing.assert_frame_equal(data1['X_train'], data2['X_train'])
        pd.testing.assert_frame_equal(data1['X_eval'], data2['X_eval'])
        pd.testing.assert_frame_equal(data1['X_test'], data2['X_test'])
        pd.testing.assert_series_equal(data1['y_train'], data2['y_train'])
        pd.testing.assert_series_equal(data1['y_eval'], data2['y_eval'])
        pd.testing.assert_series_equal(data1['y_test'], data2['y_test'])
        pd.testing.assert_frame_equal(data1['identity_train'], data2['identity_train'])
        pd.testing.assert_frame_equal(data1['identity_eval'], data2['identity_eval'])
        pd.testing.assert_frame_equal(data1['identity_test'], data2['identity_test'])
