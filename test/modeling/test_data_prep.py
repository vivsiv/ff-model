import os
import shutil
import tempfile

import pandas as pd
import pytest
import yaml

from src.modeling.data_prep import TabularModelDataPrep
from src.processing.column_registry import get_identity_columns


def _build_training_data() -> pd.DataFrame:
    n = 10
    identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
    identity_data["target_season"] = [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024]

    return pd.DataFrame({
        **identity_data,
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "f2": [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
        "receiving_yards": [12, 0, 8, 12, 0, 8, 12, 0, 8, 12],
        "target": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    })


class TestTabularModelDataPrep:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.gold_dir = os.path.join(cls.test_dir, "gold")
        os.makedirs(cls.gold_dir)

        cls.training_data = _build_training_data()
        cls.training_data.to_csv(os.path.join(cls.gold_dir, "target_1__training_set.csv"), index=False)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def _build(self, config: dict) -> TabularModelDataPrep:
        return TabularModelDataPrep(data_dir=self.test_dir, config={"target": "target_1", **config})

    def test_init__loads_target_season_gold_training_set(self):
        prep = self._build({})
        assert len(prep.training_data) == 10
        assert prep.target == "target_1"

    def test_init__sample_weights_defaults_to_uniform_one_when_omitted(self):
        prep = TabularModelDataPrep(data_dir=self.test_dir, config={"target": "target_1"})
        assert list(prep.sample_weights) == [1.0] * 10

    def test_resolve_feature_columns__defaults_to_every_non_identity_non_target_column(self):
        prep = self._build({})
        assert prep.feature_cols == ["f1", "f2", "receiving_yards"]

    def test_resolve_feature_columns__exclude_mode_drops_exact_matches(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["f2"]}})
        assert prep.feature_cols == ["f1", "receiving_yards"]

    def test_resolve_feature_columns__exclude_mode_drops_prefix_matches(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["receiving_*"]}})
        assert prep.feature_cols == ["f1", "f2"]

    def test_resolve_feature_columns__exclude_mode_does_not_treat_entries_as_prefixes_without_a_star(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["receiving"]}})
        assert prep.feature_cols == ["f1", "f2", "receiving_yards"]

    def test_resolve_feature_columns__include_mode_keeps_only_exact_and_prefix_matches(self):
        prep = self._build({"features": {"mode": "include", "columns": ["f1", "receiving_*"]}})
        assert prep.feature_cols == ["f1", "receiving_yards"]

    def test_resolve_feature_columns__unrecognized_mode_keeps_every_column(self):
        prep = self._build({"features": {"mode": "both", "columns": ["f1"]}})
        assert prep.feature_cols == ["f1", "f2", "receiving_yards"]

    def test_split__holds_out_most_recent_season_for_test_and_the_one_before_for_eval(self):
        prep = self._build({"split": {"eval_data_years": 1, "test_data_years": 1}})
        data = prep.split()

        assert set(data["identity_test"]["target_season"]) == {2024}
        assert set(data["identity_eval"]["target_season"]) == {2023}
        assert set(data["identity_train"]["target_season"]) == {2020, 2021, 2022}
        assert list(data["y_train"]) == [10, 11, 12, 13, 14, 15]
        assert list(data["y_eval"]) == [16, 17]
        assert list(data["y_test"]) == [18, 19]

    def test_split__num_training_seasons_limits_training_to_most_recent_n_seasons(self):
        prep = self._build({"split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 1}})
        data = prep.split()

        assert set(data["identity_train"]["target_season"]) == {2022}

    def test_split__raises_if_requested_seasons_exceed_available_seasons(self):
        prep = self._build({"split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 4}})
        with pytest.raises(ValueError):
            prep.split()

    def test_split__defaults_to_one_year_each_with_no_season_limit(self):
        prep = self._build({})
        data = prep.split()

        assert set(data["identity_test"]["target_season"]) == {2024}
        assert set(data["identity_eval"]["target_season"]) == {2023}
        assert set(data["identity_train"]["target_season"]) == {2020, 2021, 2022}

    def test_split__sample_weight_train_is_uniform_when_sample_weights_omitted(self):
        prep = self._build({})
        data = prep.split()

        assert list(data["sample_weight_train"]) == [1.0] * 6

    def test_split__sample_weight_train_uses_global_weight_when_no_buckets_are_given(self):
        prep = self._build({"sample_weights": {"global_weight": 0.5, "buckets": []}})
        data = prep.split()

        assert list(data["sample_weight_train"]) == [0.5] * 6

    def test_split__sample_weight_train_applies_matching_bucket(self):
        prep = self._build({
            "split": {"eval_data_years": 1, "test_data_years": 1},
            "sample_weights": {"buckets": [{"min": 0, "weight": 0.1}, {"min": 12, "weight": 1.0}]},
        })
        data = prep.split()

        # train targets [10, 11, 12, 13, 14, 15]: 10/11 < 12 -> min=0 bucket (0.1);
        # 12-15 >= 12 -> min=12 bucket (1.0)
        assert list(data["sample_weight_train"]) == [0.1, 0.1, 1.0, 1.0, 1.0, 1.0]

    def test_split__sample_weight_train_uses_the_largest_qualifying_min_bucket(self):
        prep = self._build({
            "sample_weights": {
                # listed out of order on purpose -- order in the config shouldn't matter.
                "buckets": [{"min": 12, "weight": 0.5}, {"min": 0, "weight": 0.1}, {"min": 14, "weight": 1.0}],
            },
        })
        data = prep.split()

        # train targets [10, 11, 12, 13, 14, 15]:
        # 10, 11 only qualify for min=0 -> 0.1
        # 12, 13 qualify for min=0 and min=12 -> largest (min=12) wins -> 0.5
        # 14, 15 qualify for all three -> largest (min=14) wins -> 1.0
        assert list(data["sample_weight_train"]) == pytest.approx([0.1, 0.1, 0.5, 0.5, 1.0, 1.0])

    def test_split__sample_weight_train_lowest_bucket_covers_values_below_its_own_min(self):
        prep = self._build({
            "sample_weights": {"buckets": [{"min": 12, "weight": 0.3}, {"min": 14, "weight": 1.0}]},
        })
        data = prep.split()

        # train targets [10, 11, 12, 13, 14, 15]: 10 and 11 are below the lowest bucket's own
        # min (12), but still get its weight (0.3) rather than being left unweighted.
        assert list(data["sample_weight_train"]) == pytest.approx([0.3, 0.3, 0.3, 0.3, 1.0, 1.0])

    def test_split__sample_weights_never_applied_to_eval_or_test(self):
        prep = self._build({"sample_weights": {"buckets": [{"min": 0, "weight": 0.0}]}})
        data = prep.split()

        assert "sample_weight_eval" not in data
        assert "sample_weight_test" not in data

    def test_from_config_file__loads_yaml_and_builds_a_working_data_prep(self):
        config_path = os.path.join(self.test_dir, "target_1.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump({
                "target": "target_1",
                "features": {"mode": "include", "columns": ["f1"]},
                "split": {"eval_data_years": 1, "test_data_years": 1},
            }, f)

        prep = TabularModelDataPrep.from_config_file(data_dir=self.test_dir, config_path=config_path)

        assert prep.target == "target_1"
        assert prep.feature_cols == ["f1"]
        data = prep.split()
        assert list(data["X_train"].columns) == ["f1"]
