import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from src.modeling.tabular.data_prep import SplitYears, TabularModelDataPrep
from src.processing.column_registry import get_identity_columns


def _build_training_data() -> pd.DataFrame:
    n = 10
    identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
    identity_data["position"] = ["QB", "TE", "RB", "WR", "QB", "TE", "RB", "QB", "WR", "TE"]
    identity_data["target_season"] = [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024]

    return pd.DataFrame({
        **identity_data,
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "f2": [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
        "receiving_yards": [12, 0, 8, 12, 0, 8, 12, 0, 8, 12],
        "target": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    })


def _build_prediction_data() -> pd.DataFrame:
    n = 4
    identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
    identity_data["position"] = ["QB", "RB", "WR", "TE"]
    identity_data["target_season"] = [2025, 2025, 2025, 2025]

    return pd.DataFrame({
        **identity_data,
        "f1": [11, 12, 13, 14],
        "f2": [1, 2, 3, 4],
        "receiving_yards": [5, 6, 7, 8],
        "target": [None, None, None, None],
    })


class TestTabularModelDataPrep:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.gold_dir = os.path.join(cls.test_dir, "gold")
        os.makedirs(cls.gold_dir)

        cls.training_data = _build_training_data()
        cls.training_data.to_csv(os.path.join(cls.gold_dir, "target_1__training_set.csv"), index=False)

        cls.prediction_data = _build_prediction_data()
        cls.prediction_data.to_csv(os.path.join(cls.gold_dir, "target_1__prediction_set.csv"), index=False)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def _build(self, config: dict) -> TabularModelDataPrep:
        prep = TabularModelDataPrep(data_dir=self.test_dir, config={"target": "target_1", **config})
        prep.prep_training_data()
        return prep

    def test_init__loads_gold_data_correctly(self):
        prep = self._build(config={})
        assert len(prep.training_data) == 10

    def test_init__sets_base_fields_correctly(self):
        prep = self._build(config={})
        assert prep.target == "target_1"
        assert prep.positions is None

    def test_prep_training_data__returns_the_train_eval_test_split(self):
        # prep_training_data() calls split() itself and returns the result, so consumers
        # (TabularModelTrainer, TabularModelInference) don't need to call .split() themselves.
        prep = TabularModelDataPrep(data_dir=self.test_dir, config={"target": "target_1"})
        data = prep.prep_training_data()
        assert set(data.keys()) == {
            "X_train", "X_eval", "X_test",
            "y_train", "y_eval", "y_test",
            "identity_train", "identity_eval", "identity_test",
            "sample_weight_train",
        }

    def test_init__filters_to_a_single_configured_position(self):
        prep = self._build(config={"positions": ["RB"]})
        assert len(prep.training_data) == 2
        assert set(prep.training_data["position"]) == {"RB"}

    def test_init__filters_to_multiple_configured_positions(self):
        prep = self._build(config={"positions": ["RB", "WR"]})
        assert len(prep.training_data) == 4
        assert set(prep.training_data["position"]) == {"RB", "WR"}

    def test_init__empty_positions_list_is_treated_like_omitted(self):
        prep = self._build({"positions": []})
        assert len(prep.training_data) == 10
        assert prep.positions is None

    def test_compute_sample_weights__defaults_to_uniform_one_when_omitted(self):
        prep = self._build({})
        assert list(prep._compute_sample_weights(prep.training_data["target"])) == [1.0] * 10

    def test_resolve_feature_columns__defaults_to_every_non_identity_non_target_column(self):
        prep = self._build({})
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1", "f2", "receiving_yards"]

    def test_resolve_feature_columns__exclude_mode_drops_exact_matches(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["f2"]}})
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1", "receiving_yards"]

    def test_resolve_feature_columns__exclude_mode_drops_prefix_wildcard_matches(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["receiving_*"]}})
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1", "f2"]

    def test_resolve_feature_columns__exclude_mode_drops_suffix_wildcard_matches(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["*_yards"]}})
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1", "f2"]

    def test_resolve_feature_columns__exclude_mode_drops_substring_wildcard_matches(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["*ceiving_ya*"]}})
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1", "f2"]

    def test_resolve_feature_columns__exclude_mode_does_not_treat_entries_without_a_star_as_wildcards(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["receiving"]}})
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1", "f2", "receiving_yards"]

    def test_resolve_feature_columns__include_mode_keeps_only_exact_and_prefix_matches(self):
        prep = self._build({"features": {"mode": "include", "columns": ["f1", "receiving_*"]}})
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1", "receiving_yards"]

    def test_resolve_feature_columns__unrecognized_mode_keeps_every_column(self):
        prep = self._build({"features": {"mode": "both", "columns": ["f1"]}})
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1", "f2", "receiving_yards"]

    def test_prep_prediction_data__splits_into_identity_and_resolved_feature_columns(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["f2"]}})
        prediction_set = prep.prep_prediction_data()

        assert list(prediction_set["features"].columns) == ["f1", "receiving_yards"]
        assert list(prediction_set["features"]["f1"]) == [11, 12, 13, 14]
        assert set(prediction_set["identity"]["position"]) == {"QB", "RB", "WR", "TE"}

    def test_prep_prediction_data__respects_position_filtering(self):
        prep = self._build({"positions": ["RB"]})
        prediction_set = prep.prep_prediction_data()

        assert set(prediction_set["identity"]["position"]) == {"RB"}
        assert list(prediction_set["features"]["f1"]) == [12]

    def test_prep_prediction_data__works_without_ever_calling_prep_training_data(self):
        # The whole point of splitting prep_training_data()/prep_prediction_data(): a
        # TabularModelDataPrep built purely to make live predictions from an already-trained
        # model should never need to load (or even have) a training set on disk.
        target = "target_predict_only"
        _build_prediction_data().to_csv(os.path.join(self.gold_dir, f"{target}__prediction_set.csv"), index=False)
        # deliberately not writing f"{target}__training_set.csv" at all

        prep = TabularModelDataPrep(data_dir=self.test_dir, config={"target": target})
        prediction_set = prep.prep_prediction_data()

        assert not hasattr(prep, "training_data")
        assert len(prediction_set["identity"]) == 4
        assert list(prediction_set["features"].columns) == ["f1", "f2", "receiving_yards"]

    def test_split__is_independently_callable_with_explicit_args(self):
        # split() is a pure staticmethod with no config/instance state -- directly testable
        # without a TabularModelDataPrep, prep_training_data(), or a CSV fixture at all.
        target_season = pd.Series([2020, 2020, 2021, 2021, 2022, 2022])
        identity_df = pd.DataFrame({"target_season": target_season})
        features_df = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
        target_df = pd.Series([10, 11, 12, 13, 14, 15])
        sample_weights = pd.Series([1.0] * 6)

        data = TabularModelDataPrep.split(
            split_years=SplitYears(eval_data_years=1, test_data_years=1),
            target_season=target_season,
            features_df=features_df,
            target_df=target_df,
            identity_df=identity_df,
            sample_weights=sample_weights,
        )

        assert set(data["identity_test"]["target_season"]) == {2022}
        assert set(data["identity_eval"]["target_season"]) == {2021}
        assert set(data["identity_train"]["target_season"]) == {2020}

    def test_split__holds_out_most_recent_season_for_test_and_the_one_before_for_eval(self):
        prep = self._build({"split": {"eval_data_years": 1, "test_data_years": 1}})
        data = prep.prep_training_data()

        assert set(data["identity_test"]["target_season"]) == {2024}
        assert set(data["identity_eval"]["target_season"]) == {2023}
        assert set(data["identity_train"]["target_season"]) == {2020, 2021, 2022}

        assert list(data["y_test"]) == [18, 19]
        assert list(data["y_eval"]) == [16, 17]
        assert list(data["y_train"]) == [10, 11, 12, 13, 14, 15]

    def test_split__supports_more_than_one_year_for_test_and_eval(self):
        prep = self._build({"split": {"eval_data_years": 2, "test_data_years": 2}})
        data = prep.prep_training_data()

        assert set(data["identity_test"]["target_season"]) == {2023, 2024}
        assert set(data["identity_eval"]["target_season"]) == {2021, 2022}
        assert set(data["identity_train"]["target_season"]) == {2020}

        assert list(data["y_test"]) == [16, 17, 18, 19]
        assert list(data["y_eval"]) == [12, 13, 14, 15]
        assert list(data["y_train"]) == [10, 11]

    def test_split__num_training_seasons_limits_training_to_most_recent_n_seasons(self):
        prep = self._build({"split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 1}})
        data = prep.prep_training_data()

        assert set(data["identity_train"]["target_season"]) == {2022}

    def test_split__raises_if_requested_seasons_exceed_available_seasons(self):
        # prep_training_data() calls split() itself during __init__, so a bad config raises
        # as soon as the TabularModelDataPrep is built, not later when .split() is called.
        with pytest.raises(ValueError):
            self._build({"split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 4}})

    def test_split__defaults_to_one_year_each_with_no_season_limit(self):
        prep = self._build({})
        data = prep.prep_training_data()

        assert set(data["identity_test"]["target_season"]) == {2024}
        assert set(data["identity_eval"]["target_season"]) == {2023}
        assert set(data["identity_train"]["target_season"]) == {2020, 2021, 2022}

    def test_split__sample_weight_train_is_uniform_when_sample_weights_omitted(self):
        prep = self._build({})
        data = prep.prep_training_data()

        assert list(data["sample_weight_train"]) == [1.0] * 6

    def test_split__sample_weight_train_uses_global_weight_when_no_buckets_are_given(self):
        prep = self._build({"sample_weights": {"global_weight": 0.5, "buckets": []}})
        data = prep.prep_training_data()

        assert list(data["sample_weight_train"]) == [0.5] * 6

    def test_split__sample_weight_train_applies_matching_bucket(self):
        prep = self._build({
            "split": {"eval_data_years": 1, "test_data_years": 1},
            "sample_weights": {"buckets": [{"min": 0, "weight": 0.1}, {"min": 12, "weight": 1.0}]},
        })
        data = prep.prep_training_data()

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
        data = prep.prep_training_data()

        # train targets [10, 11, 12, 13, 14, 15]:
        # 10, 11 only qualify for min=0 -> 0.1
        # 12, 13 qualify for min=0 and min=12 -> largest (min=12) wins -> 0.5
        # 14, 15 qualify for all three -> largest (min=14) wins -> 1.0
        assert list(data["sample_weight_train"]) == pytest.approx([0.1, 0.1, 0.5, 0.5, 1.0, 1.0])

    def test_split__sample_weight_train_rows_below_the_lowest_bucket_get_zero_weight(self):
        prep = self._build({
            "sample_weights": {"buckets": [{"min": 12, "weight": 0.3}, {"min": 14, "weight": 1.0}]},
        })
        data = prep.prep_training_data()

        # train targets [10, 11, 12, 13, 14, 15]: 10 and 11 are below the lowest bucket's own
        # min (12), so they get weight 0 rather than inheriting the lowest bucket's weight.
        assert list(data["sample_weight_train"]) == pytest.approx([0.0, 0.0, 0.3, 0.3, 1.0, 1.0])

    def test_split__sample_weights_never_applied_to_eval_or_test(self):
        prep = self._build({"sample_weights": {"buckets": [{"min": 0, "weight": 0.0}]}})
        data = prep.prep_training_data()

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
        prep.prep_training_data()

        assert prep.target == "target_1"
        assert prep._resolve_feature_columns(prep.training_data.columns) == ["f1"]
        data = prep.prep_training_data()
        assert list(data["X_train"].columns) == ["f1"]


class TestSplitYears:
    def test_from_config__applies_defaults_for_omitted_fields(self):
        assert SplitYears.from_config({}) == SplitYears(eval_data_years=1, test_data_years=1, num_training_seasons=None)

    def test_from_config__reads_every_field(self):
        split_years = SplitYears.from_config({"eval_data_years": 2, "test_data_years": 3, "num_training_seasons": 10})

        assert split_years == SplitYears(eval_data_years=2, test_data_years=3, num_training_seasons=10)


class TestSplitYearsToMasks:
    def test_to_masks__partitions_seasons_chronologically(self):
        target_season = pd.Series([2019, 2020, 2021, 2022, 2023, 2024])

        masks = SplitYears(eval_data_years=1, test_data_years=2).to_masks(target_season)

        assert masks.is_test.tolist() == [False, False, False, False, True, True]
        assert masks.is_eval.tolist() == [False, False, False, True, False, False]
        assert masks.is_train.tolist() == [True, True, True, False, False, False]

    def test_to_masks__num_training_seasons_keeps_only_the_most_recent_training_seasons(self):
        target_season = pd.Series([2019, 2020, 2021, 2022, 2023, 2024])

        masks = SplitYears(eval_data_years=1, test_data_years=2, num_training_seasons=2).to_masks(target_season)

        assert masks.is_train.tolist() == [False, True, True, False, False, False]

    def test_to_masks__missing_target_seasons_are_in_no_split(self):
        target_season = pd.Series([2020, 2021, 2022, np.nan])

        masks = SplitYears().to_masks(target_season)

        assert [masks.is_train[3], masks.is_eval[3], masks.is_test[3]] == [False, False, False]
        assert masks.is_test.tolist() == [False, False, True, False]

    def test_to_masks__masks_are_disjoint(self):
        target_season = pd.Series([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2024])

        masks = SplitYears(eval_data_years=2, test_data_years=1).to_masks(target_season)

        assert (masks.is_train.astype(int) + masks.is_eval.astype(int) + masks.is_test.astype(int)).max() == 1

    def test_to_masks__raises_when_requested_seasons_exceed_available_seasons(self):
        target_season = pd.Series([2020, 2021, 2022, 2022])

        with pytest.raises(ValueError, match="exceeds the 3 distinct"):
            SplitYears(eval_data_years=1, test_data_years=1, num_training_seasons=2).to_masks(target_season)

    def test_to_masks__accepts_requested_seasons_equal_to_available_seasons(self):
        target_season = pd.Series([2020, 2021, 2022])

        masks = SplitYears(eval_data_years=1, test_data_years=1, num_training_seasons=1).to_masks(target_season)

        assert masks.is_train.tolist() == [True, False, False]
