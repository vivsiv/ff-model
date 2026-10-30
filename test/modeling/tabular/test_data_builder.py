import os
import shutil
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from src.modeling.data_builder import ModelDataBuilder, SplitYears
from src.modeling.tabular.data_builder import TabularModelDataBuilder
from src.processing.column_registry import get_identity_columns
from src.processing.gold import load_training_set


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


class TestTabularModelDataBuilder:
    def test_inheritance__uses_shared_base_and_helpers(self):
        assert TabularModelDataBuilder.__bases__ == (ModelDataBuilder,)
        assert TabularModelDataBuilder.from_config_file.__func__ is ModelDataBuilder.from_config_file.__func__
        for method in ["get_data_splits", "_filter_positions", "_matches_any", "_resolve_feature_columns", "_compute_sample_weights"]:
            assert getattr(TabularModelDataBuilder, method) is getattr(ModelDataBuilder, method)

    def test_from_config_file__constructs_builder_without_loading_gold(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("target: ppr_points_per_game\npositions: [RB]\n")

        builder = TabularModelDataBuilder.from_config_file(str(tmp_path), str(config_path))

        assert type(builder) is TabularModelDataBuilder
        assert builder.target == "ppr_points_per_game"
        assert builder.positions == ["RB"]
        assert builder.gold_data_dir == str(tmp_path / "gold")
        assert not hasattr(builder, "source_data")
        assert not (tmp_path / "gold").exists()

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

    def _build(self, config: dict) -> TabularModelDataBuilder:
        prep = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_1", **config})
        prep._build_data_splits()
        return prep

    def test_build_data_splits__loads_gold_data_correctly(self):
        prep = self._build(config={})
        assert len(prep.source_data) == 10

    @pytest.mark.parametrize("recompute", [False, True])
    def test_get_data_splits__builds_lazily_then_reuses_cache(self, recompute):
        with patch("src.modeling.tabular.data_builder.load_training_set", wraps=load_training_set) as load:
            builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1"})
            load.assert_not_called()
            assert not hasattr(builder, "source_data")

            with patch.object(builder, "split", wraps=builder.split) as split:
                first = builder.get_data_splits(recompute=recompute)

                assert builder.get_data_splits() is first
                assert load.call_count == 1
                assert split.call_count == 1

    def test_get_data_splits__recompute_reloads_source_and_replaces_cache(self, tmp_path):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        source = _build_training_data()
        source_path = gold_dir / "target_1__training_set.csv"
        source.to_csv(source_path, index=False)
        builder = TabularModelDataBuilder(str(tmp_path), {"target": "target_1"})
        first = builder.get_data_splits()

        source["target"] += 100
        source.to_csv(source_path, index=False)

        assert builder.get_data_splits() is first
        refreshed = builder.get_data_splits(recompute=True)

        assert refreshed is not first
        assert builder.get_data_splits() is refreshed
        pd.testing.assert_frame_equal(builder.source_data, source)
        for split in ["train", "eval", "test"]:
            pd.testing.assert_series_equal(refreshed[f"y_{split}"], first[f"y_{split}"] + 100)

    def test_get_data_splits__cache_is_per_instance(self):
        first_builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1"})
        second_builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1"})

        first = first_builder.get_data_splits()
        second = second_builder.get_data_splits()

        assert first is not second
        pd.testing.assert_frame_equal(first["X_train"], second["X_train"])

    def test_get_data_splits__failed_first_build_can_be_retried(self):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1"})
        with patch.object(builder, "split", side_effect=ValueError("failed build")):
            with pytest.raises(ValueError, match="failed build"):
                builder.get_data_splits()

        assert builder._data_splits is None
        data = builder.get_data_splits()
        assert builder.get_data_splits() is data

    def test_get_data_splits__failed_recompute_preserves_successful_cache(self):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1"})
        first = builder.get_data_splits()
        with patch.object(builder, "split", side_effect=ValueError("failed build")) as split:
            with pytest.raises(ValueError, match="failed build"):
                builder.get_data_splits(recompute=True)

            assert builder.get_data_splits() is first
            assert split.call_count == 1

    def test_build_data_splits__always_builds_fresh_without_replacing_cache(self):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1"})
        cached = builder.get_data_splits()

        with patch("src.modeling.tabular.data_builder.load_training_set", wraps=load_training_set) as load:
            first = builder._build_data_splits()
            second = builder._build_data_splits()

        assert first is not second
        assert first is not cached
        assert second is not cached
        assert load.call_count == 2
        assert builder.get_data_splits() is cached

    def test_init__sets_base_fields_correctly(self):
        prep = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_1"})
        assert prep.target == "target_1"
        assert prep.positions is None

    def test_build_data_splits__returns_the_train_eval_test_split(self):
        # _build_data_splits() calls split() itself and returns the result, so consumers
        # (TabularModelTrainer, TabularModelInference) don't need to call .split() themselves.
        prep = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_1"})
        data = prep._build_data_splits()
        assert set(data.keys()) == {
            "X_train", "X_eval", "X_test",
            "y_train", "y_eval", "y_test",
            "identity_train", "identity_eval", "identity_test",
            "sample_weight_train",
        }

    def test_build_data_splits__filters_to_a_single_configured_position(self):
        prep = self._build(config={"positions": ["RB"]})
        data = prep._build_data_splits()
        identities = pd.concat([data[f"identity_{split}"] for split in ["train", "eval", "test"]])

        assert len(identities) == 2
        assert set(identities["position"]) == {"RB"}
        pd.testing.assert_frame_equal(prep.source_data, self.training_data)

    def test_build_data_splits__filters_to_multiple_configured_positions(self):
        prep = self._build(config={"positions": ["RB", "WR"]})
        data = prep._build_data_splits()
        identities = pd.concat([data[f"identity_{split}"] for split in ["train", "eval", "test"]])

        assert len(identities) == 4
        assert set(identities["position"]) == {"RB", "WR"}
        pd.testing.assert_frame_equal(prep.source_data, self.training_data)

    def test_build_data_splits__empty_positions_list_is_treated_like_omitted(self):
        prep = self._build({"positions": []})
        assert len(prep.source_data) == 10
        assert prep.positions is None

    def test_build_prediction_data__splits_into_identity_and_resolved_feature_columns(self):
        prep = self._build({"features": {"mode": "exclude", "columns": ["f2"]}})
        prediction_set = prep.build_prediction_data()

        assert list(prediction_set["features"].columns) == ["f1", "receiving_yards"]
        assert list(prediction_set["features"]["f1"]) == [11, 12, 13, 14]
        assert set(prediction_set["identity"]["position"]) == {"QB", "RB", "WR", "TE"}

    def test_build_prediction_data__respects_position_filtering(self):
        prep = self._build({"positions": ["RB"]})
        prediction_set = prep.build_prediction_data()

        assert set(prediction_set["identity"]["position"]) == {"RB"}
        assert list(prediction_set["features"]["f1"]) == [12]

    def test_build_prediction_data__works_without_ever_calling_get_data_splits(self):
        # The whole point of separating get_data_splits()/build_prediction_data(): a
        # TabularModelDataBuilder built purely to make live predictions from an already-trained
        # model should never need to load (or even have) a training set on disk.
        target = "target_predict_only"
        _build_prediction_data().to_csv(os.path.join(self.gold_dir, f"{target}__prediction_set.csv"), index=False)
        # deliberately not writing f"{target}__training_set.csv" at all

        prep = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": target})
        prediction_set = prep.build_prediction_data()

        assert not hasattr(prep, "source_data")
        assert len(prediction_set["identity"]) == 4
        assert list(prediction_set["features"].columns) == ["f1", "f2", "receiving_yards"]

    def test_split__is_independently_callable_with_explicit_args(self):
        # split() is a pure staticmethod with no config/instance state -- directly testable
        # without a TabularModelDataBuilder, _build_data_splits(), or a CSV fixture at all.
        target_season = pd.Series([2020, 2020, 2021, 2021, 2022, 2022])
        identity_df = pd.DataFrame({"target_season": target_season})
        features_df = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
        target_df = pd.Series([10, 11, 12, 13, 14, 15])
        sample_weights = pd.Series([1.0] * 6)

        data = TabularModelDataBuilder.split(
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

    def test_build_data_splits__holds_out_most_recent_season_for_test_and_the_one_before_for_eval(self):
        prep = self._build({"split": {"eval_data_years": 1, "test_data_years": 1}})
        data = prep._build_data_splits()

        assert set(data["identity_test"]["target_season"]) == {2024}
        assert set(data["identity_eval"]["target_season"]) == {2023}
        assert set(data["identity_train"]["target_season"]) == {2020, 2021, 2022}

        assert list(data["y_test"]) == [18, 19]
        assert list(data["y_eval"]) == [16, 17]
        assert list(data["y_train"]) == [10, 11, 12, 13, 14, 15]

    def test_build_data_splits__supports_more_than_one_year_for_test_and_eval(self):
        prep = self._build({"split": {"eval_data_years": 2, "test_data_years": 2}})
        data = prep._build_data_splits()

        assert set(data["identity_test"]["target_season"]) == {2023, 2024}
        assert set(data["identity_eval"]["target_season"]) == {2021, 2022}
        assert set(data["identity_train"]["target_season"]) == {2020}

        assert list(data["y_test"]) == [16, 17, 18, 19]
        assert list(data["y_eval"]) == [12, 13, 14, 15]
        assert list(data["y_train"]) == [10, 11]

    def test_build_data_splits__num_training_seasons_limits_training_to_most_recent_n_seasons(self):
        prep = self._build({"split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 1}})
        data = prep._build_data_splits()

        assert set(data["identity_train"]["target_season"]) == {2022}

    def test_build_data_splits__raises_if_requested_seasons_exceed_available_seasons(self):
        # The _build helper calls _build_data_splits(), so a bad split config raises here.
        with pytest.raises(ValueError):
            self._build({"split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 4}})

    def test_build_data_splits__defaults_to_one_year_each_with_no_season_limit(self):
        prep = self._build({})
        data = prep._build_data_splits()

        assert set(data["identity_test"]["target_season"]) == {2024}
        assert set(data["identity_eval"]["target_season"]) == {2023}
        assert set(data["identity_train"]["target_season"]) == {2020, 2021, 2022}

    def test_build_data_splits__sample_weight_train_is_uniform_when_sample_weights_omitted(self):
        prep = self._build({})
        data = prep._build_data_splits()

        assert list(data["sample_weight_train"]) == [1.0] * 6

    def test_build_data_splits__sample_weight_train_uses_global_weight_when_no_buckets_are_given(self):
        prep = self._build({"sample_weights": {"global_weight": 0.5, "buckets": []}})
        data = prep._build_data_splits()

        assert list(data["sample_weight_train"]) == [0.5] * 6

    def test_build_data_splits__sample_weight_train_applies_matching_bucket(self):
        prep = self._build({
            "split": {"eval_data_years": 1, "test_data_years": 1},
            "sample_weights": {"buckets": [{"min": 0, "weight": 0.1}, {"min": 12, "weight": 1.0}]},
        })
        data = prep._build_data_splits()

        # train targets [10, 11, 12, 13, 14, 15]: 10/11 < 12 -> min=0 bucket (0.1);
        # 12-15 >= 12 -> min=12 bucket (1.0)
        assert list(data["sample_weight_train"]) == [0.1, 0.1, 1.0, 1.0, 1.0, 1.0]

    def test_build_data_splits__sample_weight_train_uses_the_largest_qualifying_min_bucket(self):
        prep = self._build({
            "sample_weights": {
                # listed out of order on purpose -- order in the config shouldn't matter.
                "buckets": [{"min": 12, "weight": 0.5}, {"min": 0, "weight": 0.1}, {"min": 14, "weight": 1.0}],
            },
        })
        data = prep._build_data_splits()

        # train targets [10, 11, 12, 13, 14, 15]:
        # 10, 11 only qualify for min=0 -> 0.1
        # 12, 13 qualify for min=0 and min=12 -> largest (min=12) wins -> 0.5
        # 14, 15 qualify for all three -> largest (min=14) wins -> 1.0
        assert list(data["sample_weight_train"]) == pytest.approx([0.1, 0.1, 0.5, 0.5, 1.0, 1.0])

    def test_build_data_splits__sample_weight_train_rows_below_the_lowest_bucket_get_zero_weight(self):
        prep = self._build({
            "sample_weights": {"buckets": [{"min": 12, "weight": 0.3}, {"min": 14, "weight": 1.0}]},
        })
        data = prep._build_data_splits()

        # train targets [10, 11, 12, 13, 14, 15]: 10 and 11 are below the lowest bucket's own
        # min (12), so they get weight 0 rather than inheriting the lowest bucket's weight.
        assert list(data["sample_weight_train"]) == pytest.approx([0.0, 0.0, 0.3, 0.3, 1.0, 1.0])

    def test_build_data_splits__sample_weights_never_applied_to_eval_or_test(self):
        prep = self._build({"sample_weights": {"buckets": [{"min": 0, "weight": 0.0}]}})
        data = prep._build_data_splits()

        assert "sample_weight_eval" not in data
        assert "sample_weight_test" not in data

    def test_from_config_file__loads_yaml_and_builds_a_working_data_builder(self):
        config_path = os.path.join(self.test_dir, "target_1.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump({
                "target": "target_1",
                "features": {"mode": "include", "columns": ["f1"]},
                "split": {"eval_data_years": 1, "test_data_years": 1},
            }, f)

        prep = TabularModelDataBuilder.from_config_file(data_dir=self.test_dir, config_path=config_path)
        prep._build_data_splits()

        assert prep.target == "target_1"
        assert prep._resolve_feature_columns(prep.source_data.columns) == ["f1"]
        data = prep._build_data_splits()
        assert list(data["X_train"].columns) == ["f1"]
