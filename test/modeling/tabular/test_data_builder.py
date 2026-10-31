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

    def test_init__sets_base_fields_correctly(self):
        builder = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_1"})
        assert builder.target == "target_1"
        assert builder.positions is None

    def test_from_config_file__loads_yaml_and_builds_a_working_data_builder(self):
        config_path = os.path.join(self.test_dir, "target_1.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump({
                "target": "target_1",
                "features": {"mode": "include", "columns": ["f1"]},
                "split": {"eval_data_years": 1, "test_data_years": 1},
            }, f)

        builder = TabularModelDataBuilder.from_config_file(data_dir=self.test_dir, config_path=config_path)
        assert builder.target == "target_1"
        data = builder._build_data_splits()
        assert builder._resolve_feature_columns(builder.source_data.columns) == ["f1"]
        assert list(data["X_train"].columns) == ["f1"]

    def test_get_data_splits__builds_lazily_then_reuses_cache(self):
        with patch("src.modeling.tabular.data_builder.load_training_set", wraps=load_training_set) as load:
            builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1"})
            load.assert_not_called()
            assert not hasattr(builder, "source_data")

            with patch.object(builder, "split", wraps=builder.split) as split:
                first = builder.get_data_splits()

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

    def test_build_data_splits__loads_gold_data_correctly(self):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1"})
        builder._build_data_splits()
        assert len(builder.source_data) == 10

    def test_build_data_splits__returns_the_train_eval_test_split(self):
        # _build_data_splits() calls split() itself and returns the result, so consumers
        # (TabularModelTrainer, TabularModelInference) don't need to call .split() themselves.
        builder = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_1"})
        data = builder._build_data_splits()
        assert set(data.keys()) == {
            "X_train", "X_eval", "X_test",
            "y_train", "y_eval", "y_test",
            "identity_train", "identity_eval", "identity_test",
            "sample_weight_train",
        }

    def test_build_data_splits__applies_position_filter_without_changing_source(self):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1", "positions": ["RB"]})
        splits = builder._build_data_splits()

        for split in ["train", "eval", "test"]:
            assert set(splits[f"identity_{split}"]["position"]) <= {"RB"}

        assert sum(len(splits[f"identity_{split}"]) for split in ["train", "eval", "test"]) == 2
        pd.testing.assert_frame_equal(builder.source_data, self.training_data)

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

    @pytest.mark.parametrize("config, expected_seasons, expected_targets", [
        pytest.param(
            {"split": {"eval_data_years": 1, "test_data_years": 1}},
            [{2020, 2021, 2022}, {2023}, {2024}],
            [[10, 11, 12, 13, 14, 15], [16, 17], [18, 19]], id="one-year-holdouts",
        ),
        pytest.param(
            {"split": {"eval_data_years": 2, "test_data_years": 2}},
            [{2020}, {2021, 2022}, {2023, 2024}],
            [[10, 11], [12, 13, 14, 15], [16, 17, 18, 19]], id="multi-year-holdouts",
        ),
        pytest.param(
            {"split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 1}},
            [{2022}, {2023}, {2024}],
            [[14, 15], [16, 17], [18, 19]], id="limited-training-window",
        ),
        pytest.param(
            {}, [{2020, 2021, 2022}, {2023}, {2024}],
            [[10, 11, 12, 13, 14, 15], [16, 17], [18, 19]], id="default-split",
        ),
    ])
    def test_build_data_splits__assigns_seasons_and_targets_chronologically(self, config, expected_seasons, expected_targets):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1", **config})
        data = builder._build_data_splits()

        for split, seasons, targets in zip(["train", "eval", "test"], expected_seasons, expected_targets):
            assert set(data[f"identity_{split}"]["target_season"]) == seasons
            assert data[f"y_{split}"].tolist() == targets

    def test_build_data_splits__raises_if_requested_seasons_exceed_available_seasons(self):
        builder = TabularModelDataBuilder(self.test_dir, {
            "target": "target_1",
            "split": {"eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 4},
        })
        with pytest.raises(ValueError):
            builder._build_data_splits()

    @pytest.mark.parametrize("config, expected_weights", [
        pytest.param({}, [1.0] * 6, id="default-uniform-weight"),
        pytest.param(
            {"sample_weights": {"global_weight": 0.5, "buckets": []}},
            [0.5] * 6, id="global-weight",
        ),
        pytest.param(
            {"sample_weights": {"buckets": [{"min": 0, "weight": 0.1}, {"min": 12, "weight": 1.0}]}},
            [0.1, 0.1, 1.0, 1.0, 1.0, 1.0], id="matching-buckets",
        ),
        pytest.param(
            {"sample_weights": {"buckets": [{"min": 12, "weight": 0.5}, {"min": 0, "weight": 0.1}, {"min": 14, "weight": 1.0}]}},
            [0.1, 0.1, 0.5, 0.5, 1.0, 1.0], id="unsorted-buckets",
        ),
        pytest.param(
            {"sample_weights": {"buckets": [{"min": 12, "weight": 0.3}, {"min": 14, "weight": 1.0}]}},
            [0.0, 0.0, 0.3, 0.3, 1.0, 1.0], id="below-minimum-zero-weight",
        ),
        pytest.param(
            {"sample_weights": {"buckets": [{"min": 0, "weight": 0.0}]}},
            [0.0] * 6, id="zero-weight",
        ),
    ])
    def test_build_data_splits__applies_sample_weights_only_to_training(self, config, expected_weights):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1", **config})
        data = builder._build_data_splits()

        assert data["sample_weight_train"].tolist() == pytest.approx(expected_weights)
        assert "sample_weight_eval" not in data
        assert "sample_weight_test" not in data

    def test_build_prediction_data__splits_into_identity_and_resolved_feature_columns(self):
        builder = TabularModelDataBuilder(self.test_dir, {
            "target": "target_1",
            "features": {"mode": "exclude", "columns": ["f2"]},
        })
        prediction_set = builder.build_prediction_data()

        assert list(prediction_set["features"].columns) == ["f1", "receiving_yards"]
        assert list(prediction_set["features"]["f1"]) == [11, 12, 13, 14]
        assert set(prediction_set["identity"]["position"]) == {"QB", "RB", "WR", "TE"}

    def test_build_prediction_data__respects_position_filtering(self):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1", "positions": ["RB"]})
        prediction_set = builder.build_prediction_data()

        assert set(prediction_set["identity"]["position"]) == {"RB"}
        assert list(prediction_set["features"]["f1"]) == [12]

    def test_build_prediction_data__works_without_training_data(self):
        # Prediction-only builders should not need a training table on disk.
        _build_prediction_data().to_csv(os.path.join(self.gold_dir, "target_predict_only__prediction_set.csv"), index=False)

        builder = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_predict_only"})
        prediction_set = builder.build_prediction_data()

        assert not hasattr(builder, "source_data")
        assert len(prediction_set["identity"]) == 4
        assert list(prediction_set["features"].columns) == ["f1", "f2", "receiving_yards"]
