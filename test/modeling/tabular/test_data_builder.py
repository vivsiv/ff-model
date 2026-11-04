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
from src.processing.gold import load_gold_set


def _build_training_data() -> pd.DataFrame:
    n = 10
    identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
    identity_data["position"] = ["QB", "TE", "RB", "WR", "QB", "TE", "RB", "QB", "WR", "TE"]
    identity_data["target_season"] = [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024]

    data = pd.DataFrame({
        **identity_data,
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "f2": [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
        "receiving_yards": [12, 0, 8, 12, 0, 8, 12, 0, 8, 12],
        "target": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    })
    data["target"] = data["target"].astype(float)
    return data


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
        "target": [float("nan")] * n,
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
        cls.prediction_data = _build_prediction_data()
        cls.gold_data = pd.concat([cls.training_data, cls.prediction_data], ignore_index=True)
        cls.gold_data.to_csv(os.path.join(cls.gold_dir, "target_1__2025.csv"), index=False)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_init__sets_base_fields_correctly(self):
        builder = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_1", "split": {"prediction_season": 2025}})
        assert builder.target == "target_1"
        assert builder.positions is None

    def test_from_config_file__loads_yaml_and_builds_a_working_data_builder(self):
        config_path = os.path.join(self.test_dir, "target_1.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump({
                "target": "target_1",
                "features": {"mode": "include", "columns": ["f1"]},
                "split": {"prediction_season": 2025, "eval_data_years": 1, "test_data_years": 1},
            }, f)

        builder = TabularModelDataBuilder.from_config_file(data_dir=self.test_dir, config_path=config_path)
        assert builder.target == "target_1"
        data = builder._build_data_splits()
        assert builder._resolve_feature_columns(builder.source_data.columns) == ["f1"]
        assert list(data["X_train"].columns) == ["f1"]

    def test_get_data_splits__builds_lazily_then_reuses_cache(self):
        with patch("src.modeling.tabular.data_builder.load_gold_set", wraps=load_gold_set) as load:
            builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1", "split": {"prediction_season": 2025}})
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
        source_path = gold_dir / "target_1__2025.csv"
        source.to_csv(source_path, index=False)
        builder = TabularModelDataBuilder(str(tmp_path), {"target": "target_1", "split": {"prediction_season": 2025}})

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
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1", "split": {"prediction_season": 2025}})
        builder._build_data_splits()
        assert len(builder.source_data) == 14

    def test_build_data_splits__returns_the_train_eval_test_split(self):
        # _build_data_splits() calls split() itself and returns the result, so consumers
        # (TabularModelTrainer, TabularModelInference) don't need to call .split() themselves.
        builder = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_1", "split": {"prediction_season": 2025}})
        data = builder._build_data_splits()
        assert set(data.keys()) == {
            "X_train", "X_eval", "X_test",
            "y_train", "y_eval", "y_test",
            "identity_train", "identity_eval", "identity_test",
            "sample_weight_train",
            "X_pred", "identity_pred",
        }

    def test_build_data_splits__applies_position_filter_without_changing_source(self):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1", "split": {"prediction_season": 2025}, "positions": ["RB"]})
        splits = builder._build_data_splits()

        for split in ["train", "eval", "test"]:
            assert set(splits[f"identity_{split}"]["position"]) <= {"RB"}

        assert sum(len(splits[f"identity_{split}"]) for split in ["train", "eval", "test"]) == 2
        pd.testing.assert_frame_equal(builder.source_data, self.gold_data)

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

    def test_split__preserves_complete_row_alignment_with_shuffled_non_default_indexes(self):
        source = pd.DataFrame({
            "row_marker": [10, 20, 30, 40, 50, 60, 70, 80, 90],
            "feature": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
            "player_id": list("ABCDEFGHI"),
            "target_season": [2021, 2022, 2023, 2023, 2024, 2024, 2025, 2025, 2026],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float("nan"), float("nan"), 9.0],
            "sample_weight": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }, index=pd.Index([31, 7, 90, 12, 65, 3, 44, 18, 101], name="row_id"))
        shuffled = source.iloc[[7, 3, 8, 1, 5, 2, 6, 0, 4]]

        actual = TabularModelDataBuilder.split(
            split_years=SplitYears(prediction_season=2025, num_training_seasons=2),
            target_season=shuffled["target_season"],
            features_df=shuffled[["feature", "row_marker"]],
            target_df=shuffled["target"],
            identity_df=shuffled[["player_id", "target_season"]],
            sample_weights=shuffled["sample_weight"],
        )

        train_index = pd.Index([7, 31], name="row_id")
        eval_index = pd.Index([12, 90], name="row_id")
        test_index = pd.Index([3, 65], name="row_id")
        pred_index = pd.Index([18, 44], name="row_id")
        expected_frames = {
            "X_train": pd.DataFrame({"feature": [200.0, 100.0], "row_marker": [20, 10]}, index=train_index),
            "X_eval": pd.DataFrame({"feature": [400.0, 300.0], "row_marker": [40, 30]}, index=eval_index),
            "X_test": pd.DataFrame({"feature": [600.0, 500.0], "row_marker": [60, 50]}, index=test_index),
            "X_pred": pd.DataFrame({"feature": [800.0, 700.0], "row_marker": [80, 70]}, index=pred_index),
            "identity_train": pd.DataFrame({"player_id": ["B", "A"], "target_season": [2022, 2021]}, index=train_index),
            "identity_eval": pd.DataFrame({"player_id": ["D", "C"], "target_season": [2023, 2023]}, index=eval_index),
            "identity_test": pd.DataFrame({"player_id": ["F", "E"], "target_season": [2024, 2024]}, index=test_index),
            "identity_pred": pd.DataFrame({"player_id": ["H", "G"], "target_season": [2025, 2025]}, index=pred_index),
        }
        expected_series = {
            "y_train": pd.Series([2.0, 1.0], index=train_index, name="target"),
            "y_eval": pd.Series([4.0, 3.0], index=eval_index, name="target"),
            "y_test": pd.Series([6.0, 5.0], index=test_index, name="target"),
            "sample_weight_train": pd.Series([0.2, 0.1], index=train_index, name="sample_weight"),
        }
        assert set(actual) == set(expected_frames) | set(expected_series)
        for key, expected in expected_frames.items():
            pd.testing.assert_frame_equal(actual[key], expected)
        for key, expected in expected_series.items():
            pd.testing.assert_series_equal(actual[key], expected)

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
        builder = TabularModelDataBuilder(self.test_dir, {
            "target": "target_1", **config,
            "split": {"prediction_season": 2025, **config.get("split", {})},
        })
        data = builder._build_data_splits()

        for split, seasons, targets in zip(["train", "eval", "test"], expected_seasons, expected_targets):
            assert set(data[f"identity_{split}"]["target_season"]) == seasons
            assert data[f"y_{split}"].tolist() == targets

    def test_build_data_splits__raises_if_requested_seasons_exceed_available_seasons(self):
        builder = TabularModelDataBuilder(self.test_dir, {
            "target": "target_1",
            "split": {"prediction_season": 2025, "eval_data_years": 1, "test_data_years": 1, "num_training_seasons": 4},
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
        builder = TabularModelDataBuilder(self.test_dir, {
            "target": "target_1", **config,
            "split": {"prediction_season": 2025, **config.get("split", {})},
        })
        data = builder._build_data_splits()

        assert data["sample_weight_train"].tolist() == pytest.approx(expected_weights)
        assert "sample_weight_eval" not in data
        assert "sample_weight_test" not in data

    def test_build_data_splits__prediction_has_identity_and_resolved_features(self):
        builder = TabularModelDataBuilder(self.test_dir, {
            "target": "target_1", "split": {"prediction_season": 2025},
            "features": {"mode": "exclude", "columns": ["f2"]},
        })
        prediction_set = builder.get_data_splits()

        assert list(prediction_set["X_pred"].columns) == ["f1", "receiving_yards"]
        assert list(prediction_set["X_pred"]["f1"]) == [11, 12, 13, 14]
        assert set(prediction_set["identity_pred"]["position"]) == {"QB", "RB", "WR", "TE"}
        assert "y_pred" not in prediction_set
        assert "prediction" not in prediction_set
        pd.testing.assert_index_equal(prediction_set["X_pred"].index, prediction_set["identity_pred"].index)

    def test_build_data_splits__prediction_respects_position_filtering(self):
        builder = TabularModelDataBuilder(self.test_dir, {"target": "target_1", "split": {"prediction_season": 2025}, "positions": ["RB"]})
        prediction_set = builder.get_data_splits()

        assert set(prediction_set["identity_pred"]["position"]) == {"RB"}
        assert list(prediction_set["X_pred"]["f1"]) == [12]

    def test_build_data_splits__prediction_only_table_has_empty_scored_splits(self):
        # A combined gold table may contain only prediction rows.
        prediction_df = _build_prediction_data()
        prediction_df.to_csv(os.path.join(self.gold_dir, "target_predict_only__2025.csv"), index=False)

        builder = TabularModelDataBuilder(data_dir=self.test_dir, config={"target": "target_predict_only", "split": {"prediction_season": 2025}})
        splits = builder.get_data_splits()

        expected_features = prediction_df[["f1", "f2", "receiving_yards"]]
        expected_identity = prediction_df[get_identity_columns("nflverse", "player_stats") + ["target_season"]]
        empty_index = pd.Index([], dtype="int64")
        empty_features = expected_features.iloc[:0].set_axis(empty_index)
        empty_identity = expected_identity.iloc[:0].set_axis(empty_index)
        empty_target = pd.Series([], index=empty_index, dtype=float, name="target")
        empty_weights = pd.Series([], index=empty_index, dtype=float)
        expected_frames = {
            "X_train": empty_features, "X_eval": empty_features, "X_test": empty_features,
            "identity_train": empty_identity, "identity_eval": empty_identity, "identity_test": empty_identity,
            "X_pred": expected_features, "identity_pred": expected_identity,
        }
        expected_series = {
            "y_train": empty_target, "y_eval": empty_target, "y_test": empty_target,
            "sample_weight_train": empty_weights,
        }
        assert set(splits) == set(expected_frames) | set(expected_series)
        for key, expected in expected_frames.items():
            pd.testing.assert_frame_equal(splits[key], expected)
        for key, expected in expected_series.items():
            pd.testing.assert_series_equal(splits[key], expected)

    def test_prediction_split__backtest_excludes_targets_and_future_rows(self):
        source = self.gold_data.copy(deep=True)
        original = source.copy(deep=True)
        builder = TabularModelDataBuilder(self.test_dir, {
            "target": "target_1", "split": {"prediction_season": 2023},
        })
        with patch("src.modeling.tabular.data_builder.load_gold_set", return_value=source):
            splits = builder.get_data_splits()

        assert splits["identity_pred"]["target_season"].tolist() == [2023, 2023]
        assert splits["X_pred"]["f1"].tolist() == [7, 8]
        assert "y_pred" not in splits
        assert "target" not in splits["X_pred"]
        assert set(splits["identity_test"]["target_season"]) == {2022}
        assert set(splits["identity_eval"]["target_season"]) == {2021}
        pd.testing.assert_frame_equal(source, original)

    def test_prediction_split__is_cached_with_other_splits_after_one_gold_load(self):
        builder = TabularModelDataBuilder(self.test_dir, {
            "target": "target_1", "split": {"prediction_season": 2025},
        })
        with patch("src.modeling.tabular.data_builder.load_gold_set", wraps=load_gold_set) as load:
            first = builder.get_data_splits()
            assert builder.get_data_splits()["X_pred"] is first["X_pred"]
            assert builder.get_data_splits()["identity_pred"] is first["identity_pred"]
            load.assert_called_once()
