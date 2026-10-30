import pandas as pd
import pytest
from unittest.mock import patch

from src.modeling.data_builder import ModelDataBuilder, SplitYears


class _ConcreteDataBuilder(ModelDataBuilder):
    """Implement the abstract interface to test shared helpers without gold I/O."""

    def _build_data_splits(self):
        return {}

    def build_prediction_data(self):
        return {}


class _SplitsOnlyDataBuilder(ModelDataBuilder):
    def _build_data_splits(self):
        return {}


class _PredictionOnlyDataBuilder(ModelDataBuilder):
    def build_prediction_data(self):
        return {}


class TestModelDataBuilder:
    @pytest.mark.parametrize("recompute", [
        pytest.param(False, id="default-first-build"),
        pytest.param(True, id="forced-first-build"),
    ])
    def test_get_data_splits__builds_lazily_and_caches_empty_results(self, recompute):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game"})
        with patch.object(builder, "_build_data_splits", wraps=builder._build_data_splits) as build:
            build.assert_not_called()
            first = builder.get_data_splits(recompute=recompute)

            assert first == {}
            assert builder.get_data_splits() is first
            build.assert_called_once_with()

    def test_get_data_splits__recompute_replaces_cache(self):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game"})
        with patch.object(builder, "_build_data_splits", wraps=builder._build_data_splits) as build:
            first = builder.get_data_splits()
            refreshed = builder.get_data_splits(recompute=True)

            assert refreshed is not first
            assert builder.get_data_splits() is refreshed
            assert build.call_count == 2

    def test_get_data_splits__failed_first_build_can_be_retried(self):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game"})
        with patch.object(builder, "_build_data_splits", side_effect=ValueError("failed build")):
            with pytest.raises(ValueError, match="failed build"):
                builder.get_data_splits()
        assert builder._data_splits is None

        first = builder.get_data_splits()
        assert builder.get_data_splits() is first

    def test_get_data_splits__failed_recompute_preserves_cache(self):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game"})
        first = builder.get_data_splits()
        with patch.object(builder, "_build_data_splits", side_effect=ValueError("failed build")) as build:
            with pytest.raises(ValueError, match="failed build"):
                builder.get_data_splits(recompute=True)

            assert builder.get_data_splits() is first
            build.assert_called_once_with()

    @pytest.mark.parametrize("builder_cls, missing_methods", [
        pytest.param(ModelDataBuilder, {"_build_data_splits", "build_prediction_data"}, id="missing-both-methods"),
        pytest.param(_SplitsOnlyDataBuilder, {"build_prediction_data"}, id="missing-prediction-method"),
        pytest.param(_PredictionOnlyDataBuilder, {"_build_data_splits"}, id="missing-splits-method"),
    ])
    def test_abstract_methods__requires_both_build_methods(self, builder_cls, missing_methods):
        assert builder_cls.__abstractmethods__ == missing_methods
        with pytest.raises(TypeError, match="abstract"):
            builder_cls("data", {"target": "ppr_points_per_game"})

    def test_init__configures_fields_without_loading_gold(self, tmp_path):
        builder = _ConcreteDataBuilder(str(tmp_path), {"target": "ppr_points_per_game", "positions": ["RB"]})

        assert builder.target == "ppr_points_per_game"
        assert builder.positions == ["RB"]
        assert builder.split_years == SplitYears()
        assert builder.gold_data_dir == str(tmp_path / "gold")

        assert not hasattr(builder, "source_data")
        assert builder._data_splits is None
        assert not (tmp_path / "gold").exists()

    @pytest.mark.parametrize("split_config, expected", [
        pytest.param({}, SplitYears(), id="default-split-years"),
        pytest.param({"eval_data_years": 2}, SplitYears(eval_data_years=2), id="partial-split-config"),
        pytest.param(
            {"eval_data_years": 2, "test_data_years": 3, "num_training_seasons": 10},
            SplitYears(eval_data_years=2, test_data_years=3, num_training_seasons=10),
            id="full-split-config",
        ),
    ])
    def test_init__stores_configured_split_years(self, split_config, expected):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game", "split": split_config})

        assert builder.split_years == expected

    def test_from_config_file__constructs_calling_subclass(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("target: ppr_points_per_game\npositions: [RB]\n")

        builder = _ConcreteDataBuilder.from_config_file(str(tmp_path), str(config_path))

        assert type(builder) is _ConcreteDataBuilder
        assert builder.config == {"target": "ppr_points_per_game", "positions": ["RB"]}
        assert not hasattr(builder, "source_data")

    @pytest.mark.parametrize("positions", [
        pytest.param(None, id="omitted-positions"),
        pytest.param([], id="empty-positions"),
    ])
    def test_init__omitted_or_empty_positions_keep_all_positions(self, positions):
        config = {"target": "ppr_points_per_game"}
        if positions is not None:
            config["positions"] = positions
        assert _ConcreteDataBuilder("data", config).positions is None

    @pytest.mark.parametrize("features, expected", [
        pytest.param({}, ["f1", "f2", "receiving_yards"], id="default-features"),
        pytest.param({"mode": "exclude", "columns": ["f2"]}, ["f1", "receiving_yards"], id="exclude-exact"),
        pytest.param({"mode": "exclude", "columns": ["receiving_*"]}, ["f1", "f2"], id="exclude-prefix"),
        pytest.param({"mode": "exclude", "columns": ["*_yards"]}, ["f1", "f2"], id="exclude-suffix"),
        pytest.param({"mode": "exclude", "columns": ["*ceiving_ya*"]}, ["f1", "f2"], id="exclude-substring"),
        pytest.param(
            {"mode": "exclude", "columns": ["receiving"]}, ["f1", "f2", "receiving_yards"],
            id="no-implicit-wildcard",
        ),
        pytest.param(
            {"mode": "include", "columns": ["f1", "receiving_*"]}, ["f1", "receiving_yards"],
            id="include-exact-and-prefix",
        ),
        pytest.param({"mode": "both", "columns": ["f1"]}, ["f1", "f2", "receiving_yards"], id="unknown-mode"),
    ])
    def test_resolve_feature_columns__applies_rules_without_identity_or_target(self, features, expected):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game", "features": features})
        columns = [*builder.identity_cols, "target", "f1", "f2", "receiving_yards"]

        assert builder._resolve_feature_columns(columns) == expected

    @pytest.mark.parametrize("features, expected", [
        pytest.param({"mode": "include", "columns": ["*"]}, ["f2", "f1"], id="include-all"),
        pytest.param({"mode": "exclude", "columns": ["*"]}, [], id="exclude-all"),
        pytest.param({"mode": "include", "columns": []}, [], id="empty-include"),
        pytest.param({"mode": "exclude", "columns": []}, ["f2", "f1"], id="empty-exclude"),
        pytest.param(
            {"mode": "include", "columns": ["player_id", "season", "target_season", "position"]},
            [], id="include-identity-only",
        ),
        pytest.param({"mode": "include", "columns": ["target"]}, [], id="include-target-only"),
        pytest.param(
            {"mode": "include", "columns": ["player_id", "target", "f1", "f2"]},
            ["f2", "f1"], id="include-features-with-identity-and-target",
        ),
    ])
    def test_resolve_feature_columns__handles_boundaries_without_admitting_identity_or_target(self, features, expected):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game", "features": features})
        columns = ["player_id", "f2", "season", "target", "target_season", "f1", "position"]

        assert builder._resolve_feature_columns(columns) == expected

    @pytest.mark.parametrize("weights_config, expected", [
        pytest.param({}, [1.0, 1.0, 1.0, 1.0], id="default-uniform-weight"),
        pytest.param({"global_weight": 0.5, "buckets": []}, [0.5, 0.5, 0.5, 0.5], id="global-weight"),
        pytest.param(
            {"buckets": [{"min": 0, "weight": 0.1}, {"min": 12, "weight": 1.0}]},
            [0.1, 0.1, 1.0, 1.0], id="inclusive-bucket-boundary",
        ),
        pytest.param(
            {"buckets": [{"min": 12, "weight": 0.5}, {"min": 0, "weight": 0.1}, {"min": 14, "weight": 1.0}]},
            [0.1, 0.1, 0.5, 1.0], id="unsorted-buckets",
        ),
        pytest.param(
            {"buckets": [{"min": 12, "weight": 0.3}, {"min": 14, "weight": 1.0}]},
            [0.0, 0.0, 0.3, 1.0], id="below-minimum-zero-weight",
        ),
    ])
    def test_compute_sample_weights__preserves_index_and_applies_target_buckets(self, weights_config, expected):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game", "sample_weights": weights_config})
        target = pd.Series([10.0, 11.0, 12.0, 14.0], index=[8, 2, 5, 1])

        pd.testing.assert_series_equal(builder._compute_sample_weights(target), pd.Series(expected, index=target.index))

    @pytest.mark.parametrize("positions, expected_values", [
        pytest.param(["RB"], [1.0, 3.0], id="matching-position"),
        pytest.param(["RB", "WR"], [1.0, 2.0, 3.0], id="multiple-matching-positions"),
        pytest.param(["QB"], [], id="no-matching-position"),
    ])
    def test_filter_positions__selects_matching_rows_and_preserves_schema(self, positions, expected_values):
        data = pd.DataFrame({
            "position": ["RB", "WR", "RB"],
            "f1": [1.0, 2.0, 3.0],
        })

        filtered = _ConcreteDataBuilder._filter_positions(data, positions)

        assert filtered["f1"].tolist() == expected_values
        assert filtered.columns.tolist() == data.columns.tolist()
        pd.testing.assert_series_equal(filtered.dtypes, data.dtypes)

    @pytest.mark.parametrize("weights_config", [
        pytest.param({}, id="default-weight"),
        pytest.param({"global_weight": 0.5}, id="global-weight"),
        pytest.param({"buckets": [{"min": 0, "weight": 1.0}]}, id="bucket-weights"),
    ])
    def test_compute_sample_weights__empty_targets_return_index_aligned_float_series(self, weights_config):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game", "sample_weights": weights_config})
        index = pd.Index([], dtype="int64", name="row")
        target = pd.Series([], index=index, dtype=float)

        weights = builder._compute_sample_weights(target)

        pd.testing.assert_series_equal(weights, pd.Series([], index=index, dtype=float))

    def test_shared_helpers__preserve_feature_order_and_row_alignment(self):
        builder = _ConcreteDataBuilder("data", {
            "target": "ppr_points_per_game",
            "features": {"mode": "exclude", "columns": ["*_career_*"]},
            "sample_weights": {"buckets": [{"min": 10, "weight": 3}, {"min": 0, "weight": 1}]},
        })
        data = pd.DataFrame({
            "position": ["RB", "WR", "RB"],
            "target": [12.0, 5.0, -1.0],
            "f2": [3.0, 2.0, 1.0],
            "f1_career_avg": [4.0, 5.0, 6.0],
            "f1": [1.0, 2.0, 3.0],
        }, index=[8, 2, 5])

        filtered = builder._filter_positions(data, ["RB"])

        assert filtered.index.tolist() == [8, 5]
        assert builder._resolve_feature_columns(filtered.columns) == ["f2", "f1"]
        pd.testing.assert_series_equal(
            builder._compute_sample_weights(filtered["target"]),
            pd.Series([3.0, 0.0], index=[8, 5]),
        )
