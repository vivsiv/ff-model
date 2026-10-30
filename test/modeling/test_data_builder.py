import pandas as pd
import pytest

from src.modeling.data_builder import ModelDataBuilder


class _ConcreteDataBuilder(ModelDataBuilder):
    """Implement the abstract interface to test shared helpers without gold I/O."""

    def build_data_splits(self):
        return {}

    def build_prediction_data(self):
        return {}


class _SplitsOnlyDataBuilder(ModelDataBuilder):
    def build_data_splits(self):
        return {}


class _PredictionOnlyDataBuilder(ModelDataBuilder):
    def build_prediction_data(self):
        return {}


class TestModelDataBuilder:
    @pytest.mark.parametrize("builder_cls, missing_methods", [
        (ModelDataBuilder, {"build_data_splits", "build_prediction_data"}),
        (_SplitsOnlyDataBuilder, {"build_prediction_data"}),
        (_PredictionOnlyDataBuilder, {"build_data_splits"}),
    ])
    def test_abstract_methods__requires_both_build_methods(self, builder_cls, missing_methods):
        assert builder_cls.__abstractmethods__ == missing_methods
        with pytest.raises(TypeError, match="abstract"):
            builder_cls("data", {"target": "ppr_points_per_game"})

    def test_init__configures_fields_without_loading_gold(self, tmp_path):
        builder = _ConcreteDataBuilder(str(tmp_path), {"target": "ppr_points_per_game", "positions": ["RB"]})

        assert builder.target == "ppr_points_per_game"
        assert builder.positions == ["RB"]
        assert builder.data_dir == str(tmp_path)
        assert builder.gold_data_dir == str(tmp_path / "gold")
        assert not hasattr(builder, "source_data")
        assert not (tmp_path / "gold").exists()

    def test_from_config_file__constructs_calling_subclass(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("target: ppr_points_per_game\npositions: [RB]\n")

        builder = _ConcreteDataBuilder.from_config_file(str(tmp_path), str(config_path))

        assert type(builder) is _ConcreteDataBuilder
        assert builder.config == {"target": "ppr_points_per_game", "positions": ["RB"]}
        assert not hasattr(builder, "source_data")

    @pytest.mark.parametrize("positions", [None, []])
    def test_init__omitted_or_empty_positions_keep_all_positions(self, positions):
        config = {"target": "ppr_points_per_game"}
        if positions is not None:
            config["positions"] = positions
        assert _ConcreteDataBuilder("data", config).positions is None

    @pytest.mark.parametrize("features, expected", [
        ({}, ["f1", "f2", "receiving_yards"]),
        ({"mode": "exclude", "columns": ["f2"]}, ["f1", "receiving_yards"]),
        ({"mode": "exclude", "columns": ["receiving_*"]}, ["f1", "f2"]),
        ({"mode": "exclude", "columns": ["*_yards"]}, ["f1", "f2"]),
        ({"mode": "exclude", "columns": ["*ceiving_ya*"]}, ["f1", "f2"]),
        ({"mode": "exclude", "columns": ["receiving"]}, ["f1", "f2", "receiving_yards"]),
        ({"mode": "include", "columns": ["f1", "receiving_*"]}, ["f1", "receiving_yards"]),
        ({"mode": "both", "columns": ["f1"]}, ["f1", "f2", "receiving_yards"]),
    ])
    def test_resolve_feature_columns__applies_rules_without_identity_or_target(self, features, expected):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game", "features": features})
        columns = [*builder.identity_cols, "target", "f1", "f2", "receiving_yards"]

        assert builder._resolve_feature_columns(columns) == expected

    @pytest.mark.parametrize("weights_config, expected", [
        ({}, [1.0, 1.0, 1.0, 1.0]),
        ({"global_weight": 0.5, "buckets": []}, [0.5, 0.5, 0.5, 0.5]),
        ({"buckets": [{"min": 0, "weight": 0.1}, {"min": 12, "weight": 1.0}]}, [0.1, 0.1, 1.0, 1.0]),
        ({"buckets": [{"min": 12, "weight": 0.5}, {"min": 0, "weight": 0.1}, {"min": 14, "weight": 1.0}]}, [0.1, 0.1, 0.5, 1.0]),
        ({"buckets": [{"min": 12, "weight": 0.3}, {"min": 14, "weight": 1.0}]}, [0.0, 0.0, 0.3, 1.0]),
    ])
    def test_compute_sample_weights__preserves_index_and_applies_target_buckets(self, weights_config, expected):
        builder = _ConcreteDataBuilder("data", {"target": "ppr_points_per_game", "sample_weights": weights_config})
        target = pd.Series([10.0, 11.0, 12.0, 14.0], index=[8, 2, 5, 1])

        pd.testing.assert_series_equal(builder._compute_sample_weights(target), pd.Series(expected, index=target.index))

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
