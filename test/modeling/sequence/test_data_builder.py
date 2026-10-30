import os
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.modeling.data_builder import ModelDataBuilder
from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.processing.gold import load_training_set


def _assert_tensors_equal(actual: torch.Tensor, expected: torch.Tensor):
    """Exact equality of values, dtype and shape (NaN padding counts as equal to NaN padding)."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _assert_sequence_datasets_equal(actual: SequenceDataset, expected: SequenceDataset):
    """Field by field equality of two SequenceDatasets (a dataclass == can't compare tensors)."""
    _assert_tensors_equal(actual.X, expected.X)
    _assert_tensors_equal(actual.lengths, expected.lengths)
    for field in ["y", "loss_mask", "sample_weight"]:
        actual_value, expected_value = getattr(actual, field), getattr(expected, field)
        if expected_value is None:
            assert actual_value is None, f"{field} should be None"
        else:
            _assert_tensors_equal(actual_value, expected_value)
    pd.testing.assert_frame_equal(actual.identity, expected.identity)
    assert actual.feature_columns == expected.feature_columns


class TestSequenceDataset:
    @staticmethod
    def _dataset() -> SequenceDataset:
        """Player A has 3 real steps, player B has 1 real step followed by 2 padding steps."""
        nan = float("nan")
        return SequenceDataset(
            X=torch.tensor([
                [[1., 10.], [2., 20.], [3., 30.]],
                [[4., 40.], [nan, nan], [nan, nan]],
            ]),
            lengths=torch.tensor([3, 1]),
            identity=pd.DataFrame(
                [(0, 0, "A"), (0, 1, "A"), (0, 2, "A"), (1, 0, "B")],
                columns=["seq_idx", "step_idx", "player_id"],
            ),
            feature_columns=["f1", "f2"],
        )

    def test_real_step_mask(self):
        _assert_tensors_equal(
            self._dataset().real_step_mask,
            torch.tensor([
                [True, True, True],
                [True, False, False],
            ]),
        )

    def test_last_step_mask(self):
        dataset = self._dataset()

        _assert_tensors_equal(
            dataset.last_step_mask,
            torch.tensor([
                [False, False, True],
                [True, False, False],
            ]),
        )
        # Indexing X with it picks out each sequence's last real step, in sequence order.
        _assert_tensors_equal(dataset.X[dataset.last_step_mask], torch.tensor([[3., 30.], [4., 40.]]))

    def test_last_step_identity(self):
        pd.testing.assert_frame_equal(
            self._dataset().last_step_identity().reset_index(drop=True),
            pd.DataFrame([(0, 2, "A"), (1, 0, "B")], columns=["seq_idx", "step_idx", "player_id"]),
        )


class TestSequenceModelDataBuilder:
    def test_inheritance__uses_shared_base_and_helpers(self):
        assert SequenceModelDataBuilder.__bases__ == (ModelDataBuilder,)
        assert SequenceModelDataBuilder.from_config_file.__func__ is ModelDataBuilder.from_config_file.__func__
        for method in ["get_data_splits", "_filter_positions", "_matches_any", "_resolve_feature_columns", "_compute_sample_weights"]:
            assert getattr(SequenceModelDataBuilder, method) is getattr(ModelDataBuilder, method)

    def test_from_config_file__constructs_builder_without_loading_gold(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("target: ppr_points_per_game\npositions: [RB]\n")

        builder = SequenceModelDataBuilder.from_config_file(str(tmp_path), str(config_path))

        assert type(builder) is SequenceModelDataBuilder
        assert builder.target == "ppr_points_per_game"
        assert builder.positions == ["RB"]
        assert builder.gold_data_dir == str(tmp_path / "gold")
        assert not hasattr(builder, "source_data")
        assert not (tmp_path / "gold").exists()

    POSITIONS = {"A": "WR", "B": "QB", "D": "RB", "E": "WR", "F": "TE"}
    PLAYER_NUMBER = {"A": 1, "B": 2, "D": 4, "E": 5, "F": 6}

    # (player, season S, target_season T, target). Max T is 2024, so with the default split
    # (1 eval year, 1 test year): test = T 2024, eval = T 2023, train = T <= 2022.
    TRAINING_ROWS = [
        ("A", 2020, 2021, 10.0), ("A", 2021, 2022, 11.0), ("A", 2022, 2023, 12.0), ("A", 2023, 2024, 13.0),
        ("B", 2022, 2023, 20.0), ("B", 2023, 2024, 21.0),
        ("D", 2020, 2021, 5.0),
        ("E", 2021, 2023, 7.0), ("E", 2023, 2024, 8.0),  # E misses 2022
    ]
    # Each player's latest season is the prediction row. F has no history besides it.
    PREDICTION_ROWS = [("A", 2024, 2025, np.nan), ("B", 2024, 2025, np.nan), ("F", 2024, 2025, np.nan)]

    @classmethod
    def _gold_frame(cls, rows) -> pd.DataFrame:
        """f1 is the row's season so a tensor entry can be traced back to its season, f2 identifies the player."""
        return pd.DataFrame({
            "player_id": [r[0] for r in rows],
            "player_display_name": [f"Player {r[0]}" for r in rows],
            "position": [cls.POSITIONS[r[0]] for r in rows],
            "season": [r[1] for r in rows],
            "recent_team": ["KC"] * len(rows),
            "target_season": [r[2] for r in rows],
            "f1": [float(r[1]) for r in rows],
            "f2": [float(cls.PLAYER_NUMBER[r[0]] * 100) for r in rows],
            "f1_career_avg": [float(r[1]) * 10 for r in rows],
            "target": [r[3] for r in rows],
        })

    @staticmethod
    def _write_gold_tables(data_dir, training_df, prediction_df):
        gold_dir = os.path.join(data_dir, "gold")
        os.makedirs(gold_dir, exist_ok=True)
        training_df.to_csv(os.path.join(gold_dir, "target_1__training_set.csv"), index=False)
        prediction_df.to_csv(os.path.join(gold_dir, "target_1__prediction_set.csv"), index=False)

    @classmethod
    def _identity_frame(cls, steps) -> pd.DataFrame:
        """Expected SequenceDataset.identity from (seq_idx, step_idx, player, season, target_season) per real step."""
        return pd.DataFrame({
            "seq_idx": [s[0] for s in steps],
            "step_idx": [s[1] for s in steps],
            "player_id": [s[2] for s in steps],
            "player_display_name": [f"Player {s[2]}" for s in steps],
            "position": [cls.POSITIONS[s[2]] for s in steps],
            "season": [s[3] for s in steps],
            "recent_team": ["KC"] * len(steps),
            "target_season": [s[4] for s in steps],
        })

    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls._write_gold_tables(cls.test_dir, cls._gold_frame(cls.TRAINING_ROWS), cls._gold_frame(cls.PREDICTION_ROWS))

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def _prep(self, config=None) -> SequenceModelDataBuilder:
        base = {"target": "target_1", "features": {"mode": "exclude", "columns": ["*_career_*"]}}
        return SequenceModelDataBuilder(data_dir=self.test_dir, config={**base, **(config or {})})

    def test_build_data_splits__resolves_feature_columns_correctly(self):
        data = self._prep()._build_data_splits()
        for seq_ds in data.values():
            assert seq_ds.feature_columns == ["f1", "f2"]
        assert data["train"].feature_columns == ["f1", "f2"]

        data_no_exclude = self._prep({"features": {}})._build_data_splits()
        for seq_ds in data_no_exclude.values():
            assert seq_ds.feature_columns == ["f1", "f2", "f1_career_avg"]

    @pytest.mark.parametrize("recompute", [False, True])
    def test_get_data_splits__builds_lazily_then_reuses_cache(self, recompute):
        with patch("src.modeling.sequence.data_builder.load_training_set", wraps=load_training_set) as load:
            builder = self._prep()
            load.assert_not_called()
            assert not hasattr(builder, "source_data")

            with patch.object(builder, "_build_sequence_dataset", wraps=builder._build_sequence_dataset) as assemble:
                first = builder.get_data_splits(recompute=recompute)

                assert builder.get_data_splits() is first
                assert load.call_count == 1
                assert assemble.call_count == 3

    def test_get_data_splits__recompute_reloads_source_and_replaces_cache(self, tmp_path):
        source = self._gold_frame(self.TRAINING_ROWS)
        self._write_gold_tables(str(tmp_path), source, self._gold_frame(self.PREDICTION_ROWS))
        builder = SequenceModelDataBuilder(str(tmp_path), {"target": "target_1"})
        first = builder.get_data_splits()

        source["target"] += 100
        source.to_csv(tmp_path / "gold" / "target_1__training_set.csv", index=False)

        assert builder.get_data_splits() is first
        refreshed = builder.get_data_splits(recompute=True)

        assert refreshed is not first
        assert builder.get_data_splits() is refreshed
        pd.testing.assert_frame_equal(builder.source_data, source)
        for split in ["train", "eval", "test"]:
            old, new = first[split], refreshed[split]
            _assert_tensors_equal(new.X, old.X)
            _assert_tensors_equal(new.loss_mask, old.loss_mask)
            _assert_tensors_equal(new.y, torch.where(old.loss_mask, old.y + 100, old.y))

    def test_get_data_splits__cache_is_per_instance(self):
        first = self._prep().get_data_splits()
        second = self._prep().get_data_splits()

        assert first is not second
        for split in ["train", "eval", "test"]:
            _assert_sequence_datasets_equal(first[split], second[split])

    def test_get_data_splits__failed_first_build_can_be_retried(self):
        builder = self._prep()
        with patch.object(builder, "_build_sequence_dataset", side_effect=ValueError("failed build")):
            with pytest.raises(ValueError, match="failed build"):
                builder.get_data_splits()

        assert builder._data_splits is None
        data = builder.get_data_splits()
        assert builder.get_data_splits() is data

    def test_get_data_splits__failed_recompute_preserves_successful_cache(self):
        builder = self._prep()
        first = builder.get_data_splits()
        with patch.object(builder, "_build_sequence_dataset", side_effect=ValueError("failed build")) as assemble:
            with pytest.raises(ValueError, match="failed build"):
                builder.get_data_splits(recompute=True)

            assert builder.get_data_splits() is first
            assert assemble.call_count == 1


    def test_build_data_splits__always_builds_fresh_without_replacing_cache(self):
        builder = self._prep()
        cached = builder.get_data_splits()

        with patch("src.modeling.sequence.data_builder.load_training_set", wraps=load_training_set) as load:
            first = builder._build_data_splits()
            second = builder._build_data_splits()

        assert first is not second
        assert first is not cached
        assert second is not cached
        assert load.call_count == 2
        assert builder.get_data_splits() is cached

    def test_build_data_splits__generates_train_split_set_correctly(self):
        train = self._prep()._build_data_splits()["train"]

        # - A & D's steps with T <= 2022 are in the training set sequences, (2 and 1 step respectively)
        # - B & E have no steps with T <= 2022, they are eliminated entirely.
        nan = float("nan")
        expected = SequenceDataset(
            X=torch.tensor([
                [[2020., 100.], [2021., 100.]],  # A: 2 real steps
                [[2020., 400.], [nan, nan]],     # D: 1 real step + 1 padding
            ]),
            lengths=torch.tensor([2, 1]),
            identity=self._identity_frame([
                (0, 0, "A", 2020, 2021),
                (0, 1, "A", 2021, 2022),
                (1, 0, "D", 2020, 2021),
            ]),
            feature_columns=["f1", "f2"],
            y=torch.tensor([[10., 11.], [5., 0.]]),
            loss_mask=torch.tensor([[True, True], [True, False]]),
            sample_weight=torch.tensor([[1., 1.], [1., 0.]]),  # default weight of 1.0, zero outside loss steps
        )

        _assert_sequence_datasets_equal(train, expected)

    def test_build_data_splits__generates_eval_split_set_correctly(self):
        eval_split = self._prep()._build_data_splits()["eval"]

        nan = float("nan")
        expected = SequenceDataset(
            X=torch.tensor([
                [[2020., 100.], [2021., 100.], [2022., 100.]],  # A: 3 real steps
                [[2022., 200.], [nan, nan], [nan, nan]],        # B: 1 real step + 2 padding
                [[2021., 500.], [nan, nan], [nan, nan]],        # E: 1 real step + 2 padding
            ]),
            lengths=torch.tensor([3, 1, 1]),
            identity=self._identity_frame([
                (0, 0, "A", 2020, 2021),
                (0, 1, "A", 2021, 2022),
                (0, 2, "A", 2022, 2023),
                (1, 0, "B", 2022, 2023),
                (2, 0, "E", 2021, 2023),
            ]),
            feature_columns=["f1", "f2"],
            y=torch.tensor([[0., 0., 12.], [20., 0., 0.], [7., 0., 0.]]),
            loss_mask=torch.tensor([
                [False, False, True],
                [True, False, False],
                [True, False, False]
            ]),
        )

        _assert_sequence_datasets_equal(eval_split, expected)

    def test_build_data_splits__generates_test_split_set_correctly(self):
        test_split = self._prep()._build_data_splits()["test"]

        nan = float("nan")
        expected = SequenceDataset(
            X=torch.tensor([
                [[2020., 100.], [2021., 100.], [2022., 100.], [2023., 100.]],  # A: 4 real steps
                [[2022., 200.], [2023., 200.], [nan, nan], [nan, nan]],        # B: 2 real steps + 2 padding
                [[2021., 500.], [2023., 500.], [nan, nan], [nan, nan]],        # E: 2 real steps (2022 missed) + 2 padding
            ]),
            lengths=torch.tensor([4, 2, 2]),
            identity=self._identity_frame([
                (0, 0, "A", 2020, 2021),
                (0, 1, "A", 2021, 2022),
                (0, 2, "A", 2022, 2023),
                (0, 3, "A", 2023, 2024),
                (1, 0, "B", 2022, 2023),
                (1, 1, "B", 2023, 2024),
                (2, 0, "E", 2021, 2023),
                (2, 1, "E", 2023, 2024),
            ]),
            feature_columns=["f1", "f2"],
            y=torch.tensor([[0., 0., 0., 13.], [0., 21., 0., 0.], [0., 8., 0., 0.]]),
            loss_mask=torch.tensor([
                [False, False, False, True],
                [False, True, False, False],
                [False, True, False, False],
            ]),
        )

        _assert_sequence_datasets_equal(test_split, expected)

    def test_build_data_splits__training_window_limits_loss_steps_but_not_input_history(self):
        train = self._prep({"split": {"num_training_seasons": 1}})._build_data_splits()["train"]

        # Only T == 2022 (A's second step) is in the training window. A's earlier step is kept as unscored input,
        # D's only label (T == 2021) is outside the window so D is left out.
        expected = SequenceDataset(
            X=torch.tensor([
                [[2020., 100.], [2021., 100.]],  # A: 2 real steps
            ]),
            lengths=torch.tensor([2]),
            identity=self._identity_frame([
                (0, 0, "A", 2020, 2021),
                (0, 1, "A", 2021, 2022),
            ]),
            feature_columns=["f1", "f2"],
            y=torch.tensor([[0., 11.]]),
            loss_mask=torch.tensor([[False, True]]),
            sample_weight=torch.tensor([[0., 1.]]),
        )

        _assert_sequence_datasets_equal(train, expected)

    def test_build_data_splits__splits_use_configured_years(self):
        data = self._prep({"split": {"eval_data_years": 2, "test_data_years": 1}})._build_data_splits()

        def scored_steps(dataset: SequenceDataset) -> pd.DataFrame:
            """(player, season, target_season) of the steps in dataset that count toward the loss."""
            is_scored = dataset.loss_mask[dataset.real_step_mask].numpy()
            return dataset.identity[is_scored][["player_id", "season", "target_season"]].reset_index(drop=True)

        columns = ["player_id", "season", "target_season"]
        # train: T <= 2021, eval: T in {2022, 2023}, test: T == 2024
        pd.testing.assert_frame_equal(
            scored_steps(data["train"]),
            pd.DataFrame([("A", 2020, 2021), ("D", 2020, 2021)], columns=columns),
        )
        pd.testing.assert_frame_equal(
            scored_steps(data["eval"]),
            pd.DataFrame([("A", 2021, 2022), ("A", 2022, 2023), ("B", 2022, 2023), ("E", 2021, 2023)], columns=columns),
        )
        pd.testing.assert_frame_equal(
            scored_steps(data["test"]),
            pd.DataFrame([("A", 2023, 2024), ("B", 2023, 2024), ("E", 2023, 2024)], columns=columns),
        )

    def test_build_data_splits__sample_weight_buckets_are_applied_per_step(self):
        config = {"sample_weights": {"buckets": [{"min": 0.0, "weight": 0.5}, {"min": 10.5, "weight": 1.0}]}}
        train = self._prep(config)._build_data_splits()["train"]

        # Weights follow each step's target: A's targets are 10.0 and 11.0, D's is 5.0.
        _assert_tensors_equal(
            train.sample_weight,
            torch.tensor([
                [0.5, 1.0],  # A
                [0.5, 0.0],  # D: 1 real step + 1 padding
            ]),
        )

    def test_build_data_splits__positions_filter_drops_whole_players(self):
        data = self._prep({"positions": ["WR"]})._build_data_splits()

        assert set(data["test"].identity["player_id"]) == {"A", "E"}
        assert set(data["train"].identity["player_id"]) == {"A"}

    def test_build_data_splits__raises_when_split_needs_more_seasons_than_available(self):
        with pytest.raises(ValueError, match="exceeds"):
            self._prep({"split": {"num_training_seasons": 10}})._build_data_splits()

    def test_build_data_splits__raises_on_duplicate_player_season(self):
        test_dir = tempfile.mkdtemp()
        try:
            training_df = self._gold_frame(self.TRAINING_ROWS + [self.TRAINING_ROWS[0]])
            self._write_gold_tables(test_dir, training_df, self._gold_frame(self.PREDICTION_ROWS))
            prep = SequenceModelDataBuilder(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="More than one row"):
                prep._build_data_splits()
        finally:
            shutil.rmtree(test_dir)

    def test_build_data_splits__raises_on_non_numeric_feature(self):
        test_dir = tempfile.mkdtemp()
        try:
            training_df = self._gold_frame(self.TRAINING_ROWS)
            training_df["team_name"] = "KC"
            prediction_df = self._gold_frame(self.PREDICTION_ROWS)
            prediction_df["team_name"] = "KC"
            self._write_gold_tables(test_dir, training_df, prediction_df)
            prep = SequenceModelDataBuilder(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="team_name"):
                prep._build_data_splits()
        finally:
            shutil.rmtree(test_dir)

    def test_build_data_splits__keeps_missing_feature_values_as_nan(self):
        test_dir = tempfile.mkdtemp()
        try:
            training_df = self._gold_frame(self.TRAINING_ROWS)
            training_df.loc[(training_df["player_id"] == "D"), "f2"] = np.nan
            self._write_gold_tables(test_dir, training_df, self._gold_frame(self.PREDICTION_ROWS))
            prep = SequenceModelDataBuilder(data_dir=test_dir, config={"target": "target_1"})

            train = prep._build_data_splits()["train"]
            d_row = train.identity.index[train.identity["player_id"] == "D"][0]
            assert torch.isnan(train.X[train.real_step_mask][d_row, 1])
        finally:
            shutil.rmtree(test_dir)

    def test_build_prediction_data__generates_prediction_set_correctly(self):
        prediction = self._prep().build_prediction_data()

        # - Each listed player's training steps are followed by their prediction row (2024, predicting 2025).
        # - D & E aren't in the prediction set, F has only the prediction row.
        nan = float("nan")
        expected = SequenceDataset(
            X=torch.tensor([
                [[2020., 100.], [2021., 100.], [2022., 100.], [2023., 100.], [2024., 100.]],  # A: 5 real steps
                [[2022., 200.], [2023., 200.], [2024., 200.], [nan, nan], [nan, nan]],        # B: 3 real steps + 2 padding
                [[2024., 600.], [nan, nan], [nan, nan], [nan, nan], [nan, nan]],              # F: 1 real step + 4 padding
            ]),
            lengths=torch.tensor([5, 3, 1]),
            identity=self._identity_frame([
                (0, 0, "A", 2020, 2021),
                (0, 1, "A", 2021, 2022),
                (0, 2, "A", 2022, 2023),
                (0, 3, "A", 2023, 2024),
                (0, 4, "A", 2024, 2025),
                (1, 0, "B", 2022, 2023),
                (1, 1, "B", 2023, 2024),
                (1, 2, "B", 2024, 2025),
                (2, 0, "F", 2024, 2025),
            ]),
            feature_columns=["f1", "f2"],
        )

        _assert_sequence_datasets_equal(prediction, expected)

    def test_build_prediction_data__respects_position_filter(self):
        prediction = self._prep({"positions": ["WR", "TE"]}).build_prediction_data()

        assert set(prediction.identity["player_id"]) == {"A", "F"}

    def test_build_prediction_data__raises_when_prediction_row_is_not_the_latest_season(self):
        test_dir = tempfile.mkdtemp()
        try:
            stale_prediction_rows = [("A", 2019, 2025, np.nan)]
            self._write_gold_tables(test_dir, self._gold_frame(self.TRAINING_ROWS), self._gold_frame(stale_prediction_rows))
            prep = SequenceModelDataBuilder(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="latest season"):
                prep.build_prediction_data()
        finally:
            shutil.rmtree(test_dir)

    def test_build_prediction_data__raises_when_prediction_row_repeats_a_training_season(self):
        test_dir = tempfile.mkdtemp()
        try:
            repeated_prediction_rows = [("A", 2023, 2025, np.nan)]
            self._write_gold_tables(test_dir, self._gold_frame(self.TRAINING_ROWS), self._gold_frame(repeated_prediction_rows))
            prep = SequenceModelDataBuilder(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="More than one row"):
                prep.build_prediction_data()
        finally:
            shutil.rmtree(test_dir)

    def test_build_prediction_data__raises_when_columns_differ(self):
        test_dir = tempfile.mkdtemp()
        try:
            prediction_df = self._gold_frame(self.PREDICTION_ROWS).drop(columns=["f2"])
            self._write_gold_tables(test_dir, self._gold_frame(self.TRAINING_ROWS), prediction_df)
            prep = SequenceModelDataBuilder(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="same columns"):
                prep.build_prediction_data()
        finally:
            shutil.rmtree(test_dir)

    def test_from_config_file__loads_yaml_config(self):
        config_path = os.path.join(self.test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("target: target_1\nfeatures:\n  mode: exclude\n  columns:\n    - '*_career_*'\n")

        prep = SequenceModelDataBuilder.from_config_file(self.test_dir, config_path)

        assert isinstance(prep, SequenceModelDataBuilder)
        assert prep._build_data_splits()["train"].feature_columns == ["f1", "f2"]
