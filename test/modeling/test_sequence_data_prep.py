import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.modeling.sequence_data_prep import SequenceDataset, SequenceModelDataPrep


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

    def test_pad_mask(self):
        _assert_tensors_equal(
            self._dataset().pad_mask,
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


class TestSequenceModelDataPrep:
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

    def _prep(self, config=None) -> SequenceModelDataPrep:
        base = {"target": "target_1", "features": {"mode": "exclude", "columns": ["*_career_*"]}}
        return SequenceModelDataPrep(data_dir=self.test_dir, config={**base, **(config or {})})

    def test_prep_training_data__returns_train_eval_test_sequence_datasets(self):
        data = self._prep().prep_training_data()

        assert set(data.keys()) == {"train", "eval", "test"}
        for sequence_dataset in data.values():
            assert sequence_dataset.X.dtype == torch.float32
            assert sequence_dataset.X.shape[0] == len(sequence_dataset.lengths)
            assert sequence_dataset.X.shape[2] == 2
            assert sequence_dataset.feature_columns == ["f1", "f2"]

    def test_prep_training_data__feature_columns_exclude_identity_target_and_configured_patterns(self):
        data = self._prep().prep_training_data()

        assert data["train"].feature_columns == ["f1", "f2"]

        everything = self._prep({"features": {}}).prep_training_data()
        assert everything["train"].feature_columns == ["f1", "f2", "f1_career_avg"]

    def test_prep_training_data__train_split_only_has_players_with_train_loss_steps(self):
        train = self._prep().prep_training_data()["train"]

        # A (T 2021, 2022) and D (T 2021). B and E only have T >= 2023.
        # A's 2022 step (T 2023) is an eval T and is cut.
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

    def test_prep_training_data__splits_hold_history_up_to_last_loss_step_only(self):
        data = self._prep().prep_training_data()

        # Train: A's step for season 2022 has T == 2023 (an eval target), so it and later steps are cut.
        pd.testing.assert_frame_equal(
            data["train"].identity[["player_id", "season"]],
            pd.DataFrame([("A", 2020), ("A", 2021), ("D", 2020)], columns=["player_id", "season"]),
        )
        # Eval: A's training-era history is kept as input, steps after the eval target are cut.
        pd.testing.assert_frame_equal(
            data["eval"].identity[["player_id", "season"]],
            pd.DataFrame(
                [("A", 2020), ("A", 2021), ("A", 2022), ("B", 2022), ("E", 2021)],
                columns=["player_id", "season"],
            ),
        )
        # Test: full sequences.
        pd.testing.assert_frame_equal(
            data["test"].identity[["player_id", "season"]],
            pd.DataFrame(
                [
                    ("A", 2020), ("A", 2021), ("A", 2022), ("A", 2023),
                    ("B", 2022), ("B", 2023),
                    ("E", 2021), ("E", 2023),
                ],
                columns=["player_id", "season"],
            ),
        )

    def test_prep_training_data__train_inputs_contain_nothing_from_after_the_train_period(self):
        train = self._prep().prep_training_data()["train"]

        real_seasons = train.X[train.pad_mask][:, 0]  # f1 == season
        # Latest train feature season is 2021 (target 2022); 2022+ would be eval/test-period information.
        assert real_seasons.max().item() == 2021.0

    def test_prep_training_data__loss_mask_marks_only_steps_with_a_label_in_the_split(self):
        data = self._prep().prep_training_data()

        assert data["train"].loss_mask.tolist() == [[True, True], [True, False]]
        # Eval: only the step whose target_season is 2023 carries loss, earlier steps are context only.
        assert data["eval"].loss_mask.tolist() == [
            [False, False, True],  # A
            [True, False, False],  # B
            [True, False, False],  # E
        ]
        assert data["test"].loss_mask.tolist() == [
            [False, False, False, True],
            [False, True, False, False],
            [False, True, False, False],
        ]

    def test_prep_training_data__y_holds_the_target_on_loss_steps_and_zero_elsewhere(self):
        data = self._prep().prep_training_data()

        assert data["train"].y.tolist() == [[10.0, 11.0], [5.0, 0.0]]
        assert data["eval"].y.tolist() == [[0.0, 0.0, 12.0], [20.0, 0.0, 0.0], [7.0, 0.0, 0.0]]
        assert data["test"].y.tolist() == [
            [0.0, 0.0, 0.0, 13.0],  # A
            [0.0, 21.0, 0.0, 0.0],  # B
            [0.0, 8.0, 0.0, 0.0],   # E
        ]

    def test_prep_training_data__identity_is_aligned_with_real_steps_of_x(self):
        test = self._prep().prep_training_data()["test"]

        real_steps = test.X[test.pad_mask]
        identity = test.identity
        assert len(identity) == len(real_steps)
        # f1 was set to each row's season, f2 to its player.
        assert real_steps[:, 0].tolist() == identity["season"].tolist()
        assert real_steps[:, 1].tolist() == [self.PLAYER_NUMBER[p] * 100.0 for p in identity["player_id"]]
        assert identity["seq_idx"].tolist() == [0, 0, 0, 0, 1, 1, 2, 2]
        assert identity["step_idx"].tolist() == [0, 1, 2, 3, 0, 1, 0, 1]
        assert test.X[identity["seq_idx"].to_numpy(), identity["step_idx"].to_numpy()].equal(real_steps)

    def test_prep_training_data__missed_season_is_a_gap_in_the_sequence_not_a_dummy_step(self):
        test = self._prep().prep_training_data()["test"]

        e_seasons = test.identity[test.identity["player_id"] == "E"]["season"].tolist()
        assert e_seasons == [2021, 2023]

    def test_prep_training_data__training_window_limits_loss_steps_but_not_input_history(self):
        data = self._prep({"split": {"num_training_seasons": 1}}).prep_training_data()
        train = data["train"]

        # Only T == 2022 (A's second step) is in the training window; T == 2021 rows are not scored.
        assert int(train.loss_mask.sum()) == 1
        assert train.loss_mask.tolist() == [[False, True]]
        # But A's earlier season is still part of the input sequence.
        pd.testing.assert_frame_equal(
            train.identity[["player_id", "season"]],
            pd.DataFrame([("A", 2020), ("A", 2021)], columns=["player_id", "season"]),
        )

    def test_prep_training_data__splits_use_configured_years(self):
        data = self._prep({"split": {"eval_data_years": 2, "test_data_years": 1}}).prep_training_data()

        # eval is T in {2022, 2023}: A (step 1 & 2), B, E.
        assert int(data["eval"].loss_mask.sum()) == 4
        assert sorted(data["train"].identity["player_id"].unique()) == ["A", "D"]
        assert int(data["train"].loss_mask.sum()) == 2

    def test_prep_training_data__sample_weights_only_on_train_and_zero_outside_loss_steps(self):
        config = {"sample_weights": {"buckets": [{"min": 0.0, "weight": 0.5}, {"min": 10.5, "weight": 1.0}]}}
        data = self._prep(config).prep_training_data()

        assert data["train"].sample_weight.tolist() == [[0.5, 1.0], [0.5, 0.0]]
        assert data["eval"].sample_weight is None
        assert data["test"].sample_weight is None

    def test_prep_training_data__default_sample_weight_is_one_on_loss_steps(self):
        train = self._prep().prep_training_data()["train"]

        assert train.sample_weight.tolist() == [[1.0, 1.0], [1.0, 0.0]]

    def test_prep_training_data__positions_filter_drops_whole_players(self):
        data = self._prep({"positions": ["WR"]}).prep_training_data()

        assert set(data["test"].identity["player_id"]) == {"A", "E"}
        assert set(data["train"].identity["player_id"]) == {"A"}

    def test_prep_training_data__raises_when_split_needs_more_seasons_than_available(self):
        with pytest.raises(ValueError, match="exceeds"):
            self._prep({"split": {"num_training_seasons": 10}}).prep_training_data()

    def test_prep_training_data__raises_on_duplicate_player_season(self):
        test_dir = tempfile.mkdtemp()
        try:
            training_df = self._gold_frame(self.TRAINING_ROWS + [self.TRAINING_ROWS[0]])
            self._write_gold_tables(test_dir, training_df, self._gold_frame(self.PREDICTION_ROWS))
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="More than one row"):
                prep.prep_training_data()
        finally:
            shutil.rmtree(test_dir)

    def test_prep_training_data__raises_on_non_numeric_feature(self):
        test_dir = tempfile.mkdtemp()
        try:
            training_df = self._gold_frame(self.TRAINING_ROWS)
            training_df["team_name"] = "KC"
            prediction_df = self._gold_frame(self.PREDICTION_ROWS)
            prediction_df["team_name"] = "KC"
            self._write_gold_tables(test_dir, training_df, prediction_df)
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="team_name"):
                prep.prep_training_data()
        finally:
            shutil.rmtree(test_dir)

    def test_prep_training_data__keeps_missing_feature_values_as_nan(self):
        test_dir = tempfile.mkdtemp()
        try:
            training_df = self._gold_frame(self.TRAINING_ROWS)
            training_df.loc[(training_df["player_id"] == "D"), "f2"] = np.nan
            self._write_gold_tables(test_dir, training_df, self._gold_frame(self.PREDICTION_ROWS))
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            train = prep.prep_training_data()["train"]
            d_row = train.identity.index[train.identity["player_id"] == "D"][0]
            assert torch.isnan(train.X[train.pad_mask][d_row, 1])
        finally:
            shutil.rmtree(test_dir)

    def test_prep_prediction_data__one_sequence_per_listed_player_history_then_prediction_row(self):
        prediction = self._prep().prep_prediction_data()

        # D and E are not in the prediction set, F has only the prediction row.
        pd.testing.assert_frame_equal(
            prediction.identity[["player_id", "season"]],
            pd.DataFrame(
                [
                    ("A", 2020), ("A", 2021), ("A", 2022), ("A", 2023), ("A", 2024),
                    ("B", 2022), ("B", 2023), ("B", 2024),
                    ("F", 2024),
                ],
                columns=["player_id", "season"],
            ),
        )
        assert prediction.lengths.tolist() == [5, 3, 1]
        assert prediction.X.shape == (3, 5, 2)
        assert prediction.y is None and prediction.loss_mask is None and prediction.sample_weight is None

    def test_prep_prediction_data__last_step_is_the_prediction_row(self):
        prediction = self._prep().prep_prediction_data()

        last = prediction.last_step_identity()
        assert last["player_id"].tolist() == ["A", "B", "F"]
        assert last["season"].tolist() == [2024, 2024, 2024]
        assert last["target_season"].tolist() == [2025, 2025, 2025]
        assert prediction.last_step_mask.tolist() == [
            [False, False, False, False, True],
            [False, False, True, False, False],
            [True, False, False, False, False],
        ]
        assert prediction.X[prediction.last_step_mask][:, 0].tolist() == [2024.0, 2024.0, 2024.0]

    def test_prep_prediction_data__respects_position_filter(self):
        prediction = self._prep({"positions": ["WR", "TE"]}).prep_prediction_data()

        assert set(prediction.identity["player_id"]) == {"A", "F"}

    def test_prep_prediction_data__raises_when_prediction_row_is_not_the_latest_season(self):
        test_dir = tempfile.mkdtemp()
        try:
            stale_prediction_rows = [("A", 2019, 2025, np.nan)]
            self._write_gold_tables(test_dir, self._gold_frame(self.TRAINING_ROWS), self._gold_frame(stale_prediction_rows))
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="latest season"):
                prep.prep_prediction_data()
        finally:
            shutil.rmtree(test_dir)

    def test_prep_prediction_data__raises_when_prediction_row_repeats_a_training_season(self):
        test_dir = tempfile.mkdtemp()
        try:
            repeated_prediction_rows = [("A", 2023, 2025, np.nan)]
            self._write_gold_tables(test_dir, self._gold_frame(self.TRAINING_ROWS), self._gold_frame(repeated_prediction_rows))
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="More than one row"):
                prep.prep_prediction_data()
        finally:
            shutil.rmtree(test_dir)

    def test_prep_prediction_data__raises_when_columns_differ(self):
        test_dir = tempfile.mkdtemp()
        try:
            prediction_df = self._gold_frame(self.PREDICTION_ROWS).drop(columns=["f2"])
            self._write_gold_tables(test_dir, self._gold_frame(self.TRAINING_ROWS), prediction_df)
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="same columns"):
                prep.prep_prediction_data()
        finally:
            shutil.rmtree(test_dir)

    def test_from_config_file__loads_yaml_config(self):
        config_path = os.path.join(self.test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("target: target_1\nfeatures:\n  mode: exclude\n  columns:\n    - '*_career_*'\n")

        prep = SequenceModelDataPrep.from_config_file(self.test_dir, config_path)

        assert isinstance(prep, SequenceModelDataPrep)
        assert prep.prep_training_data()["train"].feature_columns == ["f1", "f2"]
