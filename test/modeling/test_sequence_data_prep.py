import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.modeling.sequence_data_prep import SequenceModelDataPrep

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


def _frame(rows) -> pd.DataFrame:
    """f1 is the row's season so a tensor entry can be traced back to its season, f2 identifies the player."""
    return pd.DataFrame({
        "player_id": [r[0] for r in rows],
        "player_display_name": [f"Player {r[0]}" for r in rows],
        "position": [POSITIONS[r[0]] for r in rows],
        "season": [r[1] for r in rows],
        "recent_team": ["KC"] * len(rows),
        "target_season": [r[2] for r in rows],
        "f1": [float(r[1]) for r in rows],
        "f2": [float(PLAYER_NUMBER[r[0]] * 100) for r in rows],
        "f1_career_avg": [float(r[1]) * 10 for r in rows],
        "target": [r[3] for r in rows],
    })


def _write(data_dir, training_df, prediction_df):
    gold_dir = os.path.join(data_dir, "gold")
    os.makedirs(gold_dir, exist_ok=True)
    training_df.to_csv(os.path.join(gold_dir, "target_1__training_set.csv"), index=False)
    prediction_df.to_csv(os.path.join(gold_dir, "target_1__prediction_set.csv"), index=False)


def _step_ids(sequence_set):
    """(player, season) of every real step, in X[pad_mask] order."""
    return list(zip(sequence_set.identity["player_id"], sequence_set.identity["season"]))


class TestSequenceModelDataPrep:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        _write(cls.test_dir, _frame(TRAINING_ROWS), _frame(PREDICTION_ROWS))

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def _prep(self, config=None) -> SequenceModelDataPrep:
        base = {"target": "target_1", "features": {"mode": "exclude", "columns": ["*_career_*"]}}
        return SequenceModelDataPrep(data_dir=self.test_dir, config={**base, **(config or {})})

    def test_prep_training_data__returns_train_eval_test_sequence_sets(self):
        data = self._prep().prep_training_data()

        assert set(data.keys()) == {"train", "eval", "test"}
        for sequence_set in data.values():
            assert sequence_set.X.dtype == torch.float32
            assert sequence_set.X.shape[0] == len(sequence_set.lengths)
            assert sequence_set.X.shape[2] == 2
            assert sequence_set.feature_columns == ["f1", "f2"]

    def test_prep_training_data__feature_columns_exclude_identity_target_and_configured_patterns(self):
        data = self._prep().prep_training_data()

        assert data["train"].feature_columns == ["f1", "f2"]

        everything = self._prep({"features": {}}).prep_training_data()
        assert everything["train"].feature_columns == ["f1", "f2", "f1_career_avg"]

    def test_prep_training_data__train_split_only_has_players_with_train_loss_steps(self):
        train = self._prep().prep_training_data()["train"]

        # A (T 2021, 2022) and D (T 2021). B and E only have T >= 2023.
        assert sorted(train.identity["player_id"].unique()) == ["A", "D"]
        assert train.lengths.tolist() == [2, 1]
        assert int(train.loss_mask.sum()) == 3

    def test_prep_training_data__splits_hold_history_up_to_last_loss_step_only(self):
        data = self._prep().prep_training_data()

        # Train: A's step for season 2022 has T == 2023 (an eval target), so it and later steps are cut.
        assert _step_ids(data["train"]) == [("A", 2020), ("A", 2021), ("D", 2020)]
        # Eval: A's training-era history is kept as input, steps after the eval target are cut.
        assert _step_ids(data["eval"]) == [("A", 2020), ("A", 2021), ("A", 2022), ("B", 2022), ("E", 2021)]
        # Test: full sequences.
        assert _step_ids(data["test"]) == [
            ("A", 2020), ("A", 2021), ("A", 2022), ("A", 2023),
            ("B", 2022), ("B", 2023),
            ("E", 2021), ("E", 2023),
        ]

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
        assert data["test"].y[:, 3].tolist()[0] == 13.0

    def test_prep_training_data__pads_right_with_nan_and_reports_lengths(self):
        train = self._prep().prep_training_data()["train"]

        assert train.X.shape == (2, 2, 2)
        assert train.lengths.tolist() == [2, 1]
        assert train.pad_mask.tolist() == [[True, True], [True, False]]
        assert torch.isnan(train.X[1, 1]).all()
        assert not torch.isnan(train.X[train.pad_mask]).any()

    def test_prep_training_data__identity_is_aligned_with_real_steps_of_x(self):
        test = self._prep().prep_training_data()["test"]

        real_steps = test.X[test.pad_mask]
        identity = test.identity
        assert len(identity) == len(real_steps)
        # f1 was set to each row's season, f2 to its player.
        assert real_steps[:, 0].tolist() == identity["season"].tolist()
        assert real_steps[:, 1].tolist() == [PLAYER_NUMBER[p] * 100.0 for p in identity["player_id"]]
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
        assert _step_ids(train) == [("A", 2020), ("A", 2021)]

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
            training_df = _frame(TRAINING_ROWS + [TRAINING_ROWS[0]])
            _write(test_dir, training_df, _frame(PREDICTION_ROWS))
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="More than one row"):
                prep.prep_training_data()
        finally:
            shutil.rmtree(test_dir)

    def test_prep_training_data__raises_on_non_numeric_feature(self):
        test_dir = tempfile.mkdtemp()
        try:
            training_df = _frame(TRAINING_ROWS)
            training_df["team_name"] = "KC"
            prediction_df = _frame(PREDICTION_ROWS)
            prediction_df["team_name"] = "KC"
            _write(test_dir, training_df, prediction_df)
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="team_name"):
                prep.prep_training_data()
        finally:
            shutil.rmtree(test_dir)

    def test_prep_training_data__keeps_missing_feature_values_as_nan(self):
        test_dir = tempfile.mkdtemp()
        try:
            training_df = _frame(TRAINING_ROWS)
            training_df.loc[(training_df["player_id"] == "D"), "f2"] = np.nan
            _write(test_dir, training_df, _frame(PREDICTION_ROWS))
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            train = prep.prep_training_data()["train"]
            d_row = train.identity.index[train.identity["player_id"] == "D"][0]
            assert torch.isnan(train.X[train.pad_mask][d_row, 1])
        finally:
            shutil.rmtree(test_dir)

    def test_prep_prediction_data__one_sequence_per_listed_player_history_then_prediction_row(self):
        prediction = self._prep().prep_prediction_data()

        # D and E are not in the prediction set, F has only the prediction row.
        assert _step_ids(prediction) == [
            ("A", 2020), ("A", 2021), ("A", 2022), ("A", 2023), ("A", 2024),
            ("B", 2022), ("B", 2023), ("B", 2024),
            ("F", 2024),
        ]
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
            _write(test_dir, _frame(TRAINING_ROWS), _frame(stale_prediction_rows))
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="latest season"):
                prep.prep_prediction_data()
        finally:
            shutil.rmtree(test_dir)

    def test_prep_prediction_data__raises_when_prediction_row_repeats_a_training_season(self):
        test_dir = tempfile.mkdtemp()
        try:
            repeated_prediction_rows = [("A", 2023, 2025, np.nan)]
            _write(test_dir, _frame(TRAINING_ROWS), _frame(repeated_prediction_rows))
            prep = SequenceModelDataPrep(data_dir=test_dir, config={"target": "target_1"})

            with pytest.raises(ValueError, match="More than one row"):
                prep.prep_prediction_data()
        finally:
            shutil.rmtree(test_dir)

    def test_prep_prediction_data__raises_when_columns_differ(self):
        test_dir = tempfile.mkdtemp()
        try:
            prediction_df = _frame(PREDICTION_ROWS).drop(columns=["f2"])
            _write(test_dir, _frame(TRAINING_ROWS), prediction_df)
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
