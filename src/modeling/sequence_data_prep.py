import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.modeling.data_prep import SplitYears, TabularModelDataPrep
from src.processing.gold import TARGET_COL, load_prediction_set, load_training_set

logger = logging.getLogger(__name__)

#TODO: these are defined on the gold table, name these constants there.
PLAYER_COL = "player_id"
SEASON_COL = "season"
TARGET_SEASON_COL = "target_season"

# Per-step columns attached to a split's steps while it is being assembled, so every per-step value is trimmed
# and ordered together with the rows. Never features/identity (those are resolved from the gold columns first).
LOSS_STEP_COL = "_is_loss_step"
SAMPLE_WEIGHT_COL = "_sample_weight"


@dataclass
class SequenceDataset:
    """
    A sequential dataset for a split (train, eval, or test).

    Each tabular row (player, season, target_season, target) becomes one step in a player's sequence, steps are ordered by season.
    The target is the player's target value for a step's target_season (the player's next active season).
    A sequential model reads steps 0..t to make a prediction for step t's target season (step t+1 or the prediction season).

    Fields:
        X: [N, L, F] The raw data tensor (rank 3, float32) N is players, L is steps, and F is the features for a given player at a given step (float32).
         Players are padded with steps (L) of NaN features, matching the player with the highest number of steps (L) in the set.
         Example:
            X = torch.tensor([
                # Player A: 3 real steps
                [ 
                    [1., 10.],
                    [2., 20.],
                    [3., 30.]
                ],
                # Player B: 1 real step + 2 padding     
                [
                    [4., 40.],
                    [nan, nan],
                    [nan, nan]
                ],
            ])
        lengths: [N] Tensor (rank 1, int64) tracking the number of real steps per player.
            Example:
                lengths = torch.tensor([3, 1]) # Player A has 3 real steps, player B has 1
        identity: Dataframe mapping sequences and steps in X back to player_ids in the tabular data.
            Only has rows for real steps (aligning with X[pad_mask])
            Example:
                identity = pd.dataframe(
                    {player_id: "Player A", seq_idx: 0, step_idx: 0},
                    {player_id: "Player A", seq_idx: 0, step_idx: 1},
                    {player_id: "Player A", seq_idx: 0, step_idx: 2},
                    {player_id: "Player B", seq_idx: 1, step_idx: 0},
                )
        feature_columns: the names of the features in F (X's last dimension).
        y: [N, L] The target tensor (rank 2, float32) contains target values for each real (player,step) combo, 0.0 wherever loss_mask is False.
        loss_mask: [N, L] Tensor (rank 2, bool) indicating which target values in y should count towards loss.
        sample_weight: [N, L] Tensor (rank2, float32, training only) tensor indicating how much weight a real step should be given in training, 0.0 wherever loss_mask is False.
    """
    X: torch.Tensor
    lengths: torch.Tensor
    identity: pd.DataFrame
    feature_columns: List[str]
    y: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    sample_weight: Optional[torch.Tensor] = None

    @property
    def pad_mask(self) -> torch.Tensor:
        """
        [N, L] bool, True for real steps, False for padding.
        Example:
            pad_mask = torch.tensor([
                [True, True, True], # Player A: 3 real steps
                [True, False, False], # Player B: 1 real step.  
            ])

        """
        max_step = self.X.shape[1] # torch.tensor([0, 1, 2])
        # torch.tensor([0, 1, 2]) < torch.tensor([[3], [1]]) -> torch.tensor([T, T, T], [T, F, F]) (broadcasting on the steps tensor)
        return torch.arange(max_step)[None, :] < self.lengths[:, None]

    @property
    def last_step_mask(self) -> torch.Tensor:
        """[N, L] bool, True at each sequence's last real step."""
        max_step = self.X.shape[1]
        return torch.arange(max_step)[None, :] == (self.lengths[:, None] - 1)

    def last_step_identity(self) -> pd.DataFrame:
        """Identity rows of each sequence's last real step, in sequence order (aligned with
        model_output[last_step_mask])."""
        # identity is sorted by sequence then step, so each sequence's last row is its last real step.
        return self.identity.groupby("seq_idx").tail(1)


class SequenceModelDataPrep(TabularModelDataPrep):
    """
    Loads a target's tabular training/prediction sets and assembles them into SequenceDatasets.
    Uses the same config schema as TabularModelDataPrep's to define:
    the target, years per each data split, sample weighting for training, and feature inclusion/exlusion.
    """

    def prep_training_data(self) -> dict[str, SequenceDataset]:
        """
        Loads the gold data set, splits it by the DataPreps split config, and assembles each split into a SequenceDataset.

        Splitting is done on a step's target_season.

        - A step contributes to loss for the split when it has a target value and its target_season is in the split's seasons.
        - Every split holds each player's full history up to their last loss step as input, so an eval
          step sees the player's training-era seasons, and a training window (num_training_seasons) limits
          loss steps without hiding older history from the inputs.
        - Steps after a player's last loss step are removed, so a split never contains inputs from a
          period after what it scores, whatever the model architecture (a bidirectional model could not leak).
        - Players with no loss steps in a split are left out of it.

        Returns:
            dict with "train", "eval", "test" SequenceDatasets (y/loss_mask set, sample_weight on "train" only).

        Raises:
            ValueError: If split config's num_training_seasons + eval_data_years + test_data_years exceeds
                the total number of distinct seasons available, if a player has two rows for one season, or if a
                selected feature is not numeric.
        """
        self.training_data = load_training_set(self.gold_data_dir, self.target)

        data = self.training_data
        if self.positions:
            data = self._filter_positions(data, self.positions)

        feature_cols = self._resolve_feature_columns(data.columns)
        self._check_numeric(data, feature_cols)

        data = self._order_by_step(data)
        split_masks = SplitYears.from_config(self.config.get("split", {})).to_masks(data[TARGET_SEASON_COL])
        is_labeled = data[TARGET_COL].notna() & data[TARGET_SEASON_COL].notna()
        sample_weights = self._compute_sample_weights(data[TARGET_COL])

        datasets = {}
        for split, in_split in [("train", split_masks.is_train), ("eval", split_masks.is_eval), ("test", split_masks.is_test)]:
            split_data = data.assign(**{LOSS_STEP_COL: in_split & is_labeled})
            split_data = split_data[self._get_input_rows_mask(split_data)]
            if split == "train":
                split_data[SAMPLE_WEIGHT_COL] = sample_weights

            datasets[split] = self._build_sequence_dataset(split_data, feature_cols)
            logger.info(
                f"{split}: {len(datasets[split].lengths)} sequences, "
                f"{int(datasets[split].loss_mask.sum())} loss steps"
            )

        return datasets

    def prep_prediction_data(self) -> SequenceDataset:
        """
        Assembles the sequences to predict from.
        Takes each player in the prediction set and concats their rows from the training set (their history).
        Their row from the prediction set becomes their last step, which is the one used to generate the prediction.

        Returns:
            SequenceDataset (no y/loss_mask/sample_weight) with one sequence per player in the prediction set.
            Use last_step_mask / last_step_identity() to pick out the predictions.

        Raises:
            ValueError: If the training and prediction sets don't have the same columns, a prediction row's season
                is not later than all of the player's training rows, or a selected feature is not numeric.
        """
        training_data = load_training_set(self.gold_data_dir, self.target)
        prediction_data = load_prediction_set(self.gold_data_dir, self.target)

        if self.positions:
            training_data = self._filter_positions(training_data, self.positions)
            prediction_data = self._filter_positions(prediction_data, self.positions)

        if set(training_data.columns) != set(prediction_data.columns):
            raise ValueError("Training set and prediction set must have the same columns")

        history = training_data[training_data[PLAYER_COL].isin(prediction_data[PLAYER_COL])]
        combined = pd.concat([history, prediction_data[history.columns]], ignore_index=True)

        feature_cols = self._resolve_feature_columns(combined.columns)
        self._check_numeric(combined, feature_cols)
        data = self._order_by_step(combined)

        # Gets the season for each player in the prediction set and compares it to the max season for each player in the combined set.
        # TODO: mode data validations into gold.
        prediction_seasons = prediction_data.set_index(PLAYER_COL)[SEASON_COL]
        last_seasons = data.groupby(PLAYER_COL)[SEASON_COL].transform("max")
        if (last_seasons != data[PLAYER_COL].map(prediction_seasons)).any():
            raise ValueError("A player's prediction row must be their latest season")

        return self._build_sequence_dataset(data, feature_cols)

    @staticmethod
    def _check_numeric(data: pd.DataFrame, feature_cols: List[str]) -> None:
        non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric:
            raise ValueError(f"Selected features must be numeric, got non-numeric: {non_numeric}")

    @staticmethod
    def _order_by_step(data: pd.DataFrame) -> pd.DataFrame:
        """
        Orders rows by player and season, adds a "step_idx" that starts at 0 for each player and increases for each player-season combo.

        Raises:
            ValueError: If a player has two rows for the same season.
        """
        #TODO: move data validation into gold.
        if data.duplicated([PLAYER_COL, SEASON_COL]).any():
            raise ValueError(f"More than one row for some ({PLAYER_COL}, {SEASON_COL})")

        steps = data.sort_values([PLAYER_COL, SEASON_COL]).reset_index(drop=True)
        steps["step_idx"] = steps.groupby(PLAYER_COL).cumcount()
        return steps

    @staticmethod
    def _get_input_rows_mask(steps: pd.DataFrame) -> pd.Series:
        """
        Finds the rows a split's dataset reads as input: each player's steps from their first season up to and
        including their last loss step in the split.

        Args:
            steps: Output of _order_by_step with a LOSS_STEP_COL column.

        Returns:
            Boolean Series aligned with steps, True for input rows. False for every step of a player with no loss
            steps in the split, since such players are left out of the split entirely.
        """
        last_loss_step = steps["step_idx"].where(steps[LOSS_STEP_COL]).groupby(steps[PLAYER_COL]).transform("max")
        return steps["step_idx"] <= last_loss_step

    def _build_sequence_dataset(self, data: pd.DataFrame, feature_cols: List[str]) -> SequenceDataset:
        """
        Packs data ordered by step_idx into SequenceDatasets.

        Args:
            data: Output of _order_by_step (possibly with trailing steps removed), sorted by player then step.
                Optionally has:
                  - LOSS_STEP_COL: boolean, True for steps that count toward the loss. When present y and
                    loss_mask are filled.
                  - SAMPLE_WEIGHT_COL: per-step weight, only used together with LOSS_STEP_COL. When present
                    sample_weight is filled, zeroed outside loss steps.
            feature_cols: Columns to use as features.
        """
        player_ids = data[PLAYER_COL].to_numpy()
        seq_idx = pd.factorize(player_ids)[0]  # steps are sorted by player, so this is sequence order
        step_idx = data["step_idx"].to_numpy()

        num_sequences = int(seq_idx.max()) + 1 if len(data) else 0
        max_steps = int(step_idx.max()) + 1 if len(data) else 0

        X = np.full((num_sequences, max_steps, len(feature_cols)), np.nan, dtype=np.float32)
        X[seq_idx, step_idx] = data[feature_cols].to_numpy(dtype=np.float32)
        lengths = np.bincount(seq_idx, minlength=num_sequences)

        identity = data[self.identity_cols].copy()
        identity.insert(0, "step_idx", step_idx)
        identity.insert(0, "seq_idx", seq_idx)

        def empty_seq_tensor(dtype: torch.dtype) -> torch.Tensor:
            """[N, L] tensor of zeros (False for bool), one cell per (sequence, step) slot of X."""
            return torch.zeros((num_sequences, max_steps), dtype=dtype)

        y = loss_mask = sample_weight = None
        if LOSS_STEP_COL in data.columns:
            is_loss_step = data[LOSS_STEP_COL].to_numpy(dtype=bool)

            loss_mask = empty_seq_tensor(torch.bool)
            loss_mask[seq_idx, step_idx] = torch.as_tensor(is_loss_step, dtype=loss_mask.dtype)

            y = empty_seq_tensor(torch.float32)
            y[seq_idx, step_idx] = torch.as_tensor(np.where(is_loss_step, data[TARGET_COL], 0.0), dtype=y.dtype)

            if SAMPLE_WEIGHT_COL in data.columns:
                sample_weight = empty_seq_tensor(torch.float32)
                sample_weight[seq_idx, step_idx] = torch.as_tensor(
                    np.where(is_loss_step, data[SAMPLE_WEIGHT_COL], 0.0), dtype=sample_weight.dtype
                )

        return SequenceDataset(
            X=torch.from_numpy(X),
            lengths=torch.from_numpy(lengths.astype(np.int64)),
            identity=identity.reset_index(drop=True),
            feature_columns=list(feature_cols),
            y=y,
            loss_mask=loss_mask,
            sample_weight=sample_weight,
        )
