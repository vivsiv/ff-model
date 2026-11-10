import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.modeling.data_builder import PREDICTION_SPLIT, TRAIN_SPLIT, ModelDataBuilder
from src.processing.gold import TARGET_COL, load_gold_set

logger = logging.getLogger(__name__)

#TODO: these are defined on the gold table, name these constants there.
PLAYER_COL = "player_id"
SEASON_COL = "season"
TARGET_SEASON_COL = "target_season"

# Per-step columns attached to a split's steps while it is being assembled, so every per-step value is trimmed
# and ordered together with the rows. Never features/identity (those are resolved from the gold columns first).
OUTPUT_STEP_COL = "_is_output_step"
LOSS_STEP_COL = "_is_loss_step"
SAMPLE_WEIGHT_COL = "_sample_weight"


@dataclass
class SequenceDataset:
    """
    A sequential dataset for a split (train, eval, test, or prediction).

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
            Only has rows for real steps (aligning with X[real_step_mask]). Besides the columns in the example it has
            the other identity columns (e.g. season, target_season).
            Example:
                identity = pd.DataFrame([
                    {"player_id": "Player A", "seq_idx": 0, "step_idx": 0},
                    {"player_id": "Player A", "seq_idx": 0, "step_idx": 1},
                    {"player_id": "Player A", "seq_idx": 0, "step_idx": 2},
                    {"player_id": "Player B", "seq_idx": 1, "step_idx": 0},
                ])
        feature_columns: the names of the features in F (X's last dimension).
        y: [N, L] The target tensor (rank 2, float32). Holds the target value at steps where loss_mask is True and 0.0 everywhere else (padding and unscored steps).
        loss_mask: [N, L] Tensor (rank 2, bool) indicating which target values in y should count towards loss.
        sample_weight: [N, L] Tensor (rank 2, float32, train split only) with how much weight each real step is given in training, 0.0 wherever loss_mask is False.
    """
    X: torch.Tensor
    lengths: torch.Tensor # TODO: rename to steps?
    identity: pd.DataFrame
    feature_columns: List[str]
    y: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    sample_weight: Optional[torch.Tensor] = None

    @staticmethod
    def build_real_step_mask(lengths: torch.Tensor, max_steps: int) -> torch.Tensor:
        """Return an [N, max_steps] boolean mask on the lengths tensor's device."""
        # Compare every step index with each player's length using broadcasting.
        return torch.arange(max_steps, device=lengths.device)[None, :] < lengths[:, None]

    @property
    def real_step_mask(self) -> torch.Tensor:
        """
        [N, L] bool, True for real steps, False for padding.
        Example:
            real_step_mask = torch.tensor([
                [True, True, True], # Player A: 3 real steps
                [True, False, False], # Player B: 1 real step
            ])

        """
        return self.build_real_step_mask(self.lengths.to(self.X.device), self.X.shape[1])

    @property
    def last_step_mask(self) -> torch.Tensor:
        """[N, L] bool, True at each sequence's last real step."""
        max_step = self.X.shape[1]
        return torch.arange(max_step)[None, :] == (self.lengths[:, None] - 1)

    def last_step_identity(self) -> pd.DataFrame:
        """Identity rows of each sequence's last real step, ordered by seq_idx."""
        return self.identity.groupby("seq_idx").tail(1)


class SequenceModelDataBuilder(ModelDataBuilder):
    """
    Builds SequenceDatasets from a target's combined, season-specific gold table.
    Uses ModelDataBuilder's shared config schema to define:
    the target, seasons per data split, sample weighting for training, and feature inclusion/exclusion.
    """

    def _build_data_splits(self) -> dict[str, SequenceDataset]:
        """
        Loads the gold data set, splits it by the builder's split config, and assembles each split into a SequenceDataset.

        Splitting is done on a step's target_season.

        - A step contributes to loss for the split when it has a target value and its target_season is in the split's seasons.
        - Every split holds each player's full history through their last output step:
          a loss step for train/eval/test, or the prediction endpoint for prediction.
          An eval step sees training-era history, and num_training_seasons limits
          loss steps without hiding older history from the inputs.
        - Steps after a player's last output step are removed, including for prediction,
          so inputs never extend beyond the season being scored or predicted.
        - Players with no output steps in a split are left out of it.

        Returns:
            dict with "train", "eval", "test", "prediction" SequenceDatasets.
            Prediction holds each selected player's history through their prediction step,
            with y zeroed and loss_mask false even if gold contains known targets.
            Sample weights are set on "train" only.
            Always builds fresh data; use get_data_splits() for cached access.

        Raises:
            ValueError: If split config's num_training_seasons + eval_data_years + test_data_years exceeds
                the total number of distinct seasons available.
        """
        self.source_data = load_gold_set(self.gold_data_dir, self.target, self.split_years.prediction_season)

        data = self.source_data
        if self.positions:
            data = self._filter_positions(data, self.positions)

        feature_cols = self._resolve_feature_columns(data.columns)

        data = self._order_by_step(data)
        split_masks = self.split_years.to_masks(data[TARGET_SEASON_COL])
        is_labeled = data[TARGET_COL].notna() & data[TARGET_SEASON_COL].notna()
        sample_weights = self._compute_sample_weights(data[TARGET_COL])

        datasets = {}
        for split, split_mask in split_masks.items():
            # Output steps define input endpoints; prediction outputs never contribute to loss.
            output_steps = split_mask if split == PREDICTION_SPLIT else split_mask & is_labeled
            split_data = data.assign(**{OUTPUT_STEP_COL: output_steps})
            split_data = split_data[self._get_input_rows_mask(split_data)]

            split_data = split_data.assign(**{
                LOSS_STEP_COL: split_data[OUTPUT_STEP_COL] & (split != PREDICTION_SPLIT),
            })

            if split == PREDICTION_SPLIT:
                split_data = split_data.assign(**{TARGET_COL: 0.0})

            if split == TRAIN_SPLIT:
                split_data[SAMPLE_WEIGHT_COL] = sample_weights.reindex(split_data.index)

            datasets[split] = self._build_sequence_dataset(split_data, feature_cols)
            logger.info(
                f"{split}: {len(datasets[split].lengths)} sequences, "
                f"{int(datasets[split].loss_mask.sum())} loss steps"
            )

        return datasets

    @staticmethod
    def _order_by_step(data: pd.DataFrame) -> pd.DataFrame:
        """
        Orders rows by player and season, adds a "step_idx" that starts at 0 for each player and increases for each player-season combo.

        """
        data = data.sort_values([PLAYER_COL, SEASON_COL]).reset_index(drop=True)
        data["step_idx"] = data.groupby(PLAYER_COL).cumcount()
        return data

    @staticmethod
    def _get_input_rows_mask(data: pd.DataFrame) -> pd.Series:
        """
        Finds the rows a split's dataset reads as input: each player's steps from their first season up to and
        including their last output step in the split. Output endpoints control
        input history independently of whether a step contributes to loss.

        Args:
            data: Output of _order_by_step with an OUTPUT_STEP_COL column.

        Returns:
            Boolean Series aligned with data, True for input rows. False for every step of a player with no output
            steps in the split, since such players are left out of the split entirely.
        """
        last_output_step = data["step_idx"].where(data[OUTPUT_STEP_COL]).groupby(data[PLAYER_COL]).transform("max")
        return data["step_idx"] <= last_output_step

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
