import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.modeling.data_prep import SplitYears, TabularModelDataPrep
from src.processing.gold import TARGET_COL, prediction_set_filename, training_set_filename

logger = logging.getLogger(__name__)

PLAYER_COL = "player_id"
SEASON_COL = "season"
TARGET_SEASON_COL = "target_season"


@dataclass
class SequenceSet:
    """
    A batch of variable length player sequences, right-padded to the longest one.

    Each tabular gold row (player, season, target_season, target) is one step of that player's
    sequence, steps are ordered by season. The target at a step is the value in that step's
    target_season (the player's next active season), so a causal model reading steps 0..t makes
    its prediction for step t's target_season at step t.

    Fields:
        X: [N, L, F] float32. Padded steps are NaN, and so is any real missing value (e.g. snap counts
            before 2013, draft_pick/age for undrafted players). Imputing/scaling is the model's job
            (the analog of the tabular sklearn pipeline).
        lengths: [N] int64 number of real steps per sequence.
        identity: one row per real step, aligned with X[pad_mask] (sequence-major, then step order). Has
            "seq_idx"/"step_idx" (the step's index into X) plus the identity columns, to map outputs back to players.
        feature_columns: the F feature names, in X's last dimension order.
        y: [N, L] float32 target, 0.0 wherever loss_mask is False (training splits only).
        loss_mask: [N, L] bool, True for steps that count toward the loss/metric (training splits only).
        sample_weight: [N, L] float32, already 0.0 wherever loss_mask is False (train split only).
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
        """[N, L] bool, True for real steps, False for padding."""
        max_len = self.X.shape[1]
        return torch.arange(max_len)[None, :] < self.lengths[:, None]

    @property
    def last_step_mask(self) -> torch.Tensor:
        """[N, L] bool, True at each sequence's last real step."""
        max_len = self.X.shape[1]
        return torch.arange(max_len)[None, :] == (self.lengths[:, None] - 1)

    def last_step_identity(self) -> pd.DataFrame:
        """Identity rows of each sequence's last real step, in sequence order (aligned with
        model_output[last_step_mask])."""
        last_step_idx = (self.lengths - 1).numpy()
        return self.identity[self.identity["step_idx"] == last_step_idx[self.identity["seq_idx"].to_numpy()]]


class SequenceModelDataPrep(TabularModelDataPrep):
    """
    Loads a target's tabular gold training/prediction sets and assembles them into per-player
    sequences for a sequential model (lstm/transformer), with the masks needed to train on and
    score them correctly.

    No separate sequence gold table: a tabular row already is one step. It holds only data up
    to its own "season", plus the label for its "target_season" (the player's next active season).

    Config schema is the same as TabularModelDataPrep (target/positions/split/sample_weights/features),
    with these differences in meaning:
        - rows are steps, so splitting (by target_season) and sample weights are applied per step
            and only decide which steps count toward the loss -- see prep_training_data.
        - positions filters rows; players never change position so a filter drops whole players.
        - features: the tabular gold rows contain career aggregate columns (*_career_*, *_trend, *_shrunk_avg,
            *_positional_baseline) that exist to give a non-sequential model a summary of a player's history.
            A sequence config will usually exclude them (see configs/sequence_*.yaml).
    """

    def prep_training_data(self) -> dict[str, SequenceSet]:
        """
        Loads the gold training set, assembles it into sequences and splits it chronologically.

        Splits are by each step's target_season (same cutoffs as TabularModelDataPrep.split). The split
        decides which steps carry loss, not which steps a sequence contains:
          - A step is a loss step of a split when it has a label and its target_season is in the split's seasons.
          - Every split holds each player's full history up to their last loss step as input, so an eval
            step sees the player's training-era seasons, and a training window (num_training_seasons) limits
            loss steps without hiding older history from the inputs.
          - Steps after a player's last loss step are removed, so a split never contains inputs from a
            period after what it scores, whatever the model architecture (a bidirectional model could not leak).
          - Players with no loss steps in a split are left out of it.

        Returns:
            dict with "train", "eval", "test" SequenceSets (y/loss_mask set, sample_weight on "train" only).

        Raises:
            ValueError: If split config's num_training_seasons + eval_data_years + test_data_years exceeds
                the total number of distinct seasons available, if a player has two rows for one season, or if a
                selected feature is not numeric.
        """
        filename = training_set_filename(self.target)
        self.training_data = pd.read_csv(os.path.join(self.gold_data_dir, filename))
        logger.info(f"Loaded data: {len(self.training_data)} rows")

        data = self.training_data
        if self.positions:
            data = self._filter_positions(data, self.positions)

        feature_cols = self._resolve_feature_columns(data.columns)
        self._check_numeric(data, feature_cols)

        steps = self._order_steps(data)
        masks = SplitYears.from_config(self.config.get("split", {})).to_masks(steps[TARGET_SEASON_COL])
        is_labeled = steps[TARGET_COL].notna() & steps[TARGET_SEASON_COL].notna()
        sample_weights = self._compute_sample_weights(steps[TARGET_COL])

        sets = {}
        for name, in_split in [("train", masks.is_train), ("eval", masks.is_eval), ("test", masks.is_test)]:
            loss_rows = in_split & is_labeled
            keep = self._rows_up_to_last_loss_step(steps, loss_rows)
            sets[name] = self._build_sequence_set(
                steps[keep],
                feature_cols,
                loss_rows=loss_rows[keep],
                sample_weights=sample_weights[keep] if name == "train" else None,
            )
            logger.info(
                f"{name}: {len(sets[name].lengths)} sequences, "
                f"{int(sets[name].loss_mask.sum())} loss steps"
            )

        return sets

    def prep_prediction_data(self) -> SequenceSet:
        """
        Assembles the sequences to predict from: for every player in the gold prediction set, their rows
        from the gold training set (their history) followed by their prediction row, which is their last
        step. The model's output at each sequence's last step is the prediction.

        Returns:
            SequenceSet (no y/loss_mask/sample_weight) with one sequence per player in the prediction set.
            Use last_step_mask / last_step_identity() to pick out the predictions.

        Raises:
            ValueError: If the training and prediction sets don't have the same columns, a prediction row's season
                is not later than all of the player's training rows, or a selected feature is not numeric.
        """
        training_data = pd.read_csv(os.path.join(self.gold_data_dir, training_set_filename(self.target)))
        prediction_data = pd.read_csv(os.path.join(self.gold_data_dir, prediction_set_filename(self.target)))
        logger.info(f"Loaded prediction data: {len(prediction_data)} rows")

        if self.positions:
            training_data = self._filter_positions(training_data, self.positions)
            prediction_data = self._filter_positions(prediction_data, self.positions)

        if set(training_data.columns) != set(prediction_data.columns):
            raise ValueError("Training set and prediction set must have the same columns")

        history = training_data[training_data[PLAYER_COL].isin(prediction_data[PLAYER_COL])]
        combined = pd.concat([history, prediction_data[history.columns]], ignore_index=True)

        feature_cols = self._resolve_feature_columns(combined.columns)
        self._check_numeric(combined, feature_cols)

        steps = self._order_steps(combined)
        # The prediction row is a player's latest season; a training row after it would mean the
        # sequence's last step is not the one being predicted.
        prediction_seasons = prediction_data.set_index(PLAYER_COL)[SEASON_COL]
        last_seasons = steps.groupby(PLAYER_COL)[SEASON_COL].transform("max")
        if (last_seasons != steps[PLAYER_COL].map(prediction_seasons)).any():
            raise ValueError("A player's prediction row must be their latest season")

        return self._build_sequence_set(steps, feature_cols)

    @staticmethod
    def _check_numeric(data: pd.DataFrame, feature_cols: List[str]) -> None:
        non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric:
            raise ValueError(f"Selected features must be numeric, got non-numeric: {non_numeric}")

    @staticmethod
    def _order_steps(data: pd.DataFrame) -> pd.DataFrame:
        """
        Orders rows into sequences (by player, then season) and adds "step_idx", each row's 0-based position
        in its player's sequence.

        Raises:
            ValueError: If a player has two rows for the same season.
        """
        if data.duplicated([PLAYER_COL, SEASON_COL]).any():
            raise ValueError(f"More than one row for some ({PLAYER_COL}, {SEASON_COL})")

        steps = data.sort_values([PLAYER_COL, SEASON_COL]).reset_index(drop=True)
        steps["step_idx"] = steps.groupby(PLAYER_COL).cumcount()
        return steps

    @staticmethod
    def _rows_up_to_last_loss_step(steps: pd.DataFrame, loss_rows: pd.Series) -> pd.Series:
        """
        Returns:
            Boolean Series, True for every step at or before the last step (in its player's sequence) where
            loss_rows is True. False for every step of a player with no loss rows.
        """
        last_loss_step = steps["step_idx"].where(loss_rows).groupby(steps[PLAYER_COL]).transform("max")
        return steps["step_idx"] <= last_loss_step

    def _build_sequence_set(
        self,
        steps: pd.DataFrame,
        feature_cols: List[str],
        loss_rows: Optional[pd.Series] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> SequenceSet:
        """
        Packs ordered steps into padded tensors.

        Args:
            steps: Output of _order_steps (possibly with trailing steps removed), sorted by player then step.
            feature_cols: Columns to use as features.
            loss_rows: Boolean Series aligned with steps, True for steps that count toward the loss. Pass for
                training splits; also fills y.
            sample_weights: Series aligned with steps. Zeroed outside loss_rows.
        """
        player_ids = steps[PLAYER_COL].to_numpy()
        seq_idx = pd.factorize(player_ids)[0]  # steps are sorted by player, so this is sequence order
        step_idx = steps["step_idx"].to_numpy()

        num_sequences = int(seq_idx.max()) + 1 if len(steps) else 0
        max_len = int(step_idx.max()) + 1 if len(steps) else 0

        X = np.full((num_sequences, max_len, len(feature_cols)), np.nan, dtype=np.float32)
        X[seq_idx, step_idx] = steps[feature_cols].to_numpy(dtype=np.float32)
        lengths = np.bincount(seq_idx, minlength=num_sequences)

        identity = steps[self.identity_cols].copy()
        identity.insert(0, "step_idx", step_idx)
        identity.insert(0, "seq_idx", seq_idx)

        result = SequenceSet(
            X=torch.from_numpy(X),
            lengths=torch.from_numpy(lengths.astype(np.int64)),
            identity=identity.reset_index(drop=True),
            feature_columns=list(feature_cols),
        )

        if loss_rows is not None:
            loss_mask = np.zeros((num_sequences, max_len), dtype=bool)
            loss_mask[seq_idx, step_idx] = loss_rows.to_numpy()

            y = np.zeros((num_sequences, max_len), dtype=np.float32)
            y[seq_idx, step_idx] = np.where(loss_rows, steps[TARGET_COL], 0.0)

            result.y = torch.from_numpy(y)
            result.loss_mask = torch.from_numpy(loss_mask)

        if sample_weights is not None:
            weights = np.zeros((num_sequences, max_len), dtype=np.float32)
            weights[seq_idx, step_idx] = np.where(loss_rows, sample_weights, 0.0)
            result.sample_weight = torch.from_numpy(weights)

        return result
