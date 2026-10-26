import os
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import yaml
import pandas as pd

from src.processing.column_registry import get_identity_columns
from src.processing.gold import TARGET_COL, load_prediction_set, load_training_set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tabular_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_PREP_CONFIG_ARTIFACT_PATH = "data_prep_config.json"

# TODO: lets have generic data prep constants live in a utilities file while tabular and sequence get moved to their own subdirs.
@dataclass
class SplitMasks:
    """Boolean row masks assigning each row to the train, eval or test split, aligned with the
    target_season Series they were built from (see SplitYears.to_masks). A row whose
    target_season is NaN is in no split."""
    is_train: pd.Series
    is_eval: pd.Series
    is_test: pd.Series


@dataclass
class SplitYears:
    """How many seasons go to each of train/eval/test -- see TabularModelDataPrep's config
    schema ("split") for field meanings. Turned into row masks by to_masks()."""
    eval_data_years: int = 1
    test_data_years: int = 1
    num_training_seasons: Optional[int] = None

    @classmethod
    def from_config(cls, split_config: dict[str, Any]) -> "SplitYears":
        """
        Args:
            split_config: The config's "split" section (may be empty), see TabularModelDataPrep's
                class docstring for the schema.

        Returns:
            SplitYears with the documented defaults for any omitted field.
        """
        return cls(
            eval_data_years=split_config.get("eval_data_years", 1),
            test_data_years=split_config.get("test_data_years", 1),
            num_training_seasons=split_config.get("num_training_seasons"),
        )

    def to_masks(self, target_season: pd.Series) -> SplitMasks:
        """
        Validates these years against the data and chronologically assigns each row to
        train/eval/test by target_season.

        The most recent test_data_years worth of seasons become the test set.
        The most recent eval_data_years worth of seasons not already claimed by
        the test set become the eval set. The remaining seasons are training data, the
        number of seasons used for training can be limited with num_training_seasons.

        Args:
            target_season: target_season column of the data being split.

        Returns:
            SplitMasks aligned with target_season.

        Raises:
            ValueError: If num_training_seasons + eval_data_years + test_data_years exceeds the
                total number of distinct seasons in target_season.
        """
        if self.num_training_seasons is not None:
            total_seasons = target_season.nunique()
            requested_seasons = self.num_training_seasons + self.eval_data_years + self.test_data_years
            if requested_seasons > total_seasons:
                raise ValueError(
                    f"num_training_seasons ({self.num_training_seasons}) + eval_data_years "
                    f"({self.eval_data_years}) + test_data_years ({self.test_data_years}) = "
                    f"{requested_seasons}, which exceeds the {total_seasons} distinct "
                    "season(s) available in the training set"
                )

        max_target_season = target_season.max()
        test_cutoff_season = max_target_season - self.test_data_years + 1
        eval_cutoff_season = test_cutoff_season - self.eval_data_years

        is_test = target_season >= test_cutoff_season
        is_eval = (target_season >= eval_cutoff_season) & ~is_test

        if self.num_training_seasons is None:
            is_train = target_season < eval_cutoff_season
        else:
            train_cutoff_season = eval_cutoff_season - self.num_training_seasons
            is_train = (target_season >= train_cutoff_season) & (target_season < eval_cutoff_season)

        return SplitMasks(is_train=is_train, is_eval=is_eval, is_test=is_test)


class TabularModelDataPrep:
    """
    Loads a target's gold training table and prepares it for modeling, by:
    - splitting the data into train/eval/test sets
    - selecting the features for modelling
    - providing sample weights for the training rows.


    Config schema (see configs/*.yaml for real examples):
        target: <str> e.g. "ppr_points_per_game", "fantasy_points_ppr"
        positions: optional -- list[str] of "position" values (e.g. ["RB"], ["WR", "TE"]) to
            restrict the training data to. Omit entirely (or leave empty) to keep every
            position, i.e. train the general model.
        split:
            eval_data_years: <int, default 1>
            test_data_years: <int, default 1>
            num_training_seasons: <int, default None -- keep every remaining older season>
        sample_weights: optional -- How to weight an individual row in the training set.
            global_weight: <float, default 1.0>  -- used when "buckets" is empty/omitted, as
                a single uniform weight for every row
            buckets: list of {min: <float>, weight: <float>}. The weight whose target bucket
                matches bucket_n.min <= target < bucket_{n+1}.min (or inf) is applied to the
                row. Rows below the lowest bucket's own min get a weight of 0 -- include a
                min: 0 bucket if every row should get some non-zero weight.
            If no value is provided a global_weight of 1.0 is applied to all rows.
        features: optional -- Specify features to use or not use from the gold table.
            mode: include | exclude  -- mutually exclusive: either "columns" is an allow-list
                (only these survive) or a deny-list (everything but these survives).
            columns: <list[str]>  -- The list of features to include or exclude,
                The "*" wildcard can be used to denote prefixes or suffixes that should match.
            If no value is provided all non target/identity columns are used as features.
    """

    def __init__(self, data_dir: str, config: dict[str, Any]):
        """
        Args:
            data_dir: Parent directory for the gold layer (gold_dir/{target}__training_set.csv
                is loaded from data_dir/gold)
            config: See class docstring for schema.
        """
        self.data_dir = data_dir
        self.gold_data_dir = os.path.join(data_dir, "gold")
        self.config = config
        self.target = config["target"]
        self.positions = config.get("positions") or None
        self.identity_cols = get_identity_columns("nflverse", "player_stats") + ["target_season"]

    @classmethod
    def from_config_file(cls, data_dir: str, config_path: str) -> "TabularModelDataPrep":
        """
        Args:
            data_dir: Parent directory for the gold layer.
            config_path: Path to a YAML config file matching the schema in the class docstring.

        Returns:
            A TabularModelDataPrep built from the loaded config.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(data_dir=data_dir, config=config)

    # TODO: this isn't just training data its split data.
    # TODO: while lazy this shouldn't recompute everytime its called on a dataprep, once its called it should cache.
    def prep_training_data(self) -> dict[str, pd.DataFrame]:
        """
        Loads the gold training set and prepares it for modeling.

        Returns a dict with:
            - X_{train/eval/test}
            - y_{train/eval/test}
            - identity_{train,eval,test}
            - sample_weight_train

        Raises:
            ValueError: If split config's num_training_seasons + eval_data_years +
                test_data_years exceeds the total number of distinct seasons available.
        """
        self.training_data = load_training_set(self.gold_data_dir, self.target)

        if self.positions:
            self.training_data = self._filter_positions(self.training_data, self.positions)

        feature_cols = self._resolve_feature_columns(self.training_data.columns)
        target_df = self.training_data[TARGET_COL]
        target_season = self.training_data["target_season"]

        return self.split(
            split_years=SplitYears.from_config(self.config.get("split", {})),
            target_season=target_season,
            features_df=self.training_data[feature_cols],
            target_df=target_df,
            identity_df=self.training_data[self.identity_cols],
            sample_weights=self._compute_sample_weights(target_df),
        )

    # TODO: can this just be done in prep_training_data?
    def prep_prediction_data(self) -> dict[str, pd.DataFrame]:
        """
        Loads the prediction set (the live, not-yet-played season's rows) 
        and prepares it the same way the training set is prepared.

        Returns:
            dict with "identity" and "features" DataFrames, row-aligned.
        """
        data = load_prediction_set(self.gold_data_dir, self.target)

        if self.positions:
            data = self._filter_positions(data, self.positions)

        feature_cols = self._resolve_feature_columns(data.columns)

        return {
            "identity": data[self.identity_cols],
            "features": data[feature_cols],
        }

    @staticmethod
    def _filter_positions(data: pd.DataFrame, positions: List[str]) -> pd.DataFrame:
        """
        Restricts data to rows whose "position" column is one of positions.

        Args:
            data: Training data to filter.
            positions: Position values to keep, e.g. ["RB"] or ["WR", "TE"].

        Returns:
            Filtered copy of data.
        """
        filtered = data[data["position"].isin(positions)]
        logger.info(f"Filtered to positions {positions}: {len(filtered)} rows (from {len(data)})")

        return filtered

    @staticmethod
    def _matches_any(col: str, patterns: List[str]) -> bool:
        """Returns True if col matches any entry in patterns. "*" on both ends matches col as
        a substring, on one end matches col as a prefix/suffix; entries with no "*" must match
        col exactly."""

        for pattern in patterns:
            if pattern.startswith("*") and pattern.endswith("*"):
                if pattern[1:-1] in col:
                    return True
            elif pattern.endswith("*"):
                if col.startswith(pattern[:-1]):
                    return True
            elif pattern.startswith("*"):
                if col.endswith(pattern[1:]):
                    return True
            elif col == pattern:
                return True

        return False

    def _resolve_feature_columns(self, columns: List[str]) -> List[str]:
        """
        Collects the set of relevant feature columns from the full column list by:
            - excluding identity and target columns.
            - applying feature exclusion/inclusion rules in config.

        Args:
            columns: Training set or prediction set's own columns, to resolve candidates from.

        Returns:
            List of feature column names, in their original column order.
        """
        features_config = self.config.get("features", {})
        mode = features_config.get("mode", "")
        patterns = features_config.get("columns", [])

        candidate_cols = [col for col in columns if col not in [*self.identity_cols, TARGET_COL]]

        if mode == "include":
            return [col for col in candidate_cols if self._matches_any(col, patterns)]
        elif mode == "exclude":
            return [col for col in candidate_cols if not self._matches_any(col, patterns)]
        else:
            return candidate_cols

    def _compute_sample_weights(self, target_vals: pd.Series) -> pd.Series:
        """
        Computes a per-row weight from config["sample_weights"].

        When buckets of weights are provided the weight bucket_n.min <= target < bucket_n+1.min is applied to each row,
        otherwise the global_weight (default 1) is applied to each row.

        Rows below the lowest bucket's own min get a weight of 0.

        Args:
            target_vals: Target values to compute weights for.

        Returns:
            Series of weights aligned with target, index-for-index.
        """
        weights_config = self.config.get("sample_weights", {})
        buckets = sorted(weights_config.get("buckets", []), key=lambda bucket: bucket["min"])

        if not buckets:
            global_weight = weights_config.get("global_weight", 1.0)
            return pd.Series(global_weight, index=target_vals.index, dtype=float)

        weights = pd.Series(0.0, index=target_vals.index, dtype=float)
        for bucket in buckets:
            weights[target_vals >= bucket["min"]] = bucket["weight"]

        return weights

    @staticmethod
    def split(
        split_years: SplitYears,
        target_season: pd.Series,
        features_df: pd.DataFrame,
        target_df: pd.Series,
        identity_df: pd.DataFrame,
        sample_weights: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        """
        Splits the given data chronologically by target_season into train/eval/test sets
        based on split_years (see SplitYears.to_masks for how years map to seasons).

        Args:
            split_years: How many seasons to allocate to each of train/eval/test.
            target_season: target_season column.
            features_df: Resolved feature columns to split.
            target_df: Target column to split.
            identity_df: Identity columns to split.
            sample_weights: Per-row sample weights to split.

        Returns:
            dict with X_train/X_eval/X_test, y_train/y_eval/y_test,
            identity_train/identity_eval/identity_test, sample_weight_train

        Raises:
            ValueError: If split_years asks for more seasons than target_season has.
        """
        masks = split_years.to_masks(target_season)
        is_train, is_eval, is_test = masks.is_train, masks.is_eval, masks.is_test

        return {
            "X_train": features_df[is_train],
            "X_eval": features_df[is_eval],
            "X_test": features_df[is_test],
            "y_train": target_df[is_train],
            "y_eval": target_df[is_eval],
            "y_test": target_df[is_test],
            "identity_train": identity_df[is_train],
            "identity_eval": identity_df[is_eval],
            "identity_test": identity_df[is_test],
            "sample_weight_train": sample_weights[is_train],
        }
