import os
import logging
from typing import Any, List

import yaml
import pandas as pd

from src.processing.column_registry import get_identity_columns
from src.processing.gold import TARGET_COL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tabular_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
        sample_weights: optional -- uniform weight 1.0 for every training row if omitted
            entirely (or {} with no "global_weight"/"buckets").
            global_weight: <float, default 1.0>  -- used when "buckets" is empty/omitted, as
                a single uniform weight for every row.
            buckets: list of {min: <float>, weight: <float>}. The weight whose target bucket
                matches bucket_n.min <= target < bucket_{n+1}.min (or inf) is applied to the
                row.
        features:
            mode: include | exclude  -- mutually exclusive: either "columns" is an allow-list
                (only these survive) or a deny-list (everything but these survives).
            columns: <list[str]>  -- entries ending in "*" are treated as prefixes.
            (omit the whole "features" section for "keep every non-identity/target column")
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

        self.training_data = self._load_data()
        if self.positions:
            self.training_data = self._filter_positions(self.training_data, self.positions)

        self.identity_cols = get_identity_columns("nflverse", "player_stats") + ["target_season"]
        self.feature_cols = self._resolve_feature_columns()

        self.identity_df = self.training_data[self.identity_cols]
        self.features_df = self.training_data[self.feature_cols]
        self.target_df = self.training_data[TARGET_COL]
        self.sample_weights = self._compute_sample_weights(self.target_df)

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

    def _load_data(self) -> pd.DataFrame:
        filename = f"{self.target}__training_set.csv"
        data = pd.read_csv(os.path.join(self.gold_data_dir, filename))
        logger.info(f"Loaded data: {len(data)} rows")

        return data

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
        """Returns True if col matches any entry in patterns. The "*" wildcard can
        be used to denote prefixes or suffixes that should match; all other
        entries must match col exactly."""
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

    def _resolve_feature_columns(self) -> List[str]:
        """
        Collects the set of relevant feature columns by:
            - excluding identity and target columns.
            - applying feature exclusion/inclusion rules in config.

        Returns:
            List of feature column names, in their original column order.
        """
        features_config = self.config.get("features", {})
        mode = features_config.get("mode", "")
        patterns = features_config.get("columns", [])

        candidate_cols = [
            col for col in self.training_data.columns if col not in self.identity_cols + [TARGET_COL]
        ]

        if mode == "include":
            return [col for col in candidate_cols if self._matches_any(col, patterns)]
        elif mode == "exclude":
            return [col for col in candidate_cols if not self._matches_any(col, patterns)]
        else:
            return candidate_cols

    def _compute_sample_weights(self, target_vals: pd.Series) -> pd.Series:
        """
        Computes a per-row weight from config["sample_weights"]. When buckets of weights
        are provided the weight bucket_n.min <= target < bucket_n+1.min is applied to each row,
        otherwise the global_weight (default 1) is applied to each row.

        Args:
            target_vals: Target values to compute weights for.

        Returns:
            Series of weights aligned with target, index-for-index.
        """
        weights_config = self.config.get("sample_weights", {"global_weight": 1.0})
        buckets = sorted(weights_config.get("buckets", []), key=lambda bucket: bucket["min"])

        if not buckets:
            global_weight = weights_config.get("global_weight", 1.0)
            return pd.Series(global_weight, index=target_vals.index, dtype=float)

        weights = pd.Series(buckets[0]["weight"], index=target_vals.index, dtype=float)
        for bucket in buckets:
            weights[target_vals >= bucket["min"]] = bucket["weight"]

        return weights

    def split(self) -> dict[str, pd.DataFrame]:
        """
        Splits training data chronologically by target_season into train/eval/test sets based
        on the config's "split" field.

        The most recent test_data_years worth of seasons become the test set. The most recent
        eval_data_years worth of seasons not already claimed by the test set become the eval
        set. The remaining seasons are training data, the number of seasons used for training
        can be limited with num_training_seasons.

        Returns:
            dict with X_train/X_eval/X_test, y_train/y_eval/y_test,
            identity_train/identity_eval/identity_test, sample_weight_train

        Raises:
            ValueError: If num_training_seasons + eval_data_years + test_data_years exceeds
                the total number of distinct seasons in the training set.
        """
        split_config = self.config.get("split", {})
        eval_data_years = split_config.get("eval_data_years", 1)
        test_data_years = split_config.get("test_data_years", 1)
        num_training_seasons = split_config.get("num_training_seasons")

        target_season = self.training_data["target_season"]
        total_seasons = target_season.nunique()

        if num_training_seasons is not None:
            requested_seasons = num_training_seasons + eval_data_years + test_data_years
            if requested_seasons > total_seasons:
                raise ValueError(
                    f"num_training_seasons ({num_training_seasons}) + eval_data_years "
                    f"({eval_data_years}) + test_data_years ({test_data_years}) = "
                    f"{requested_seasons}, which exceeds the {total_seasons} distinct "
                    "season(s) available in the training set"
                )

        max_target_season = target_season.max()
        test_cutoff_season = max_target_season - test_data_years + 1
        eval_cutoff_season = test_cutoff_season - eval_data_years

        is_test = target_season >= test_cutoff_season
        is_eval = (target_season >= eval_cutoff_season) & ~is_test

        if num_training_seasons is None:
            is_train = ~is_eval & ~is_test
        else:
            train_cutoff_season = eval_cutoff_season - num_training_seasons
            is_train = (target_season >= train_cutoff_season) & ~is_eval & ~is_test

        return {
            "X_train": self.features_df[is_train],
            "X_eval": self.features_df[is_eval],
            "X_test": self.features_df[is_test],
            "y_train": self.target_df[is_train],
            "y_eval": self.target_df[is_eval],
            "y_test": self.target_df[is_test],
            "identity_train": self.identity_df[is_train],
            "identity_eval": self.identity_df[is_eval],
            "identity_test": self.identity_df[is_test],
            "sample_weight_train": self.sample_weights[is_train],
        }
