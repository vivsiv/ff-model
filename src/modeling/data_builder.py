import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Self

import pandas as pd
import yaml
from nflreadpy import get_current_season

from src.processing.column_registry import get_identity_columns
from src.processing.gold import TARGET_COL

logger = logging.getLogger(__name__)

# Preserve the artifact name used for data-preparation config discovery.
DATA_BUILDER_CONFIG_ARTIFACT_PATH = "data_prep_config.json"

TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
TEST_SPLIT = "test"
PREDICTION_SPLIT = "prediction"


@dataclass
class SplitYears:
    """Prediction cutoff and season counts for chronological splits."""
    eval_data_years: int = 1
    test_data_years: int = 1
    num_training_seasons: Optional[int] = None
    prediction_season: int = field(default_factory=lambda: get_current_season(roster=True))

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_season, int) or isinstance(self.prediction_season, bool):
            raise ValueError("prediction_season must be an integer year")

    @classmethod
    def from_config(cls, split_config: dict[str, Any]) -> "SplitYears":
        """Read the split config, defaulting omitted fields."""
        return cls(
            eval_data_years=split_config.get("eval_data_years", 1),
            test_data_years=split_config.get("test_data_years", 1),
            num_training_seasons=split_config.get("num_training_seasons"),
            prediction_season=split_config.get("prediction_season", get_current_season(roster=True)),
        )

    def to_masks(self, target_season: pd.Series) -> dict[str, pd.Series]:
        """Assign rows chronologically by target_season.

        Only target seasons strictly before prediction_season are eligible.
        The most recent eligible test_data_years seasons form the test split, followed by
        eval_data_years seasons for evaluation. Older seasons form training,
        optionally limited to the most recent num_training_seasons.

        Returns:
            Boolean row masks keyed by "train", "eval", "test", and "prediction",
            aligned with target_season. Prediction rows target prediction_season;
            missing and later seasons are in no split. Prediction rows are never scored.

        Raises:
            ValueError: If requested season counts exceed the distinct seasons available.
        """
        is_historical = target_season < self.prediction_season
        is_prediction = target_season == self.prediction_season
        historical_seasons = target_season[is_historical]
        if self.num_training_seasons is not None:
            total_seasons = historical_seasons.nunique()
            requested_seasons = self.num_training_seasons + self.eval_data_years + self.test_data_years
            if requested_seasons > total_seasons:
                raise ValueError(
                    f"num_training_seasons ({self.num_training_seasons}) + eval_data_years "
                    f"({self.eval_data_years}) + test_data_years ({self.test_data_years}) = "
                    f"{requested_seasons}, which exceeds the {total_seasons} distinct "
                    "season(s) available in the training set"
                )

        max_target_season = historical_seasons.max()
        test_cutoff_season = max_target_season - self.test_data_years + 1
        eval_cutoff_season = test_cutoff_season - self.eval_data_years

        is_test = is_historical & (target_season >= test_cutoff_season)
        is_eval = is_historical & (target_season >= eval_cutoff_season) & ~is_test

        if self.num_training_seasons is None:
            is_train = is_historical & (target_season < eval_cutoff_season)
        else:
            train_cutoff_season = eval_cutoff_season - self.num_training_seasons
            is_train = is_historical & (target_season >= train_cutoff_season) & (target_season < eval_cutoff_season)

        return {
            TRAIN_SPLIT: is_train, EVAL_SPLIT: is_eval, TEST_SPLIT: is_test,
            PREDICTION_SPLIT: is_prediction,
        }


# TODO: introduce a typed model-data config with centralized defaults, validation,
# and YAML/JSON round-trip serialization; preserve existing configs and MLflow artifacts.
class ModelDataBuilder(ABC):
    """Shared configuration and row-level helpers for tabular and sequence data builders.

    Subclasses build all four splits through _build_data_splits, including prediction.

    Config schema (see configs/*.yaml for real examples):
        target: <str> e.g. "ppr_points_per_game", "fantasy_points_ppr"
        positions: optional -- list[str] of "position" values (e.g. ["RB"], ["WR", "TE"]) to
            restrict the data to. Omit entirely (or leave empty) to keep every position.
        split:
            prediction_season: <int, default current roster year> -- gold snapshot
                year and scoring cutoff. Resolved into the saved split config so
                trained models keep loading the same year after rollover.
            eval_data_years: <int, default 1>
            test_data_years: <int, default 1>
            num_training_seasons: <int, default None -- keep every remaining older season>
        sample_weights: optional -- How to weight an individual training target.
            global_weight: <float, default 1.0> -- uniform weight when buckets are omitted/empty.
            buckets: list of {min: <float>, weight: <float>}. Apply the weight for
                bucket_n.min <= target < bucket_{n+1}.min (or inf). Rows below the lowest
                bucket's min get zero weight; include a min: 0 bucket to weight all rows.
        features: optional -- Specify features to include or exclude from the gold table.
            mode: include | exclude -- columns is an allow-list or deny-list, respectively.
            columns: <list[str]> -- exact names or wildcard patterns. "*" at one end matches
                prefixes/suffixes; at both ends it matches substrings.
            If omitted, use all non-target/non-identity columns, in their original order.
    """

    def __init__(self, data_dir: str, config: dict[str, Any]):
        """Configure the data builder without loading data from data_dir/gold."""
        self.gold_data_dir = os.path.join(data_dir, "gold")
        self.config = dict(config)
        self.target = config["target"]
        split_config = dict(config.get("split", {}))
        self.split_years = SplitYears.from_config(split_config)
        # TODO: should go away with config class.
        self.config["split"] = {**split_config, "prediction_season": self.split_years.prediction_season}
        self.positions = config.get("positions") or None
        # TODO: why does target_season need to be added here? Why cant it be in the registry?
        self.identity_cols = get_identity_columns("nflverse", "player_stats") + ["target_season"]
        self._data_splits: Optional[dict[str, Any]] = None

    @classmethod
    def from_config_file(cls, data_dir: str, config_path: str) -> Self:
        """Build the calling subclass from a YAML config matching the shared schema."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(data_dir=data_dir, config=config)

    def get_data_splits(self, *, recompute: bool = False) -> dict[str, Any]:
        """Lazily build splits and return the same cached objects on subsequent calls.

        Use recompute=True to reload source data and replace the cache after a
        successful build.
        """
        if self._data_splits is None or recompute:
            self._data_splits = self._build_data_splits()
        return self._data_splits

    @abstractmethod
    def _build_data_splits(self) -> dict[str, Any]:
        """Build fresh train/eval/test/prediction datasets in model-specific format."""
        raise NotImplementedError

    @staticmethod
    def _filter_positions(data: pd.DataFrame, positions: List[str]) -> pd.DataFrame:
        """Restrict rows to the configured positions, preserving their index."""
        filtered = data[data["position"].isin(positions)]
        logger.info(f"Filtered to positions {positions}: {len(filtered)} rows (from {len(data)})")
        return filtered

    @staticmethod
    def _matches_any(col: str, patterns: List[str]) -> bool:
        """Match exact names, prefix/suffix wildcards, or substring wildcards."""
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
        """Exclude identity/target columns, then apply configured feature rules."""
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
        """Return index-aligned bucket weights, or a uniform global weight (default 1)."""
        weights_config = self.config.get("sample_weights", {})
        buckets = sorted(weights_config.get("buckets", []), key=lambda bucket: bucket["min"])

        if not buckets:
            global_weight = weights_config.get("global_weight", 1.0)
            return pd.Series(global_weight, index=target_vals.index, dtype=float)

        weights = pd.Series(0.0, index=target_vals.index, dtype=float)
        for bucket in buckets:
            weights[target_vals >= bucket["min"]] = bucket["weight"]

        return weights
