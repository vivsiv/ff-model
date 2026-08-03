import os
import logging
from typing import List

import pandas as pd

from src.processing.column_registry import get_included_columns, get_targets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("gold_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingSetBuilder:
    """Builds gold layer training/live data from nflverse silver layer data."""

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the builder.

        Args:
            data_dir: Parent directory for the silver and gold layers.
        """
        self.silver_dir = os.path.join(data_dir, "silver", "nflv")
        if not os.path.exists(self.silver_dir):
            raise FileNotFoundError(f"{self.silver_dir} not found")

        self.gold_dir = os.path.join(data_dir, "gold")
        os.makedirs(self.gold_dir, exist_ok=True)

    def _positional_baseline(
        self,
        df: pd.DataFrame,
        stat_columns: List[str],
        window_years: int = 5,
    ) -> pd.DataFrame:
        """
        Computes, for each (position, season), the last `window_years` league-wide
        average of each stat in consideration for modelling.

        Args:
            df: Dataframe of player stats by season, must contain "position" and "season" columns
            stat_columns: The stat columns to compute a positional baseline for
            window_years: Trailing window size in seasons (default: 5)

        Returns:
            DataFrame with one row per (position, season) and one
            "{stat}_positional_baseline" column per stat in stat_columns
        """
        season_position_means = (
            df.groupby(["position", "season"])[stat_columns]
            .mean()
            .reset_index()
            .sort_values(["position", "season"])
        )

        baseline_columns = {
            f"{stat}_positional_baseline": (
                season_position_means
                .groupby("position")[stat]
                .transform(lambda x: x.rolling(window=window_years, min_periods=1).mean())
            )
            for stat in stat_columns
        }

        return pd.concat(
            [season_position_means[["position", "season"]], pd.DataFrame(baseline_columns, index=season_position_means.index)],
            axis=1,
        )

    def _add_career_features(
        self,
        df: pd.DataFrame,
        positional_baseline_df: pd.DataFrame,
        stat_columns: List[str],
        player_grouping_col: str = "player_id",
        shrinkage_k: float = 3.0,
    ) -> pd.DataFrame:
        """
        For each player, sorted by season, computes expanding career average features for each
        stat in stat_columns. Features are inclusive of the current row's own season, 
        so a row here always answers "as of the end of season x, what does their career look like."

        Adds, per stat:
          - {stat}_career_avg / _career_std / _career_max / _career_min: expanding aggregates.
            career_std is 0 for a player's first season (undefined with 1 data point) rather
            than NaN.
          - {stat}_trend: this season's own value minus {stat}_career_avg (how far above/below
            their own career norm this season was)
          - {stat}_shrunk_avg: {stat}_career_avg blended toward the positional baseline,
            weighted by years_played, so a short career doesn't get treated as equally reliable
            as a long one: shrunk_avg = (n / (n + k)) * career_avg + (k / (n + k)) * baseline

        Also adds years_played: count of seasons with data up to and including this one.

        Args:
            df: Dataframe of Player stats by season, must contain grouping_col, "position", and "season"
            positional_baseline_df: Output of _positional_baseline, joined in by (position, season)
            stat_columns: The stat columns to compute career features for
            player_grouping_col: Column identifying a unique player (default: "player_id")
            shrinkage_k: Shrinkage strength constant — higher pulls harder toward the positional
                baseline for a given years_played (default: 3.0, a starting point to tune later)

        Returns:
            DataFrame with the career feature columns added
        """
        df = df.sort_values([player_grouping_col, "season"]).copy()
        grouped = df.groupby(player_grouping_col)

        new_columns = {"years_played": grouped.cumcount() + 1}
        for stat in stat_columns:
            career_avg = grouped[stat].transform(lambda x: x.expanding().mean())
            new_columns[f"{stat}_career_avg"] = career_avg
            new_columns[f"{stat}_career_std"] = grouped[stat].transform(lambda x: x.expanding().std()).fillna(0)
            new_columns[f"{stat}_career_max"] = grouped[stat].transform(lambda x: x.expanding().max())
            new_columns[f"{stat}_career_min"] = grouped[stat].transform(lambda x: x.expanding().min())
            new_columns[f"{stat}_trend"] = df[stat] - career_avg

        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        df = df.merge(positional_baseline_df, on=["position", "season"], how="left")

        shrinkage_weight = df["years_played"] / (df["years_played"] + shrinkage_k)
        shrunk_columns = {
            f"{stat}_shrunk_avg": (
                shrinkage_weight * df[f"{stat}_career_avg"]
                + (1 - shrinkage_weight) * df[f"{stat}_positional_baseline"]
            )
            for stat in stat_columns
        }

        return pd.concat([df, pd.DataFrame(shrunk_columns, index=df.index)], axis=1)

    def _join_with_target(
        self,
        features_df: pd.DataFrame,
        target_col: str,
        player_grouping_col: str = "player_id",
    ) -> pd.DataFrame:
        """
        Joins each player's season-N target value onto their most recent prior season's
        feature row — not necessarily season N-1. A player who missed one or more seasons
        (injury, out of the league) still gets matched to their last active season instead of
        being dropped. Only players with at least one prior season produce an output row.

        Args:
            features_df: Output of _add_career_features (one row per player-season, with that
                season's raw stats and career-to-date-through-that-season features)
            target_col: Column in features_df to use as the prediction target, e.g.
                "fantasy_points_ppr". 
            player_grouping_col: Column identifying a unique player (default: "player_id")

        Returns:
            DataFrame of feature rows (most recent season before target_season) with
            "target_season", "target", and "seasons_since_played" columns added
        """
        season_dtype = features_df["season"].dtype

        target_df = features_df[[player_grouping_col, "season", target_col]].copy()
        target_df = target_df.rename(columns={"season": "target_season", target_col: "target"})

        merged = pd.merge_asof(
            target_df.sort_values("target_season"),
            features_df.sort_values("season"),
            left_on="target_season",
            right_on="season",
            by=player_grouping_col,
            direction="backward",
            allow_exact_matches=False,
        )
        merged = merged[merged["season"].notna()].copy()
        merged["season"] = merged["season"].astype(season_dtype)
        merged["seasons_since_played"] = merged["target_season"] - merged["season"] - 1

        feature_columns = list(features_df.columns)
        return merged[feature_columns + ["target_season", "target", "seasons_since_played"]]

    def build_training_set(self, target_col: str = "fantasy_points_ppr") -> pd.DataFrame:
        """
        Builds the gold training set from silver player_stats: positional baseline -> career
        features -> joined with each player's next-season target. stat_columns comes from the
        column registry (column_registry.yaml), not hardcoded here.

        Does not yet include the team-shift join (origin/destination team, lagged to season
        N-1 -- see the plan) -- career features only for now.

        Args:
            target_col: Column to predict; must be a registered target for player_stats
                (default: "fantasy_points_ppr")

        Returns:
            DataFrame of the training set, also saved to gold_dir/training_set.csv
        """
        targets = get_targets("player_stats")
        assert target_col in targets, f"{target_col} is not a registered target for player_stats: {targets}"

        stat_columns = get_included_columns("player_stats")

        player_df = pd.read_csv(os.path.join(self.silver_dir, "player_stats.csv"), low_memory=False)

        baseline_df = self._positional_baseline(player_df, stat_columns)
        features_df = self._add_career_features(player_df, baseline_df, stat_columns)
        training_df = self._join_with_target(features_df, target_col)

        output_path = os.path.join(self.gold_dir, "training_set.csv")
        training_df.to_csv(output_path, index=False)
        logger.info(f"Saved training set to {output_path}")

        return training_df
