import os
import logging
from typing import List

import pandas as pd

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

        baseline_columns = ["position", "season"]
        for stat in stat_columns:
            baseline_col = f"{stat}_positional_baseline"
            season_position_means[baseline_col] = (
                season_position_means
                .groupby("position")[stat]
                .transform(lambda x: x.rolling(window=window_years, min_periods=1).mean())
            )
            baseline_columns.append(baseline_col)

        return season_position_means[baseline_columns]

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

        df["years_played"] = grouped.cumcount() + 1

        for stat in stat_columns:
            df[f"{stat}_career_avg"] = grouped[stat].transform(lambda x: x.expanding().mean())
            df[f"{stat}_career_std"] = grouped[stat].transform(lambda x: x.expanding().std()).fillna(0)
            df[f"{stat}_career_max"] = grouped[stat].transform(lambda x: x.expanding().max())
            df[f"{stat}_career_min"] = grouped[stat].transform(lambda x: x.expanding().min())
            df[f"{stat}_trend"] = df[stat] - df[f"{stat}_career_avg"]

        df = df.merge(positional_baseline_df, on=["position", "season"], how="left")

        shrinkage_weight = df["years_played"] / (df["years_played"] + shrinkage_k)
        for stat in stat_columns:
            df[f"{stat}_shrunk_avg"] = (
                shrinkage_weight * df[f"{stat}_career_avg"]
                + (1 - shrinkage_weight) * df[f"{stat}_positional_baseline"]
            )

        return df
