import os
import logging
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

from src.processing.column_registry import get_identity_columns, get_stat_columns, get_targets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("gold_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# The name every gold table uses for its target column.
TARGET_COL = "target"


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
        For each player-season combo computes the expanding career average/max/min/stddev features for each
        stat in stat_columns. Feature computations are inclusive of the current row's own season.

        Adds, per stat:
          - {stat}_career_avg / _career_std / _career_max / _career_min: expanding aggregates.
            career_std is 0 for a player's first season.
          - {stat}_trend: this season's own value minus {stat}_career_avg (how far above/below
            their own career norm this season was)
          - {stat}_shrunk_avg: {stat}_career_avg blended toward the positional baseline,
            weighted by years_played, so a short career doesn't get treated as equally reliable
            as a long one: shrunk_avg = (n / (n + k)) * career_avg + (k / (n + k)) * baseline
          - years_played: count of seasons with data up to and including this one.

        Args:
            df: Dataframe of player stats by season.
            positional_baseline_df: Output of _positional_baseline, joined in by (position, season)
            stat_columns: The stat columns to compute career features
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

    def load_player_features(self) -> pd.DataFrame:
        """
        Loads the `player_stats` silver tables and adds computed feature values to it. Both
        the construction of the training and prediction sets use the output.

        Returns:
            One row per player-season, with career-to-date features through that season
        """
        identity_columns = get_identity_columns("nflverse", "player_stats")
        stat_columns = get_stat_columns("nflverse", "player_stats")

        player_df = pd.read_csv(os.path.join(self.silver_dir, "player_stats.csv"), low_memory=False)
        player_df = player_df[identity_columns + stat_columns]

        baseline_df = self._positional_baseline(player_df, stat_columns)
        return self._add_career_features(player_df, baseline_df, stat_columns)

    def _join_with_target(
        self,
        features_df: pd.DataFrame,
        target_col: str,
        player_grouping_col: str = "player_id",
    ) -> pd.DataFrame:
        """
        Joins each player's season N target value onto their most recent prior season's
        feature row (usually season N-1). A player who missed season N-1 still gets matched
        to their last active season instead of being dropped.
        Only players with at least one prior season produce an output row.

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
        target_df = target_df.rename(columns={"season": "target_season", target_col: TARGET_COL})

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
        return merged[feature_columns + ["target_season", TARGET_COL, "seasons_since_played"]]

    def build_training_set(self, features_df: pd.DataFrame, target_col: str = "fantasy_points_ppr") -> pd.DataFrame:
        """
        Builds the training set from the features dataframe and specified target
        and saves it to the gold layer.

        Args:
            features_df: Output of load_player_features
            target_col: Column to predict; must be a registered target for player_stats
                (default: "fantasy_points_ppr")

        Returns:
            DataFrame of the training set, also saved to gold_dir/{target_col}__training_set.csv
        """
        targets = get_targets("nflverse", "player_stats")
        assert target_col in targets, f"{target_col} is not a registered target for player_stats: {targets}"

        training_df = self._join_with_target(features_df, target_col)

        output_path = os.path.join(self.gold_dir, f"{target_col}__training_set.csv")
        training_df.to_csv(output_path, index=False)
        logger.info(f"Saved training set to {output_path}")

        return training_df

    def _build_prediction_rows(
        self,
        features_df: pd.DataFrame,
        prediction_season: int,
        player_grouping_col: str = "player_id",
    ) -> pd.DataFrame:
        """
        Takes each player's most recent season from the features dataframe and reframes it as a row
        for predicting next season's target values.

        Includes every player with at least one season on record, final predictions need
        to filtering down to who's actually still active/rostered for prediction_season.

        Args:
            features_df: Output of load_player_features (one row per player-season)
            prediction_season: The season to build a prediction row for, e.g. 2026
            player_grouping_col: Column identifying a unique player (default: "player_id")

        Returns:
            DataFrame with one row per player, "target_season" set to prediction_season,
            "target" as NaN, and "seasons_since_played" computed the same way as
            _join_with_target
        """
        latest_df = (
            features_df.sort_values("season")
            .groupby(player_grouping_col, as_index=False)
            .tail(1)
            .copy()
        )
        latest_df["target_season"] = prediction_season
        latest_df["seasons_since_played"] = prediction_season - latest_df["season"] - 1
        latest_df[TARGET_COL] = np.nan

        return latest_df

    def build_prediction_set(self, features_df: pd.DataFrame, target_col: str, prediction_season: int) -> pd.DataFrame:
        """
        Builds the gold prediction set from career-feature rows: each player's most recent
        season reframed as a row for predicting prediction_season, with target left blank
        (NaN).

        Args:
            features_df: Output of load_player_features
            target_col: Column that will eventually be predicted; must be a registered target
                for player_stats (only used for naming the output file consistently with
                build_training_set -- the actual target values are blank)
            prediction_season: The season to build a prediction row for, e.g. 2026

        Returns:
            DataFrame of the prediction set, also saved to
            gold_dir/{target_col}__prediction_set.csv
        """
        targets = get_targets("nflverse", "player_stats")
        assert target_col in targets, f"{target_col} is not a registered target for player_stats: {targets}"

        prediction_df = self._build_prediction_rows(features_df, prediction_season)

        output_path = os.path.join(self.gold_dir, f"{target_col}__prediction_set.csv")
        prediction_df.to_csv(output_path, index=False)
        logger.info(f"Saved prediction set to {output_path}")

        return prediction_df


def main():
    parser = argparse.ArgumentParser(
        description="Builds gold training/prediction sets from nflverse silver layer data"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="fantasy_points_ppr",
        help="Column to predict; must be a registered target for player_stats (default: fantasy_points_ppr)"
    )
    parser.add_argument(
        "--prediction-season",
        type=int,
        default=2026,
        help="If provided, also builds a prediction set for this season (e.g. 2026)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Parent directory for the silver and gold layers (default: class default)"
    )

    args = parser.parse_args()

    kwargs = {"data_dir": args.data_dir} if args.data_dir is not None else {}
    builder = TrainingSetBuilder(**kwargs)

    features_df = builder.load_player_features()
    builder.build_training_set(features_df, target_col=args.target_col)
    builder.build_prediction_set(features_df, target_col=args.target_col, prediction_season=args.prediction_season)


if __name__ == "__main__":
    main()
