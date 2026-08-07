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

    def __init__(self, data_dir: str):
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

    def load_team_features(self) -> pd.DataFrame:
        """
        Loads the `team_stats` silver table, filtered to the registered identity/stat columns.

        Unlike load_player_features, no career-to-date aggregates are computed here --
        `_join_team_features` only ever needs a team's own single-season stats for the
        season immediately before a target season, not that team's career history.

        Returns:
            One row per team-season, with only the registered identity/stat columns kept.
            Drops the league-total row nflverse includes with a missing team code.
        """
        identity_columns = get_identity_columns("nflverse", "team_stats")
        stat_columns = get_stat_columns("nflverse", "team_stats")

        team_df = pd.read_csv(os.path.join(self.silver_dir, "team_stats.csv"), low_memory=False)
        team_df = team_df[team_df["team"].notna()]
        return team_df[identity_columns + stat_columns]

    def _join_team_features(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
        team_features_df: pd.DataFrame,
        team_stat_columns: List[str],
        player_grouping_col: str = "player_id",
        team_col: str = "recent_team",
    ) -> pd.DataFrame:
        """
        Adds team-context features to each row of df, capturing not just the level of a
        player's new team but the shift in team quality when a player changes teams. Since
        the player is no longer on their origin team by target_season, only the destination
        team's level and the shift relative to origin are kept as features -- the origin
        team's raw stats are an intermediate used only to compute the shift, not a feature
        themselves.

        Each row of df (output of _join_with_target or _build_prediction_rows) already has
        a "season" (the feature/origin season, e.g. year N-1) and a team_col value (the team
        the player produced that season's stats with, i.e. their origin team). This adds:
          - "team_{stat}": the player's *destination* team's stat -- the actual team_col
            value in target_season (year N), looked up from `features_df`, falling back to
            the origin team when target_season hasn't happened yet or has no matching row
            (e.g. prediction rows, where a player's future team isn't knowable from this
            data) -- i.e. assumes no team change absent better information.
          - "team_shift_{stat}" = destination team's stat - origin team's stat (zero for
            players who didn't change teams).
          - Both the destination and origin lookups use `team_features_df` for the *origin*
            season only, never target_season -- a team's performance in the season being
            predicted doesn't exist yet at real prediction time, so using it would be a
            look-ahead leak.

        The team_col value used to resolve the destination team, and the resolved
        destination team itself, are join-only intermediates and are not present in the
        returned columns.

        Args:
            df: Output of _join_with_target or _build_prediction_rows (must have
                player_grouping_col, "season", "target_season", and team_col columns)
            features_df: Output of load_player_features (one row per player-season, used to
                look up each player's actual team in target_season)
            team_features_df: Output of load_team_features (one row per team-season)
            team_stat_columns: Team stat columns to join in and compute a shift for
            player_grouping_col: Column identifying a unique player (default: "player_id")
            team_col: Column identifying a player's team on a given row (default:
                "recent_team")

        Returns:
            df with "team_{stat}"/"team_shift_{stat}" columns added per stat in
            team_stat_columns
        """
        df = df.copy()
        origin_team = df[team_col]

        destination_lookup = (
            features_df[[player_grouping_col, "season", team_col]]
            .rename(columns={"season": "target_season", team_col: "destination_team"})
        )
        df = df.merge(destination_lookup, on=[player_grouping_col, "target_season"], how="left")
        df["destination_team"] = df["destination_team"].fillna(origin_team)

        team_lookup = team_features_df[["team", "season"] + team_stat_columns]

        origin_stats = team_lookup.rename(
            columns={"team": team_col, **{stat: f"_origin_team_{stat}" for stat in team_stat_columns}}
        )
        df = df.merge(origin_stats, on=[team_col, "season"], how="left")

        destination_stats = team_lookup.rename(
            columns={"team": "destination_team", **{stat: f"team_{stat}" for stat in team_stat_columns}}
        )
        df = df.merge(destination_stats, on=["destination_team", "season"], how="left")

        shift_columns = {
            f"team_shift_{stat}": df[f"team_{stat}"] - df[f"_origin_team_{stat}"]
            for stat in team_stat_columns
        }
        df = pd.concat([df, pd.DataFrame(shift_columns, index=df.index)], axis=1)

        drop_columns = ["destination_team"] + [f"_origin_team_{stat}" for stat in team_stat_columns]
        return df.drop(columns=drop_columns)

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

    def build_training_set(
        self,
        features_df: pd.DataFrame,
        team_features_df: pd.DataFrame,
        target_col: str = "fantasy_points_ppr",
    ) -> pd.DataFrame:
        """
        Builds the training set from the features dataframe and specified target
        and saves it to the gold layer.

        Args:
            features_df: Output of load_player_features
            team_features_df: Output of load_team_features
            target_col: Column to predict; must be a registered target for player_stats
                (default: "fantasy_points_ppr")

        Returns:
            DataFrame of the training set, also saved to gold_dir/{target_col}__training_set.csv
        """
        targets = get_targets("nflverse", "player_stats")
        assert target_col in targets, f"{target_col} is not a registered target for player_stats: {targets}"

        team_stat_columns = get_stat_columns("nflverse", "team_stats")

        training_df = self._join_with_target(features_df, target_col)
        training_df = self._join_team_features(training_df, features_df, team_features_df, team_stat_columns)

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

    def build_prediction_set(
        self,
        features_df: pd.DataFrame,
        team_features_df: pd.DataFrame,
        target_col: str,
        prediction_season: int,
    ) -> pd.DataFrame:
        """
        Builds the gold prediction set from career-feature rows: each player's most recent
        season reframed as a row for predicting prediction_season, with target left blank
        (NaN).

        Args:
            features_df: Output of load_player_features
            team_features_df: Output of load_team_features
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

        team_stat_columns = get_stat_columns("nflverse", "team_stats")

        prediction_df = self._build_prediction_rows(features_df, prediction_season)
        prediction_df = self._join_team_features(prediction_df, features_df, team_features_df, team_stat_columns)

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
        default="data",
        help="Parent directory for the silver and gold layers, relative to the repo root (default: data)"
    )

    args = parser.parse_args()

    builder = TrainingSetBuilder(data_dir=args.data_dir)

    features_df = builder.load_player_features()
    team_features_df = builder.load_team_features()
    builder.build_training_set(features_df, team_features_df, target_col=args.target_col)
    builder.build_prediction_set(
        features_df, team_features_df, target_col=args.target_col, prediction_season=args.prediction_season
    )


if __name__ == "__main__":
    main()
