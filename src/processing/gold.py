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

TARGET_COL = "target"
ROUNDING_EXCLUDED_COLUMNS = [
    "season", "target_season", "seasons_since_played", "years_played", "games", "age", "draft_pick",
]


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

    def _round_significant_figures(
        self,
        df: pd.DataFrame,
        sig_figs: int = 4,
        exclude_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Rounds every numeric column to `sig_figs` significant figures.

        Args:
            df: DataFrame to round
            sig_figs: Number of significant figures to keep (default: 4)
            exclude_columns: Numeric columns to leave untouched.

        Returns:
            A copy of df with eligible numeric columns rounded to sig_figs significant
            figures. NaN/inf and non-numeric columns are left untouched.
        """
        df = df.copy()
        exclude_columns = set(exclude_columns or [])

        for col in df.select_dtypes(include=[np.number]).columns:
            if col in exclude_columns:
                continue

            values = df[col].to_numpy(dtype=float)
            roundable = np.isfinite(values) & (values != 0)

            magnitude = np.zeros_like(values)
            magnitude[roundable] = np.floor(np.log10(np.abs(values[roundable])))
            scale = np.power(10.0, sig_figs - 1 - magnitude)

            rounded = values.copy()
            rounded[roundable] = np.round(values[roundable] * scale[roundable]) / scale[roundable]
            df[col] = rounded

        return df

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

    def load_draft_features(self) -> pd.DataFrame:
        """
        Loads the `draft_picks` silver table, filtered to the registered identity/stat
        columns. The silver layer already guarantees exactly one row per player_id and no
        rows with a missing player_id.

        "season" (the draft year) is renamed to "draft_season" to avoid colliding with
        player_stats'/the training set's own "season"/"target_season" columns once joined
        in by _join_draft_features.

        Returns:
            One row per drafted player: "draft_season", "player_id", "draft_pick",
            "age_at_draft".
        """
        identity_columns = get_identity_columns("nflverse", "draft_picks")
        stat_columns = get_stat_columns("nflverse", "draft_picks")

        draft_df = pd.read_csv(os.path.join(self.silver_dir, "draft_picks.csv"), low_memory=False)
        draft_df = draft_df[identity_columns + stat_columns]
        return draft_df.rename(columns={"season": "draft_season"})

    def _join_draft_features(
        self,
        df: pd.DataFrame,
        draft_features_df: pd.DataFrame,
        player_grouping_col: str = "player_id",
    ) -> pd.DataFrame:
        """
        Adds draft-derived features to each row of df by player_grouping_col:
          - "draft_pick": static -- identical across every row for a given player, never
            career-averaged/shrunk the way player_stats' own stat columns are.
          - "age": age_at_draft + (target_season - draft_season), i.e. the player's age
            during target_season specifically. Unlike performance stats, age is a
            deterministic fact we can compute for any future season with no leakage risk,
            so (unique among this training set's features) it's computed as of
            target_season rather than the feature row's own "season".

        Players with no draft record (e.g. undrafted free agents) get NaN for both
        "draft_pick" and "age" -- left for the modeling pipeline's imputer to handle.

        Args:
            df: Dataframe with player_grouping_col and "target_season" columns.
            draft_features_df: Output of load_draft_features.
            player_grouping_col: Column identifying a unique player (default: "player_id")

        Returns:
            df with "draft_pick" and "age" columns added; "draft_season"/"age_at_draft"
            (only needed to compute "age") are dropped.
        """
        df = df.merge(draft_features_df, on=player_grouping_col, how="left")
        df["age"] = df["age_at_draft"] + (df["target_season"] - df["draft_season"])
        return df.drop(columns=["draft_season", "age_at_draft"])

    def load_team_features(self) -> pd.DataFrame:
        """
        Loads the `team_stats` silver table, filtered to the registered identity/stat columns.

        Returns:
            One row per team-season, with only the registered identity/stat columns kept.
        """
        identity_columns = get_identity_columns("nflverse", "team_stats")
        stat_columns = get_stat_columns("nflverse", "team_stats")

        team_df = pd.read_csv(os.path.join(self.silver_dir, "team_stats.csv"), low_memory=False)
        team_df = team_df[team_df["team"].notna()]
        return team_df[identity_columns + stat_columns]

    def _league_average_team_stats(self, team_features_df: pd.DataFrame, stat_columns: List[str]) -> pd.DataFrame:
        """
        Computes each season's league-wide average of every stat in stat_columns.

        Args:
            team_features_df: Output of load_team_features (one row per team-season)
            stat_columns: The team stat columns to average

        Returns:
            One row per season, with a "{stat}_league_avg" column per stat in stat_columns
        """
        return (
            team_features_df.groupby("season")[stat_columns]
            .mean()
            .add_suffix("_league_avg")
            .reset_index()
        )

    def _join_team_features(
        self,
        df: pd.DataFrame,
        player_team_by_season: pd.DataFrame,
        team_features_df: pd.DataFrame,
        team_stat_columns: List[str],
        player_grouping_col: str = "player_id",
        team_col: str = "recent_team",
    ) -> pd.DataFrame:
        """
        Adds team-context features to each row of df, capturing the level of a
        player's team relative to league avg and the shift in team quality when a player changes teams.

        The "season" column in df is used to join with the "season" column in team_features_df
        (along with "team") as only the team's stats prior to "target_season" can be in a training row.

        Adds the following types of columns:
          - "team_{stat}": how much better/worse at {stat} the player's team in "target_season" was than leage avg.
          - "team_shift_{stat}" = How much better/worse at {stat} the player's team in "target_season" was than
            the player's team in "season".

        Args:
            df: Dataframe with player_stats.
            player_team_by_season: One row per player-season, with at least
                [player_grouping_col, "season", team_col] columns -- used only to look up
                each player's actual team in target_season (e.g.
                features_df[[player_grouping_col, "season", team_col]]; doesn't need the
                full feature set)
            team_features_df: Output of load_team_features.
            team_stat_columns: Team stat columns to join in and compute a shift for
            player_grouping_col: Column identifying a unique player (default: "player_id")
            team_col: Column identifying a player's team on a given row (default: "recent_team")

        Returns:
            df with "team_{stat}"/"team_shift_{stat}" columns added per stat in
            team_stat_columns
        """
        df = df.copy()

        destination_lookup = (
            player_team_by_season[[player_grouping_col, "season", team_col]]
            .rename(columns={"season": "target_season", team_col: "destination_team"})
        )

        df = df.merge(destination_lookup, on=[player_grouping_col, "target_season"], how="left")
        # Fill rows with no destination team with the player's current team.
        df["destination_team"] = df["destination_team"].fillna(df[team_col])

        team_lookup = team_features_df[["team", "season"] + team_stat_columns]

        origin_stats = team_lookup.rename(
            columns={"team": team_col, **{stat: f"_origin_team_{stat}"
            for stat in team_stat_columns}}
        )
        df = df.merge(origin_stats, on=[team_col, "season"], how="left")

        destination_stats = team_lookup.rename(
            columns={"team": "destination_team", **{stat: f"_destination_team_{stat}"
            for stat in team_stat_columns}}
        )
        df = df.merge(destination_stats, on=["destination_team", "season"], how="left")


        league_avg_df = self._league_average_team_stats(team_features_df, team_stat_columns)
        df = df.merge(league_avg_df, on="season", how="left")

        team_columns = {
            f"team_{stat}": df[f"_destination_team_{stat}"] - df[f"{stat}_league_avg"]
            for stat in team_stat_columns
        }
        shift_columns = {
            f"team_shift_{stat}": df[f"_destination_team_{stat}"] - df[f"_origin_team_{stat}"]
            for stat in team_stat_columns
        }
        df = pd.concat(
            [df, pd.DataFrame(team_columns, index=df.index), pd.DataFrame(shift_columns, index=df.index)], axis=1
        )

        drop_columns = (
            ["destination_team"]
            + [f"_origin_team_{stat}" for stat in team_stat_columns]
            + [f"_destination_team_{stat}" for stat in team_stat_columns]
            + [f"{stat}_league_avg" for stat in team_stat_columns]
        )
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
        draft_features_df: pd.DataFrame,
        target_col: str = "fantasy_points_ppr",
    ) -> pd.DataFrame:
        """
        Builds the training set from the features dataframe and specified target
        and saves it to the gold layer.

        Args:
            features_df: Output of load_player_features
            team_features_df: Output of load_team_features
            draft_features_df: Output of load_draft_features
            target_col: Column to predict; must be a registered target for player_stats
                (default: "fantasy_points_ppr")

        Returns:
            DataFrame of the training set, rounded to four sig figs and saved to
            gold_dir/{target_col}__training_set.csv.
        """
        targets = get_targets("nflverse", "player_stats")
        assert target_col in targets, f"{target_col} is not a registered target for player_stats: {targets}"

        team_stat_columns = get_stat_columns("nflverse", "team_stats")

        training_df = self._join_with_target(features_df, target_col)
        player_team_by_season_df = features_df[["player_id", "season", "recent_team"]]
        training_df = self._join_team_features(
            training_df, player_team_by_season_df, team_features_df, team_stat_columns
        )
        training_df = self._join_draft_features(training_df, draft_features_df)

        output_path = os.path.join(self.gold_dir, f"{target_col}__training_set.csv")
        self._round_significant_figures(training_df, exclude_columns=ROUNDING_EXCLUDED_COLUMNS).to_csv(
            output_path, index=False
        )
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
        draft_features_df: pd.DataFrame,
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
            draft_features_df: Output of load_draft_features
            target_col: Column that will eventually be predicted; must be a registered target
                for player_stats (only used for naming the output file consistently with
                build_training_set -- the actual target values are blank)
            prediction_season: The season to build a prediction row for, e.g. 2026

        Returns:
            DataFrame of the prediction set, rounded to 4 sigfigs, and saved to
            gold_dir/{target_col}__prediction_set.csv.
        """
        targets = get_targets("nflverse", "player_stats")
        assert target_col in targets, f"{target_col} is not a registered target for player_stats: {targets}"

        team_stat_columns = get_stat_columns("nflverse", "team_stats")

        prediction_df = self._build_prediction_rows(features_df, prediction_season)

        # TODO: This is wrong, this method should actually take a df of players for whom we want to make predictions
        # along with their teams for the current season. For now we pass an empty df which has the effect of assuming
        # everyone stayed on the same team.
        no_destination_teams_df = pd.DataFrame(columns=["player_id", "season", "recent_team"])
        prediction_df = self._join_team_features(
            prediction_df, no_destination_teams_df, team_features_df, team_stat_columns
        )
        prediction_df = self._join_draft_features(prediction_df, draft_features_df)

        output_path = os.path.join(self.gold_dir, f"{target_col}__prediction_set.csv")
        self._round_significant_figures(prediction_df, exclude_columns=ROUNDING_EXCLUDED_COLUMNS).to_csv(
            output_path, index=False
        )
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
    draft_features_df = builder.load_draft_features()
    builder.build_training_set(features_df, team_features_df, draft_features_df, target_col=args.target_col)
    builder.build_prediction_set(
        features_df, team_features_df, draft_features_df,
        target_col=args.target_col, prediction_season=args.prediction_season
    )


if __name__ == "__main__":
    main()
