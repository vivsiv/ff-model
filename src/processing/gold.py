import os
import logging
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
from nflreadpy import get_current_season

from src.processing.column_registry import (
    get_identity_columns,
    get_stat_columns,
    get_counting_stat_columns,
    get_targets,
)

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
PER_GAME_SUFFIX = "_per_game"


def gold_set_filename(target_col: str, current_season: int) -> str:
    """Filename of a target's combined gold table for a specific current season."""
    return f"{target_col}__{current_season}.csv"


def load_gold_set(gold_dir: str, target_col: str, current_season: int) -> pd.DataFrame:
    """
    Loads a target's gold table, as saved by GoldSetBuilder.save_to_gold.

    Args:
        gold_dir: The gold layer directory (the one save_to_gold saved to).
        target_col: The target the gold table was built for, e.g. "ppr_points_per_game".
        current_season: The current season identifying the gold snapshot.

    Returns:
        Historical and current rows, with blank targets for current_season.
    """
    gold_df = pd.read_csv(os.path.join(gold_dir, gold_set_filename(target_col, current_season)))
    logger.info(f"Loaded gold set for {target_col}, current season {current_season}: {len(gold_df)} rows")
    return gold_df


ROUNDING_EXCLUDED_COLUMNS = [
    "season", "target_season", "seasons_since_played", "years_played", "games", "age", "draft_pick",
]


class GoldSetBuilder:
    """Builds gold layer historical/current data from nflverse silver layer data."""

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
        shrinkage_k: float,
        player_grouping_col: str = "player_id",
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
            shrinkage_k: Shrinkage strength constant — higher pulls harder toward the positional
                baseline for a given years_played.
            player_grouping_col: Column identifying a unique player (default: "player_id")

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

    def _add_per_game_stats(
        self,
        df: pd.DataFrame,
        counting_stat_columns: List[str],
        games_col: str = "games",
    ) -> tuple[pd.DataFrame, List[str]]:
        """
        Adds per game rate versions of each column in counting_stat_columns, named "{stat}_per_game".
        The new columns can then be passed to _positional_baseline/_add_career_features to get
        (avg/std/max/min/trend/shrunk_avg) versions of the new per game columns.

        Args:
            df: Dataframe with counting_stat_columns and games_col
            counting_stat_columns: Season-total stat columns to add a per-game version of
                (see get_counting_stat_columns)
            games_col: Column to divide by (default: "games"), also ignored if in counting_stat_columns.

        Returns:
            (df, per_game_columns) -- df with one new "{stat}_per_game" column added per
            eligible stat; per_game_columns lists the names of those new columns.
        """
        eligible_stats = [stat for stat in counting_stat_columns if stat != games_col]
        per_game_columns = {
            f"{stat}{PER_GAME_SUFFIX}": np.where(df[games_col] > 0, df[stat] / df[games_col], 0.0)
            for stat in eligible_stats
        }

        df = pd.concat([df, pd.DataFrame(per_game_columns, index=df.index)], axis=1)
        return df, [f"{stat}{PER_GAME_SUFFIX}" for stat in eligible_stats]

    def load_player_features(self, shrinkage_k: float = 1.5) -> pd.DataFrame:
        """
        Loads the `player_stats` silver table, merges in the `snap_counts` silver table (joined on player_id + season),
        and adds computed feature values to the combined set of stat columns -- both season-total career features and
        (for counting stats) a second, per-game-basis set of the same features. Both historical
        and current row preparation use the output.

        Args:
            shrinkage_k: Shrinkage strength constant passed through to _add_career_features (default: 1.5)

        Returns:
            One row per player-season, with career-to-date features through that season
        """
        player_df = pd.read_csv(os.path.join(self.silver_dir, "player_stats.csv"), low_memory=False)
        player_identity_columns = get_identity_columns("nflverse", "player_stats")
        player_stat_columns = get_stat_columns("nflverse", "player_stats")
        player_df = player_df[player_identity_columns + player_stat_columns]

        snap_counts_df = pd.read_csv(os.path.join(self.silver_dir, "snap_counts.csv"), low_memory=False)
        snap_count_identity_columns = get_identity_columns("nflverse", "snap_counts")
        snap_count_stat_columns = get_stat_columns("nflverse", "snap_counts")
        snap_counts_df = snap_counts_df[snap_count_identity_columns + snap_count_stat_columns]

        player_df = player_df.merge(snap_counts_df, on=snap_count_identity_columns, how="left")
        stat_columns = player_stat_columns + snap_count_stat_columns

        counting_stat_columns = (
            get_counting_stat_columns("nflverse", "player_stats")
            + get_counting_stat_columns("nflverse", "snap_counts")
        )
        player_df, per_game_columns = self._add_per_game_stats(player_df, counting_stat_columns)
        all_stat_columns = stat_columns + per_game_columns

        baseline_df = self._positional_baseline(player_df, all_stat_columns)
        return self._add_career_features(player_df, baseline_df, all_stat_columns, shrinkage_k=shrinkage_k)

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
            target_col: Column in features_df to use as the target, e.g.
                "fantasy_points_ppr". 
            player_grouping_col: Column identifying a unique player (default: "player_id")

        Returns:
            DataFrame of feature rows (most recent season before target_season) with
            "target_season", "target", and "seasons_since_played" columns added
        """
        targets = get_targets("nflverse", "player_stats")
        assert target_col in targets, f"{target_col} is not a registered target for player_stats: {targets}"

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

    def build_gold_set(
        self,
        rows_df: pd.DataFrame,
        team_features_df: pd.DataFrame,
        draft_features_df: pd.DataFrame,
        player_team_by_season: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adds team-context and draft features to prepared historical/current rows.

        Args:
            rows_df: Output of _join_with_target and/or _build_current_rows.
            team_features_df: Output of load_team_features
            draft_features_df: Output of load_draft_features
            player_team_by_season: Player/team lookup for each target season. Combine
                historical player teams with load_current_players' current teams
                when building a combined gold table.

        Returns:
            DataFrame of gold rows, without rounding or saving.
        """
        team_stat_columns = get_stat_columns("nflverse", "team_stats")
        gold_df = self._join_team_features(
            rows_df, player_team_by_season, team_features_df, team_stat_columns
        )
        return self._join_draft_features(gold_df, draft_features_df)

    #TODO: should this move to common code in the data quality checks?
    @staticmethod
    def validate_data(
        gold_df: pd.DataFrame,
        current_season: int,
    ) -> None:
        """Validate the combined gold table before saving.

        Historical targets must be present and current targets must be blank.
        Features must be numeric and player/feature-season keys unique. Each
        current row must be later than that player's historical feature rows.
        Feature NaNs are allowed for downstream imputation.

        Raises:
            ValueError: If a structural invariant fails.

        This does not mutate the tables or run the separate analysis quality checks.
        """
        identity_columns = get_identity_columns("nflverse", "player_stats") + ["target_season"]
        non_feature_columns = identity_columns + [TARGET_COL]
        feature_cols = [col for col in gold_df.columns if col not in non_feature_columns]
        non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(gold_df[col])]
        if non_numeric:
            raise ValueError(f"Gold features must be numeric, got non-numeric: {non_numeric}")
        if gold_df.duplicated(["player_id", "season"]).any():
            raise ValueError("More than one row for some (player_id, season) in gold data")
        if gold_df["target_season"].isna().any() or (gold_df["target_season"] > current_season).any():
            raise ValueError("Gold target seasons must be present and no later than current_season")
        if (gold_df["season"] >= current_season).any():
            raise ValueError("Gold feature seasons must be before current_season")

        historical_df = gold_df[gold_df["target_season"] < current_season]
        current_df = gold_df[gold_df["target_season"] == current_season]
        if historical_df[TARGET_COL].isna().any():
            raise ValueError("Historical data must not have null target values")
        if current_df[TARGET_COL].notna().any():
            raise ValueError("Current targets must be blank")
        if current_df["player_id"].duplicated().any():
            raise ValueError("Current data must have one row per player")
        last_historical_seasons = historical_df.groupby("player_id")["season"].max()
        prior_seasons = current_df["player_id"].map(last_historical_seasons)
        if (current_df["season"] <= prior_seasons).any():
            raise ValueError("A player's current row must be later than all historical feature seasons")

    def save_to_gold(self, df: pd.DataFrame, filename: str) -> None:
        """Save a table under gold_dir, rounding numeric values to four significant figures.

        Columns in ROUNDING_EXCLUDED_COLUMNS are left untouched, as is the input
        DataFrame. The CSV is written without a pandas index.

        Args:
            df: The table to save.
            filename: Output filename relative to gold_dir.
        """
        output_path = os.path.join(self.gold_dir, filename)
        self._round_significant_figures(df, exclude_columns=ROUNDING_EXCLUDED_COLUMNS).to_csv(
            output_path, index=False
        )
        logger.info(f"Saved gold table to {output_path}")

    def _build_current_rows(
        self,
        features_df: pd.DataFrame,
        players_df: pd.DataFrame,
        current_season: int,
        player_grouping_col: str = "player_id",
    ) -> pd.DataFrame:
        """
        Takes each player's most recent season from the features dataframe and reframes it as
        a current-season row with a blank target.

        Only players present in players_df get a row, so retired players are excluded. A player who
        missed some seasons still gets a row built from their last active season, and players with no prior
        season on record (rookies) are dropped as there is nothing to build features from.

        Args:
            features_df: Output of load_player_features (one row per player-season)
            players_df: Output of load_current_players, the players to build current rows for
            current_season: The season to build a current row for, e.g. 2026
            player_grouping_col: Column identifying a unique player (default: "player_id")

        Returns:
            DataFrame with one row per player in players_df with prior stats, "target_season" set to
            current_season, "target" as NaN, and "seasons_since_played" computed the same way as
            _join_with_target
        """
        latest_df = (
            features_df[
                features_df[player_grouping_col].isin(players_df[player_grouping_col])
                & (features_df["season"] < current_season)
            ]
            .sort_values("season")
            .groupby(player_grouping_col, as_index=False)
            .tail(1)
            .copy()
        )
        latest_df["target_season"] = current_season
        latest_df["seasons_since_played"] = current_season - latest_df["season"] - 1
        latest_df[TARGET_COL] = np.nan

        return latest_df.reset_index(drop=True)

    def load_current_players(self) -> pd.DataFrame:
        """
        Loads the `ff_rankings` silver table: the fantasy viable players for the current
        season and the team each is currently listed on.

        The latest rankings are assumed to describe nflreadpy's current roster season
        (which rolls over on March 15). Historical player pools are not supported.

        Returns:
            One row per player with "player_id", "season" (== current_season), and "recent_team"
            (the team they're on in current_season, NaN for free agents)
        """
        players_df = pd.read_csv(os.path.join(self.silver_dir, "ff_rankings.csv"), low_memory=False)
        if players_df.empty:
            raise ValueError("No current players found in ff_rankings")
        players_df["season"] = get_current_season(roster=True)

        return players_df.rename(columns={"team": "recent_team"})[["player_id", "season", "recent_team"]]


def main():
    parser = argparse.ArgumentParser(
        description="Builds a combined gold table for the current season"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="fantasy_points_ppr",
        help="Column to predict; must be a registered target for player_stats (default: fantasy_points_ppr)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Parent directory for the silver and gold layers, relative to the repo root (default: data)"
    )
    parser.add_argument(
        "--shrinkage-k",
        type=float,
        default=1.5,
        help="Shrinkage strength constant for career stat features -- higher pulls harder "
             "toward the positional baseline for a given years_played (default: 1.5)"
    )

    args = parser.parse_args()

    builder = GoldSetBuilder(data_dir=args.data_dir)

    players_df = builder.load_current_players()
    current_season = int(players_df["season"].iloc[0])

    features_df = builder.load_player_features(shrinkage_k=args.shrinkage_k)
    features_df = features_df[features_df["season"] < current_season]
    team_features_df = builder.load_team_features()
    draft_features_df = builder.load_draft_features()

    historical_rows = builder._join_with_target(features_df, target_col=args.target_col)
    current_rows = builder._build_current_rows(features_df, players_df, current_season)
    rows_df = pd.concat([historical_rows, current_rows], ignore_index=True)
    player_team_by_season = pd.concat(
        [features_df[["player_id", "season", "recent_team"]], players_df], ignore_index=True
    )

    gold_df = builder.build_gold_set(rows_df, team_features_df, draft_features_df, player_team_by_season)
    builder.validate_data(gold_df, current_season)
    builder.save_to_gold(gold_df, gold_set_filename(args.target_col, current_season))


if __name__ == "__main__":
    main()
