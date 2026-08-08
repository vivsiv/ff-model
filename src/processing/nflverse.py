import os
import logging
import argparse

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("nflverse_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]
DRAFT_TEAMS = 32

TEAM_RELOCATIONS = {
    "STL": "LA",   # Rams
    "SD": "LAC",   # Chargers
    "OAK": "LV",   # Raiders
    "JAC": "JAX",  # Jacksonville
}

class NflverseProcessor:
    """Generates silver layer data from nflverse bronze layer data."""

    def __init__(self, data_dir: str):
        """
        Initialize the processor.

        Args:
            data_dir: Parent directory for the bronze and silver layers.
        """
        self.bronze_dir = os.path.join(data_dir, "bronze", "nflv")
        if not os.path.exists(self.bronze_dir):
            raise FileNotFoundError(f"{self.bronze_dir} not found")

        self.silver_dir = os.path.join(data_dir, "silver", "nflv")
        os.makedirs(self.silver_dir, exist_ok=True)

    def _load_bronze(self, file_name: str) -> pd.DataFrame:
        """
        Loads an nflverse file from the bronze layer.

        Args:
            file_name: The bronze file to load, e.g. "player_stats.csv"

        Returns:
            DataFrame with the full contents of the bronze file
        """
        file_path = os.path.join(self.bronze_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        return pd.read_csv(file_path, low_memory=False)

    def build_player_stats(self) -> pd.DataFrame:
        """
        Loads player stats, perfornms relevant transformations, and saves the result
        to the silver layer.

        Args:
            positions: Player positions to keep (default: QB, RB, WR, TE)

        Returns:
            DataFrame with the filtered player stats, with "recent_team" normalized to a
            single code per franchise across its history.
        """
        player_stats_df = self._load_bronze("player_stats.csv")
        player_stats_df = player_stats_df[player_stats_df["position"].isin(FANTASY_POSITIONS)]
        player_stats_df["recent_team"] = player_stats_df["recent_team"].replace(TEAM_RELOCATIONS)

        output_path = os.path.join(self.silver_dir, "player_stats.csv")
        player_stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved player stats to {output_path}")

        return player_stats_df

    def build_team_stats(self) -> pd.DataFrame:
        """
        Loads nflverse team stats, performs relevant transformations, and saves the result to the silver layer.

        Returns:
            DataFrame with the team stats, with "team" normalized to a single code per
            franchise across its history.
        """
        team_stats_df = self._load_bronze("team_stats.csv")
        team_stats_df["team"] = team_stats_df["team"].replace(TEAM_RELOCATIONS)

        output_path = os.path.join(self.silver_dir, "team_stats.csv")
        team_stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved team stats to {output_path}")

        return team_stats_df

    def build_draft_picks(self) -> pd.DataFrame:
        """
        Loads nflverse draft picks, filters to fantasy-relevant positions, and saves the
        result to the silver layer.

        Returns:
            DataFrame with the filtered draft picks. Dropped:
              - "team", "position": only needed to filter to fantasy-relevant positions;
                not needed downstream since player_stats already has its own (current,
                season-specific) "recent_team"/"position" to join in.
              - "category" (redundant with "position") and "side" (constant "O" once
                filtered to offensive fantasy positions).
              - "car_av" and defensive-only stat columns: always null/not applicable for
                fantasy-relevant positions.
              - "pfr_player_id", "cfb_player_id", "pfr_player_name", "college": identifying
                metadata not needed once joined to player_stats via "player_id".
              - "games", "pass_completions", "pass_attempts", "pass_yards", "pass_tds",
                "pass_ints", "rush_atts", "rush_yards", "rush_tds", "receptions",
                "rec_yards", "rec_tds": career-to-date counting stats that are both
                redundant with (and less complete than) player_stats' own season-by-season
                data, and a leakage risk since they're running totals as of the data pull,
                not truncated to a specific season.
            "gsis_id" is renamed to "player_id" to match player_stats' join key, and "age"
            is renamed to "age_at_draft" since it's a fixed point-in-time value (not the
            player's age in any given season) -- gold-layer feature engineering can use it
            plus a row's season to derive the player's actual age that season. "round"
            and "pick" are combined into a single "draft_pick" feature (overall pick
            number: (round - 1) * DRAFT_TEAMS + pick) and dropped, since round/pick only
            matter together as one measure of draft capital.

            "hof", "to", "allpro", "probowls", "seasons_started", "w_av", "dr_av" are all
            dropped: like the counting stats above, they're career-final totals (e.g. "to"
            is literally the player's last season played), not resolved per season, so
            using them as a feature on an earlier-season training row would leak the rest
            of the player's career into the input.

            A handful of players (e.g. Bo Jackson, Craig Erickson) were drafted twice
            (didn't sign, then re-entered a later draft), so player_id isn't naturally
            unique here. Rows with a player_id are de-duplicated down to their most recent
            (highest season) draft record, so each player_id maps to exactly one row.
            Rows with no player_id (players who never appeared in nflverse's ID crosswalk,
            e.g. career busts) are left as-is and can't be de-duplicated by player.
        """
        draft_picks_df = self._load_bronze("draft_picks.csv")
        draft_picks_df = draft_picks_df[draft_picks_df["position"].isin(FANTASY_POSITIONS)]
        draft_picks_df["draft_pick"] = (draft_picks_df["round"] - 1) * DRAFT_TEAMS + draft_picks_df["pick"]
        draft_picks_df = draft_picks_df.drop(columns=[
            "round", "pick", "team", "position", "category", "side",
            "car_av", "def_solo_tackles", "def_ints", "def_sacks",
            "pfr_player_id", "cfb_player_id", "pfr_player_name", "college",
            "games", "pass_completions", "pass_attempts", "pass_yards", "pass_tds",
            "pass_ints", "rush_atts", "rush_yards", "rush_tds", "receptions",
            "rec_yards", "rec_tds",
            "hof", "to", "allpro", "probowls", "seasons_started", "w_av", "dr_av",
        ])
        draft_picks_df = draft_picks_df.rename(columns={"gsis_id": "player_id", "age": "age_at_draft"})

        has_player_id = draft_picks_df["player_id"].notna()
        deduped_df = (
            draft_picks_df[has_player_id]
            .sort_values("season", kind="stable")
            .drop_duplicates(subset="player_id", keep="last")
        )
        draft_picks_df = pd.concat([deduped_df, draft_picks_df[~has_player_id]], ignore_index=True)

        output_path = os.path.join(self.silver_dir, "draft_picks.csv")
        draft_picks_df.to_csv(output_path, index=False)
        logger.info(f"Saved draft picks to {output_path}")

        return draft_picks_df

    def process_all_data(self, positions: list[str] = FANTASY_POSITIONS) -> None:
        """Builds all silver layer tables from bronze layer data."""
        self.build_player_stats()
        self.build_team_stats()
        self.build_draft_picks()


def main():
    parser = argparse.ArgumentParser(
        description="Processes nflverse bronze data into silver layer tables"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory the nflverse bronze data lives in and silver data will be saved "
             "to, relative to the repo root (default: data)"
    )

    args = parser.parse_args()

    processor = NflverseProcessor(data_dir=args.data_dir)
    processor.process_all_data()


if __name__ == "__main__":
    main()
