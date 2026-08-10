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
            DataFrame with player_id, overall draft_pick, and age_at_draft.

            Rows with no player_id (players nflverse never mapped to a gsis id, e.g. career
            busts) are dropped entirely -- they can't be joined to player_stats, so they're
            useless downstream.

            A handful of players (e.g. Bo Jackson, Craig Erickson) were drafted twice, so
            player_id isn't naturally unique here. Rows are de-duplicated down to each
            player_id's most recent draft record.
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
        ], errors="ignore")
        draft_picks_df = draft_picks_df.rename(columns={"gsis_id": "player_id", "age": "age_at_draft"})

        draft_picks_df = draft_picks_df[draft_picks_df["player_id"].notna()]
        draft_picks_df = draft_picks_df.sort_values("season", kind="stable").drop_duplicates(
            subset="player_id", keep="last"
        )

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
