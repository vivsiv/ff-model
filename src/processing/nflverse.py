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

    def process_all_data(self, positions: list[str] = FANTASY_POSITIONS) -> None:
        """Builds all silver layer tables from bronze layer data."""
        self.build_player_stats()
        self.build_team_stats()


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
