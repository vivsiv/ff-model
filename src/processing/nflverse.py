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


class NflverseProcessor:
    """Generates silver layer data from nflverse bronze layer data."""

    def __init__(self, data_dir: str = "../data/nflv"):
        """
        Initialize the processor.

        Args:
            data_dir: Root directory the nflverse bronze data lives in and silver data will be saved to.
        """
        self.bronze_dir = os.path.join(data_dir, "bronze")
        if not os.path.exists(self.bronze_dir):
            raise FileNotFoundError(f"{self.bronze_dir} not found")

        self.silver_dir = os.path.join(data_dir, "silver")
        os.makedirs(self.silver_dir, exist_ok=True)

    def _load_bronze(self, filename: str) -> pd.DataFrame:
        """
        Loads a bronze file as-is, with no filtering. Year-range selection is a gold-layer
        concern (so the training window can change without re-running these transformations).

        Args:
            filename: The bronze file to load, e.g. "player_stats.csv"

        Returns:
            DataFrame with the full contents of the bronze file
        """
        file_path = os.path.join(self.bronze_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        return pd.read_csv(file_path, low_memory=False)

    def build_player_stats(self, positions: list[str] = FANTASY_POSITIONS) -> pd.DataFrame:
        """
        Loads nflverse player stats, filters to fantasy-relevant positions, and saves the result
        to the silver layer. player_id is nflverse's stable join key, so no name standardization
        is done here.

        Args:
            positions: Player positions to keep (default: QB, RB, WR, TE)

        Returns:
            DataFrame with the filtered player stats
        """
        player_stats_df = self._load_bronze("player_stats.csv")
        player_stats_df = player_stats_df[player_stats_df["position"].isin(positions)]

        output_path = os.path.join(self.silver_dir, "player_stats.csv")
        player_stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved player stats to {output_path}")

        return player_stats_df

    def build_team_stats(self) -> pd.DataFrame:
        """
        Loads nflverse team stats and saves the result to the silver layer.

        Returns:
            DataFrame with the team stats
        """
        team_stats_df = self._load_bronze("team_stats.csv")

        output_path = os.path.join(self.silver_dir, "team_stats.csv")
        team_stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved team stats to {output_path}")

        return team_stats_df

    def process_all_data(self, positions: list[str] = FANTASY_POSITIONS) -> None:
        """Builds all silver layer tables from bronze layer data."""
        self.build_player_stats(positions)
        self.build_team_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Processes nflverse bronze data into silver layer tables"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root directory the nflverse bronze data lives in and silver data will be saved to (default: class default)"
    )

    args = parser.parse_args()

    kwargs = {"data_dir": args.data_dir} if args.data_dir is not None else {}
    processor = NflverseProcessor(**kwargs)
    processor.process_all_data()


if __name__ == "__main__":
    main()
