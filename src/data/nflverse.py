import os
import logging
import argparse

import nflreadpy as nfl
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("nflverse_source.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NflverseScraper:
    """Data source for player and team stats from the nflverse project (via nflreadpy)."""

    def __init__(self, data_dir: str):
        """
        Initialize the data source.

        Args:
            data_dir: Directory to save fetched data
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.bronze_dir = os.path.join(data_dir, "bronze", "nflv")
        os.makedirs(self.bronze_dir, exist_ok=True)

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        """
        Writes a dataframe to a single csv file in the bronze layer.
        """
        output_path = os.path.join(self.bronze_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {filename} to {output_path}")

    def fetch_player_stats(self) -> pd.DataFrame:
        """
        Fetch player season stats for all available seasons,
        Save them to the bronze layer as a single file.

        Returns:
            DataFrame with player stats for all available seasons
        """
        logger.info("Fetching nflverse player stats for all available seasons")

        df = nfl.load_player_stats(seasons=True, summary_level="reg").to_pandas()
        if not df.empty:
            self._save(df, "player_stats.csv")
        else:
            logger.warning("No player stats returned")

        return df

    def fetch_team_stats(self) -> pd.DataFrame:
        """
        Fetch team season stats for all available seasons,
        save them to the bronze layer as a single file.

        Returns:
            DataFrame with team stats for all available seasons
        """
        logger.info("Fetching nflverse team stats for all available seasons")

        df = nfl.load_team_stats(seasons=True, summary_level="reg").to_pandas()
        if not df.empty:
            self._save(df, "team_stats.csv")
        else:
            logger.warning("No team stats returned")

        return df

    def fetch_draft_picks(self) -> pd.DataFrame:
        """
        Fetch draft pick data for all available seasons,
        save them to the bronze layer as a single file.

        Returns:
            DataFrame with draft pick data for all available seasons
        """
        logger.info("Fetching nflverse draft picks for all available seasons")

        df = nfl.load_draft_picks(seasons=True).to_pandas()
        if not df.empty:
            self._save(df, "draft_picks.csv")
        else:
            logger.warning("No draft picks returned")

        return df

    def fetch_snap_counts(self) -> pd.DataFrame:
        """
        Fetch per-game snap count data for all available seasons (2013+ -- nflverse has no
        snap count data before then), save them to the bronze layer as a single file.

        Returns:
            DataFrame with one row per player per game, identified by "pfr_player_id" (no
            gsis "player_id" is available directly on this table).
        """
        logger.info("Fetching nflverse snap counts for all available seasons")

        df = nfl.load_snap_counts(seasons=True).to_pandas()
        if not df.empty:
            self._save(df, "snap_counts.csv")
        else:
            logger.warning("No snap counts returned")

        return df

    def fetch_players(self) -> pd.DataFrame:
        """
        Fetch the nflverse player ID dictionary (one row per player, mapping "gsis_id" to
        other sources' player IDs, e.g. "pfr_id"), save it to the bronze layer as a single
        file.

        Used to join tables (e.g. snap counts) that only carry an external source's player
        ID back onto "player_id" (gsis_id), the ID player_stats/draft_picks use.

        Returns:
            DataFrame with one row per player
        """
        logger.info("Fetching nflverse player ID dictionary")

        df = nfl.load_players().to_pandas()
        if not df.empty:
            self._save(df, "players.csv")
        else:
            logger.warning("No players returned")

        return df

    def fetch_all(self) -> None:
        """Fetch player, team, draft pick, snap count, and player ID data for all available seasons."""
        self.fetch_player_stats()
        self.fetch_team_stats()
        self.fetch_draft_picks()
        self.fetch_snap_counts()
        self.fetch_players()


def main():
    parser = argparse.ArgumentParser(
        description="Fetches NFL stats from the nflverse project"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save fetched data, relative to the repo root (default: data)"
    )

    args = parser.parse_args()

    source = NflverseScraper(data_dir=args.data_dir)
    source.fetch_all()


if __name__ == "__main__":
    main()
