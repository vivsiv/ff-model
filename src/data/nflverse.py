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


class NflverseDataScraper:
    """Data source for player and team stats from the nflverse project (via nflreadpy)."""

    def __init__(self, data_dir: str = "../data/nflv"):
        """
        Initialize the data source.

        Args:
            data_dir: Directory to save fetched data
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.bronze_data_dir = os.path.join(data_dir, "bronze")
        os.makedirs(self.bronze_data_dir, exist_ok=True)

    def _save_by_season(self, df: pd.DataFrame, file_prefix: str) -> None:
        """
        Splits a multi-season dataframe by its 'season' column and writes one CSV per season
        to the bronze layer, matching the per-year file convention used by other sources.
        """
        for season, season_df in df.groupby("season"):
            output_path = os.path.join(self.bronze_data_dir, f"{season}_{file_prefix}.csv")
            season_df.to_csv(output_path, index=False)
            logger.info(f"Saved {file_prefix} for {season} to {output_path}")

    def fetch_player_stats(self, start_year: int, end_year: int, summary_level: str = "reg") -> pd.DataFrame:
        """
        Fetch player season stats (passing, rushing, receiving, kicking, fantasy points, etc.)
        for a range of years and save them to the bronze layer, one file per year.

        Args:
            start_year: First season to fetch
            end_year: Last season to fetch
            summary_level: One of "week", "reg", "post", "reg+post" (default: "reg")

        Returns:
            DataFrame with player stats for all requested years
        """
        years = list(range(start_year, end_year + 1))
        logger.info(f"Fetching nflverse player stats for {years}")

        df = nfl.load_player_stats(seasons=years, summary_level=summary_level).to_pandas()
        if not df.empty:
            self._save_by_season(df, "nflverse_player_stats")
        else:
            logger.warning(f"No player stats returned for {years}")

        return df

    def fetch_team_stats(self, start_year: int, end_year: int, summary_level: str = "reg") -> pd.DataFrame:
        """
        Fetch team season stats for a range of years and save them to the bronze layer, one file per year.

        Args:
            start_year: First season to fetch
            end_year: Last season to fetch
            summary_level: One of "week", "reg", "post", "reg+post" (default: "reg")

        Returns:
            DataFrame with team stats for all requested years
        """
        years = list(range(start_year, end_year + 1))
        logger.info(f"Fetching nflverse team stats for {years}")

        df = nfl.load_team_stats(seasons=years, summary_level=summary_level).to_pandas()
        if not df.empty:
            self._save_by_season(df, "nflverse_team_stats")
        else:
            logger.warning(f"No team stats returned for {years}")

        return df

    def fetch_years(self, start_year: int, end_year: int) -> None:
        """
        Fetch player and team stats for a range of years.

        Args:
            start_year: First year to fetch
            end_year: Last year to fetch
        """
        self.fetch_player_stats(start_year, end_year)
        self.fetch_team_stats(start_year, end_year)


def main():
    parser = argparse.ArgumentParser(
        description="Fetches NFL stats from the nflverse project"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1999,
        help="Start year to fetch (default: 1999)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year to fetch (default: 2024)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save fetched data."
    )

    args = parser.parse_args()

    kwargs = {"data_dir": args.data_dir} if args.data_dir is not None else {}
    source = NflverseDataScraper(**kwargs)
    source.fetch_years(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
