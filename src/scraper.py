#!/usr/bin/env python3
"""
NFL Fantasy Football Data Scraper

This module provides functionality to scrape NFL player and team statistics
from Pro Football Reference for fantasy football analysis.
"""

import os
import time
import json
import logging
import random
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProFootballReferenceScraper:
    """Scraper for Pro Football Reference website."""

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the scraper.

        Args:
            data_dir: Directory to save scraped data
        """
        self.base_url = "https://www.pro-football-reference.com"
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, "raw")
        self.processed_data_dir = os.path.join(data_dir, "processed")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Create data directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def _get_soup(self, url: str, raw_file_name: str, delay: float = 3.0, overwrite: bool = False) -> BeautifulSoup:
        """
        Get BeautifulSoup object from URL with rate limiting.

        Args:
            url: URL to scrape
            delay: Time to wait between requests (seconds)
            overwrite: Whether to overwrite the raw file if it already exists (default: False)

        Returns:
            BeautifulSoup object
        """
        raw_path = os.path.join(self.raw_data_dir, f"{raw_file_name}.html")

        # Check if the raw file already exists
        if os.path.exists(raw_path) and not overwrite:
            logger.info(f"Using existing raw data file {raw_path}")
            with open(raw_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return BeautifulSoup(html_content, 'lxml')

        # Add jitter to delay to avoid detection
        time.sleep(delay + random.uniform(0.5, 1.5))

        try:
            logger.info(f"Requesting {url}")
            response = self.session.get(url)
            response.raise_for_status()

            # Save raw HTML
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            return BeautifulSoup(response.text, 'lxml')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def scrape_season_fantasy_stats(self, year: int) -> pd.DataFrame:
        """
        Scrape fantasy stats for a specific season.

        Args:
            year: NFL season year (e.g., 2023)

        Returns:
            DataFrame with player fantasy stats
        """
        url = f"{self.base_url}/years/{year}/fantasy.htm"
        soup = self._get_soup(url, f"{year}_player_fantasy_stats")

        if not soup:
            logger.error(f"Failed to get data for {year}")
            return pd.DataFrame()

        # Find the fantasy stats table
        table = soup.find('table', id='fantasy')
        if not table:
            logger.error(f"Fantasy table not found for {year}")
            return pd.DataFrame()

        # Extract columns from thead elements
        columns = []
        for th in table.find('thead').find_all('th'):
            col_name = th.get_text(strip=True)
            columns.append(col_name)

        # Extract data from tbody elements
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            # Skip column names
            if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
                continue

            row = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
            rows.append(row)

        if len(rows) > 0:
            # Adjust columns to match whats in the data, this ignores category headers that precede the actual data.
            actual_column_count = len(rows[0])
            # If we have more headers than actual columns, take the last N headers
            if len(columns) > actual_column_count:
                original_count = len(columns)
                columns = columns[-actual_column_count:]
                logger.info(f"Adjusted headers from {original_count} to {actual_column_count}")
            elif len(columns) < actual_column_count:
                # Add dummy column names for extra data columns
                extra_cols_needed = actual_column_count - len(columns)
                for i in range(extra_cols_needed):
                    columns.append(f"Unknown_Col_{i+1}")
                logger.info(f"Added {extra_cols_needed} dummy columns")
            else:
                logger.info(f"Headers and data columns match for {year}")
        else:
            logger.warning(f"No data rows found for {year}")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)

        if not df.empty:
            df['Season'] = year

            output_path = os.path.join(self.processed_data_dir, f"{year}_player_fantasy_stats.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved fantasy stats for {year} to {output_path}")

        return df

    def scrape_team_stats(self, year: int) -> Dict[str, pd.DataFrame]:
        """
        Scrape multiple team offensive stats tables for a specific season.
        Tables scraped: team_offense, passing_offense, rushing_offense.

        Args:
            year: NFL season year

        Returns:
            Dictionary of DataFrames, keyed by table ID.
        """
        url = f"{self.base_url}/years/{year}/#all_team_stats"
        # Provide a raw_file_name for caching the main page content and force overwrite
        soup = self._get_soup(url, raw_file_name=f"{year}_team_offense", overwrite=True)

        if not soup:
            logger.error(f"Failed to get page content for team stats for {year} from {url}")
            return {}

        table_ids_to_scrape = ['team_offense', 'passing_offense', 'rushing_offense']
        all_dataframes: Dict[str, pd.DataFrame] = {}

        for table_id in table_ids_to_scrape:
            table = soup.find('table', id=table_id)
            if not table:
                logger.warning(f"Table with id '{table_id}' not found on {url} for year {year}")
                continue

            columns_from_header = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]

            data_rows = []
            for tr in table.find('tbody').find_all('tr'):
                if 'class' in tr.attrs and 'thead' in tr.attrs['class']: # Skip any visually embedded header rows in tbody
                    continue
                row_data = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
                if row_data: # Only append if row actually has data
                    data_rows.append(row_data)

            if not data_rows:
                logger.warning(f"No data rows found in table with id '{table_id}' for year {year}")
                continue

            # Adjust columns based on the first data row, similar to scrape_season_fantasy_stats
            adjusted_columns = list(columns_from_header)
            actual_data_column_count = len(data_rows[0])

            if len(adjusted_columns) > actual_data_column_count:
                original_header_count = len(adjusted_columns)
                adjusted_columns = adjusted_columns[-actual_data_column_count:]
                logger.info(f"For table '{table_id}', year {year}, header columns ({original_header_count}) were more than data columns ({actual_data_column_count}). Adjusted to last {actual_data_column_count} header columns.")
            elif len(adjusted_columns) < actual_data_column_count:
                extra_cols_needed = actual_data_column_count - len(adjusted_columns)
                for i in range(extra_cols_needed):
                    adjusted_columns.append(f"dummy_col_{i+1}")
                logger.info(f"For table '{table_id}', year {year}, header columns ({len(columns_from_header)}) were fewer than data columns ({actual_data_column_count}). Added {extra_cols_needed} dummy columns.")

            try:
                df = pd.DataFrame(data_rows, columns=adjusted_columns)
                df['Season'] = year

                output_filename = f"{year}_{table_id}.csv"
                output_path = os.path.join(self.processed_data_dir, output_filename)
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {table_id} stats for {year} to {output_path}")
                all_dataframes[table_id] = df
            except Exception as e:
                logger.error(f"Error processing table '{table_id}' for year {year}: {e}")
        
        return all_dataframes



    def scrape_player_offensive_stats(self, year: int, category: str) -> pd.DataFrame:
        """
        Scrape player receiving stats for a given year.

        Args:
            year: NFL season year
            category: The category of offensive stats to scrape, one of:
                - "passing"
                - "rushing"
                - "receiving"

        Returns:
            DataFrame with {category} stats for all players in a given year.
        """
        assert category in ["passing", "rushing", "receiving"], "Invalid category"

        url = f"{self.base_url}/years/{year}/{category}_advanced.htm"

        soup = self._get_soup(url, raw_file_name=f"{year}_{category}_stats")

        if not soup:
            logger.error(f"Failed to get {category} stats for {year}")
            return pd.DataFrame()

        if category == "passing":
            table = soup.find('table', id='passing_advanced')
        elif category == "rushing":
            table = soup.find('table', id='adv_rushing')
        elif category == "receiving":
            table = soup.find('table', id='adv_receiving')
        else:
            logger.error(f"Invalid category: {category}")
            return pd.DataFrame()
        if not table:
            return pd.DataFrame()
        
        # Extract columns from thead elements
        columns = []
        for th in table.find('thead').find_all('th'):
            col_name = th.get_text(strip=True)
            columns.append(col_name)

        # Extract data from tbody elements
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            # Skip column names
            if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
                continue

            row = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
            rows.append(row)

        if len(rows) > 0:
            # Adjust columns to match whats in the data, this ignores category headers that precede the actual data.
            actual_column_count = len(rows[0])
            # If we have more headers than actual columns, take the last N headers
            if len(columns) > actual_column_count:
                original_count = len(columns)
                columns = columns[-actual_column_count:]
                logger.info(f"Adjusted headers from {original_count} to {actual_column_count}")
            elif len(columns) < actual_column_count:
                # Add dummy column names for extra data columns
                extra_cols_needed = actual_column_count - len(columns)
                for i in range(extra_cols_needed):
                    columns.append(f"Unknown_Col_{i+1}")
                logger.info(f"Added {extra_cols_needed} dummy columns")
            else:
                logger.info(f"Headers and data columns match for {year}")
        else:
            logger.warning(f"No data rows found for {year}")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)

        if not df.empty:
            df['Season'] = year

            output_path = os.path.join(self.processed_data_dir, f"{year}_{category}_stats.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {category} stats for {year} to {output_path}")

        return df


    def get_player_ids_for_year(self, year: int, min_fantasy_points: float = 50.0) -> List[str]:
        """
        Get list of relevant player IDs for a specific season.

        Args:
            year: NFL season year
            min_fantasy_points: Minimum fantasy points to consider a player relevant

        Returns:
            List of player IDs
        """
        # First scrape the fantasy stats for the year if we don't have them
        fantasy_file = os.path.join(self.data_dir, "processed", f"fantasy_stats_{year}.csv")

        if not os.path.exists(fantasy_file):
            self.scrape_season_fantasy_stats(year)

        if not os.path.exists(fantasy_file):
            logger.error(f"Could not get fantasy stats for {year}")
            return []

        # Load fantasy stats
        df = pd.read_csv(fantasy_file)

        # Extract player IDs for relevant players
        # This assumes the DataFrame has a FantPt column and player_id column
        # Adjust column names as needed based on actual data
        if 'FantPt' in df.columns and 'player_id' in df.columns:
            return df[df['FantPt'] >= min_fantasy_points]['player_id'].tolist()
        else:
            logger.warning(f"Required columns not found in fantasy stats for {year}")
            return []

    def scrape_years(self, start_year: int, end_year: int, include_game_logs: bool = True):
        """
        Scrape data for multiple years.

        Args:
            start_year: First year to scrape
            end_year: Last year to scrape
            include_game_logs: Whether to scrape game logs for relevant players
        """
        years = range(start_year, end_year + 1)

        for year in tqdm(years, desc="Scraping years"):
            logger.info(f"Scraping data from {year}")

            #self.scrape_season_fantasy_stats(year)
            #self.scrape_team_stats(year)
            self.scrape_player_offensive_stats(year, "passing")
            self.scrape_player_offensive_stats(year, "rushing")
            self.scrape_player_offensive_stats(year, "receiving")
            #self.scrape_advanced_team_stats(year)

            if include_game_logs:
                player_ids = self.get_player_ids_for_year(year)
                logger.info(f"Found {len(player_ids)} relevant players in {year}")

                for player_id in tqdm(player_ids, desc=f"Scraping players data in {year}"):
                    self.scrape_player_game_logs(player_id, year)

            logger.info(f"Completed scraping data from {year}")


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(
        description="Scrapes stats from external sources"
    )
    parser.add_argument(
        "--sources",
        type=list,
        default=["pro-football-reference"],
        help="Sources to scrape, currently only pro-football-reference is supported."
    )
    parser.add_argument(
        "--years-to-scrape",
        type=int,
        default=10,
        help="Number of years to scrape back from current year (default: 10)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory to save scraped data, paths are relative to this script (default: ../data)"
    )

    args = parser.parse_args()

    scrapers_map = {
        "pro-football-reference": ProFootballReferenceScraper
    }
    scrapers = [(source, scrapers_map[source]) for source in args.sources]

    current_year = datetime.now().year
    end_year = current_year - 1
    start_year = end_year - args.years_to_scrape + 1
    logger.info(f"Scraping {args.sources} from {start_year} to {end_year}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Saving scraped data to: {data_dir}")

    for source, scraper in scrapers:
        scraper.scrape_years(start_year, end_year)
        logger.info(f"Completed scraping {source}")


if __name__ == "__main__":
    main()
