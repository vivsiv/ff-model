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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def _get_soup(self, url: str, delay: float = 3.0, overwrite: bool = False) -> BeautifulSoup:
        """
        Get BeautifulSoup object from URL with rate limiting.

        Args:
            url: URL to scrape
            delay: Time to wait between requests (seconds)
            overwrite: Whether to overwrite the raw file if it already exists (default: False)

        Returns:
            BeautifulSoup object
        """
        # Calculate the raw file path
        url_path = url.replace(self.base_url, "").replace("/", "_").strip("_")
        if not url_path:
            url_path = "index"
        raw_path = os.path.join(self.raw_data_dir, f"{url_path}.html")

        # Check if the raw file already exists
        if os.path.exists(raw_path) and not overwrite:
            logger.info(f"Using existing file {raw_path}")
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
        soup = self._get_soup(url)

        if not soup:
            logger.error(f"Failed to get data for {year}")
            return pd.DataFrame()

        # Find the fantasy stats table
        table = soup.find('table', id='fantasy')
        if not table:
            logger.error(f"Fantasy table not found for {year}")
            return pd.DataFrame()

        # Extract columns
        columns = []
        for th in table.find('thead').find_all('th'):
            col_name = th.get_text(strip=True)
            columns.append(col_name)

        data = []
        for tr in table.find('tbody').find_all('tr'):
            # Skip column names
            if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
                continue

            row = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
            data.append(row)

        if data:
            # Adjust columns to match whats in the data, this ignores category headers that precede the actual data.
            actual_column_count = len(data[0])
            # If we have more headers than actual columns, take the last N headers
            if len(columns) > actual_column_count:
                original_count = len(columns)
                columns = columns[-actual_column_count:]
                logger.info(f"Adjusted headers from {original_count} to {actual_column_count}")
        else:
            logger.warning(f"No data rows found for {year}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=columns)

        if not df.empty:
            df['Season'] = year

            output_path = os.path.join(self.processed_data_dir, f"fantasy_stats_{year}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved fantasy stats for {year} to {output_path}")

        return df

    def scrape_team_stats(self, year: int) -> pd.DataFrame:
        """
        Scrape team offensive stats for a specific season.

        Args:
            year: NFL season year

        Returns:
            DataFrame with team stats
        """
        url = f"{self.base_url}/years/{year}/"
        soup = self._get_soup(url)

        if not soup:
            logger.error(f"Failed to get team stats for {year}")
            return pd.DataFrame()

        table = soup.find('table', id='team_stats')
        if not table:
            logger.error(f"Team stats table not found for {year}")
            return pd.DataFrame()

        columns = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]

        data = []
        for tr in table.find('tbody').find_all('tr'):
            # Skip column names
            if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
                continue
            row = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
            data.append(row)

        try:
            # Adjust the set of columns to match whats in the data.
            if data and len(columns) != len(data[0]):
                if len(columns) > len(data[0]):
                    columns = columns[-len(data[0]):]
                else:
                    # Add dummy column names for extra data columns
                    extra_cols_needed = len(data[0]) - len(columns)
                    for i in range(extra_cols_needed):
                        columns.append(f"Unknown_Col_{i+1}")

            df = pd.DataFrame(data, columns=columns)

            df['Season'] = year

            output_path = os.path.join(self.processed_data_dir, f"team_stats_{year}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved team stats for {year} to {output_path}")

            return df
        except Exception as e:
            logger.error(f"Error processing team stats for {year}: {e}")
            return pd.DataFrame()

    def scrape_advanced_team_stats(self, year: int) -> pd.DataFrame:
        """
        Scrape advanced team stats for a specific season.

        Args:
            year: NFL season year

        Returns:
            DataFrame with advanced team stats
        """
        url = f"{self.base_url}/years/{year}/opp.htm"
        soup = self._get_soup(url)

        if not soup:
            logger.error(f"Failed to get advanced team stats for {year}")
            return pd.DataFrame()

        table = soup.find('table', id='team_stats')
        if not table:
            logger.error(f"Advanced team stats table not found for {year}")
            return pd.DataFrame()

        columns = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]

        data = []
        for tr in table.find('tbody').find_all('tr'):
            # Skip column names
            if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
                continue
            row = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
            data.append(row)

        try:
            # Adjust the set of columns to match whats in the data.
            if data and len(columns) != len(data[0]):
                if len(columns) > len(data[0]):
                    columns = columns[-len(data[0]):]
                else:
                    # Add dummy column names for extra data columns
                    extra_cols_needed = len(data[0]) - len(columns)
                    for i in range(extra_cols_needed):
                        columns.append(f"Unknown_Col_{i+1}")

            df = pd.DataFrame(data, columns=columns)
            df['Season'] = year

            output_path = os.path.join(self.processed_data_dir, f"advanced_team_stats_{year}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved advanced team stats for {year} to {output_path}")

            return df
        except Exception as e:
            logger.error(f"Error processing advanced team stats for {year}: {e}")
            return pd.DataFrame()

    def scrape_player_game_logs(self, player_id: str, year: int) -> pd.DataFrame:
        """
        Scrape game logs for a specific player and season.

        Args:
            player_id: Player ID from Pro Football Reference
            year: NFL season year

        Returns:
            DataFrame with player game logs
        """
        first_letter = player_id[0]
        url = f"{self.base_url}/players/{first_letter}/{player_id}/gamelog/{year}/"

        soup = self._get_soup(url)

        if not soup:
            logger.error(f"Failed to get game logs for player {player_id} in {year}")
            return pd.DataFrame()

        table = soup.find('table', id='gamelog')
        if not table:
            logger.warning(f"Game log table not found for player {player_id} in {year}")
            return pd.DataFrame()

        player_name = soup.find('h1', {'itemprop': 'name'})
        if player_name:
            player_name = player_name.text.strip()
        else:
            player_name = "Unknown"

        headers = []
        header_row = table.find('thead')
        if header_row:
            # Handle multi-level headers if present
            header_rows = header_row.find_all('tr')
            if len(header_rows) > 1:
                # Multi-level headers
                top_headers = [th.get_text(strip=True) for th in header_rows[0].find_all('th')]
                bottom_headers = [th.get_text(strip=True) for th in header_rows[1].find_all('th')]

                # Combine multi-level headers
                current_top = ""
                for i, header in enumerate(top_headers):
                    if header:
                        current_top = header
                    if i < len(bottom_headers):
                        if bottom_headers[i]:
                            headers.append(f"{current_top}_{bottom_headers[i]}" if current_top else bottom_headers[i])
                        else:
                            headers.append(current_top)
            else:
                # Single level headers
                headers = [th.get_text(strip=True) for th in header_rows[0].find_all('th')]

        # Extract rows
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            # Skip header rows
            if 'class' in tr.attrs and ('thead' in tr.attrs['class'] or 'divider' in tr.attrs['class']):
                continue

            # Extract row data
            row = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
            if row and len(row) > 1:  # Skip empty rows
                rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Clean up DataFrame
        # Remove rows that are headers or have NaN in date column
        if 'Date' in df.columns:
            df = df[df['Date'].notna()]

        # Add player information
        df['Player_ID'] = player_id
        df['Player_Name'] = player_name
        df['Season'] = year

        # Save to CSV
        output_path = os.path.join(self.data_dir, "processed", f"game_logs_{player_id}_{year}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved game logs for {player_id} ({player_name}) in {year} to {output_path}")

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

            self.scrape_season_fantasy_stats(year)
            self.scrape_team_stats(year)
            self.scrape_advanced_team_stats(year)

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
