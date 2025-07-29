import os
import time
import logging
import random
import argparse

import requests
from bs4 import BeautifulSoup, Comment
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
        os.makedirs(self.data_dir, exist_ok=True)

        self.html_dir = os.path.join(data_dir, "html")
        os.makedirs(self.html_dir, exist_ok=True)

        self.bronze_data_dir = os.path.join(data_dir, "bronze")
        os.makedirs(self.bronze_data_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _get_soup(self, url: str, html_table_path: str, delay: float = 3.0, overwrite: bool = False) -> BeautifulSoup:
        """
        Get BeautifulSoup object from URL with rate limiting.

        Args:
            url: URL to scrape
            html_table_path: Fully qualified path that the html table exists at or will be saved to.
            delay: Time to wait between requests (seconds)
            overwrite: Overwrite the existing html file (default: False)

        Returns:
            BeautifulSoup object
        """

        # If the file specified by html_table_path exsists, use it. 
        # Otherwise scrape the web and save to this location.
        if os.path.exists(html_table_path) and not overwrite:
            logger.info(f"Using existing html file {html_table_path}")
            with open(html_table_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return BeautifulSoup(html_content, 'lxml')

        # Add jitter to delay to avoid detection
        time.sleep(delay + random.uniform(0.5, 1.5))

        try:
            logger.info(f"Requesting {url}")
            response = self.session.get(url)
            response.raise_for_status()

            with open(html_table_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            return BeautifulSoup(response.text, 'lxml')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def scrape_html_table(self, url: str, html_table_file: str, table_id: str, year: int, overwrite: bool = False) -> pd.DataFrame:
        """
        Scrape an HTML table from a URL.

        Args:
            url: URL to scrape
            html_table_file: File the html table is in or will be saved to
            table_id: ID of the table to scrape
            year: Year of the data
            overwrite: Overwrite the existing html file (default: False)

        Returns:
            DataFrame with the scraped table data
        """
        html_table_path = os.path.join(self.html_dir, f"{html_table_file}.html")
        soup = self._get_soup(url, html_table_path, overwrite=overwrite)
        if not soup:
            logger.error(f"Failed to get data for {url}")
            return pd.DataFrame()

        table = soup.find('table', id=table_id)
        if not table:
            # Try to find the table inside HTML comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment_soup = BeautifulSoup(comment, 'lxml')
                table = comment_soup.find('table', id=table_id)
                if table:
                    logger.info(f"Found table '{table_id}' inside HTML comment for year {year}")
                    break

        if not table:
            logger.error(f"Table with id '{table_id}' not found on {url}")
            return pd.DataFrame()

        # Extract columns from thead elements
        columns = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]

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

        return pd.DataFrame(rows, columns=columns)

    def scrape_player_fantasy_stats(self, year: int) -> None:
        """
        Scrape fantasy stats for a specific season.

        Args:
            year: NFL season year (e.g., 2023)

        Returns:
            None (saves data to bronze layer)
        """
        url = f"{self.base_url}/years/{year}/fantasy.htm"

        df = self.scrape_html_table(url, f"{year}_player_fantasy_stats", "fantasy", year)

        if not df.empty:
            output_path = os.path.join(self.bronze_data_dir, f"{year}_player_fantasy_stats.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved fantasy stats for {year} to {output_path}")

    def scrape_player_offensive_stats(self, year: int, category: str) -> None:
        """
        Scrape player receiving stats for a given year.

        Args:
            year: NFL season year
            category: The category of offensive stats to scrape, one of:
                - "passing"
                - "rushing"
                - "rushing_advanced"
                - "receiving"
                - "receiving_advanced"

        Returns:
            None (saves data to bronze layer)
        """
        assert category in ["passing", "rushing", "receiving", "rushing_advanced", "receiving_advanced"], "Invalid category"

        url = f"{self.base_url}/years/{year}/{category}.htm"
        html_table_file = f"{year}_{category}_stats"

        if category == "passing":
            df = self.scrape_html_table(
                url=url,
                html_table_file=html_table_file,
                table_id="passing",
                year=year)
        elif category == "rushing":
            df = self.scrape_html_table(
                url=url,
                html_table_file=html_table_file,
                table_id="rushing",
                year=year)
        elif category == "rushing_advanced":
            df = self.scrape_html_table(
                url=url,
                html_table_file=html_table_file,
                table_id="adv_rushing",
                year=year)
        elif category == "receiving":
            df = self.scrape_html_table(
                url=url,
                html_table_file=html_table_file,
                table_id="receiving",
                year=year)
        elif category == "receiving_advanced":
            df = self.scrape_html_table(
                url=url,
                html_table_file=html_table_file,
                table_id="adv_receiving", year=year)
        else:
            logger.error(f"Invalid category: {category}")

        if not df.empty:
            output_path = os.path.join(self.bronze_data_dir, f"{year}_player_{category}_stats.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {category} stats for {year} to {output_path}")

        return df

    def scrape_team_offensive_stats(self, year: int) -> None:
        """
        Scrape multiple team offensive stats tables for a specific season.
        Tables scraped: team_offense, passing_offense, rushing_offense.

        Args:
            year: NFL season year

        Returns:
            None (saves data to bronze layer)
        """
        url = f"{self.base_url}/years/{year}/#team_stats"

        df = self.scrape_html_table(url, f"{year}_team_offense", "team_stats", year)

        if not df.empty:
            output_path = os.path.join(self.bronze_data_dir, f"{year}_team_offense.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved team offense stats for {year} to {output_path}")

    def scrape_years(self, start_year: int, end_year: int):
        """
        Scrape data for multiple years.

        Args:
            start_year: First year to scrape
            end_year: Last year to scrape
        """
        years = range(start_year, end_year + 1)

        for year in tqdm(years, desc="Scraping years"):
            logger.info(f"Scraping data from {year}")

            self.scrape_player_fantasy_stats(year)

            self.scrape_player_offensive_stats(year, "passing")
            self.scrape_player_offensive_stats(year, "rushing")
            self.scrape_player_offensive_stats(year, "rushing_advanced")
            self.scrape_player_offensive_stats(year, "receiving")
            self.scrape_player_offensive_stats(year, "receiving_advanced")

            self.scrape_team_offensive_stats(year)

            logger.info(f"Completed scraping data from {year}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrapes stats from external sources"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1999,
        help="Start year to scrape (default: 1999)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year to scrape (default: 2024)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory to save scraped data, paths are relative to this script (default: ../data)"
    )

    args = parser.parse_args()

    scraper = ProFootballReferenceScraper(data_dir=args.data_dir)

    scraper.scrape_years(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
