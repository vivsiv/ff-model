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
    """Raw data scraper for pro football reference's website."""

    def __init__(self, data_dir: str = "../data/pfr"):
        """
        Initialize the scraper.

        Args:
            data_dir: Root directory to save scraped data
        """
        self.base_url = "https://www.pro-football-reference.com"

        self.data_dir = data_dir 
        self.html_dir = os.path.join(data_dir, "html")
        self.bronze_dir = os.path.join(data_dir, "bronze")

        for d in [self.data_dir, self.html_dir, self.bronze_dir]:
            os.makedirs(d, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _get_html_page_soup(self, 
        url: str,
        html_page_path: str,
        delay: float = 3.0,
        overwrite: bool = False) -> BeautifulSoup:
        """
        Send a web request for a url.

        Args:
            url: URL to scrape
            html_page_path: Fully qualified path that the scraped html page will be saved to.
            delay: Time (in seconds) to wait between requests.
            overwrite: Whether or not to overwrite an existing html file. (default: False)

        Returns:
            A BeautifulSoup object wrapping the response for the requested url.
        """

        if os.path.exists(html_page_path) and not overwrite:
            logger.info(f"{url} already saved at {html_page_path}. skipping...")
            with open(html_page_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return BeautifulSoup(html_content, 'lxml')

        time.sleep(delay + random.uniform(0.5, 1.5))

        try:
            logger.info(f"Requesting {url}")
            response = self.session.get(url)
            response.raise_for_status()

            with open(html_page_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            return BeautifulSoup(response.text, 'lxml')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def scrape_html_table(self,
        html_page_soup: BeautifulSoup,
        table_id: str,
        year: int) -> pd.DataFrame:
        """
        Scrape an HTML table object from an html page.

        Args:
            html_page_soup: BeautifulSoup object wrapping an html page.
            table_id: ID of the table to scrape
            year: Year of the data

        Returns:
            DataFrame with the scraped table data
        """
        table = html_page_soup.find('table', id=table_id)
        if not table:
            # Try to find the table inside HTML comments
            for comment in html_page_soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment_soup = BeautifulSoup(comment, 'lxml')
                table = comment_soup.find('table', id=table_id)
                if table:
                    logger.info(f"Found table '{table_id}' inside HTML comment for year {year}")
                    break

        if not table:
            logger.error(f"Table with id '{table_id}' not found.")
            return pd.DataFrame()

        # Extract columns from thead elements
        columns = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]

        # Extract rows from tbody elements
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            # Skip column names
            if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
                continue
            row = [td.get_text(strip=True) for td in tr.find_all(['th', 'td'])]
            rows.append(row)

        if len(rows) > 0:
            # Adjust columns to match whats in the data, this ignores category headers that precede the actual data.
            header_column_count = len(columns)
            data_column_count = len(rows[0])
            # If we have more headers than actual columns, take the last N header columns
            if header_column_count > data_column_count:
                columns = columns[-data_column_count:]
                logger.info(f"Adjusted headers from {header_column_count} to {data_column_count}")
            elif header_column_count < data_column_count:
                # Add dummy column names for extra data columns
                dummy_col_count = data_column_count - header_column_count
                for i in range(dummy_col_count):
                    columns.append(f"Unknown_Col_{i+1}")
                logger.info(f"Added {dummy_col_count} dummy columns")
            else:
                logger.info(f"Headers and data columns match for {year}")
        else:
            logger.warning(f"No data rows found for {year}")
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=columns)

    def scrape_player_fantasy_stats(self, year: int, overwrite: bool = False) -> pd.DataFrame:
        """
        Scrape a player's fantasy stats for a specific season and save them to a file
        in the bronze layer.

        Args:
            year: NFL season (e.g., 2023)
            overwrite: Overwrite the existing html file (default: False)

        Returns:
            DataFrame containing the players stats.
        """
        url = f"{self.base_url}/years/{year}/fantasy.htm"
        html_page_path = os.path.join(self.html_dir, f"{year}_player_fantasy_stats.html")
        html_page_soup = self._get_html_page_soup(url, html_page_path, overwrite=overwrite)

        df = pd.DataFrame()
        if html_page_soup:
            df = self.scrape_html_table(html_page_soup=html_page_soup, table_id="fantasy", year=year)
        else:
            logger.error(f"Failed to get player fantasy data from: {url}")

        if not df.empty:
            output_path = os.path.join(self.bronze_dir, f"{year}_player_fantasy_stats.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved fantasy stats for {year} to {output_path}")

        return df

    def scrape_player_offensive_stats(self, year: int, category: str, overwrite: bool = False) -> pd.DataFrame:
        """
        Scrape player receiving stats for a specific season and save them to a file
        in the bronze layer.

        Args:
            year: NFL season year
            category: The category of offensive stats to scrape, one of:
                - "passing"
                - "rushing"
                - "rushing_advanced"
                - "receiving"
                - "receiving_advanced"
            overwrite: Overwrite the existing html file (default: False)

        Returns:
            DataFrame containing the player's stats.
        """

        category_to_table_id = {
            "passing": "passing",
            "rushing": "rushing",
            "receiving": "receiving",
            "rushing_advanced": "adv_rushing",
            "receiving_advanced": "adv_receiving"
        }

        assert category in category_to_table_id.keys(), "Invalid Category"
        table_id = category_to_table_id[category]

        url = f"{self.base_url}/years/{year}/{category}.htm"
        html_page_path = os.path.join(self.html_dir, f"{year}_{category}_stats.html")
        html_page_soup = self._get_html_page_soup(url, html_page_path, overwrite=overwrite)

        df = pd.DataFrame()
        if html_page_soup:
            df = self.scrape_html_table(html_page_soup=html_page_soup, table_id=table_id, year=year)
        else:
            logger.error(f"Failed to get player {category} data from: {url}")

        if not df.empty:
            output_path = os.path.join(self.bronze_dir, f"{year}_player_{category}_stats.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {category} stats for {year} to {output_path}")

        return df

    def scrape_team_offensive_stats(self, year: int, overwrite: bool = False) -> pd.DataFrame:
        """
        Scrape multiple team's offensive stats for a specific season and save them to a file
        in the bronze layer.
            - Tables scraped: team_offense, passing_offense, rushing_offense.

        Args:
            year: NFL season year
            overwrite: Overwrite the existing html file (default: False)

        Returns:
            DataFrame containing the player's stats.
        """
        url = f"{self.base_url}/years/{year}/#team_stats"
        html_page_path = os.path.join(self.html_dir, f"{year}_team_offense.html")
        html_page_soup = self._get_html_page_soup(url, html_page_path, overwrite=overwrite)

        df = pd.DataFrame()
        if html_page_soup:
            df = self.scrape_html_table(html_page_soup=html_page_soup, table_id="team_stats", year=year)
        else:
            logger.error(f"Failed to get team data from: {url}")

        if not df.empty:
            output_path = os.path.join(self.bronze_dir, f"{year}_team_offense.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved team offense stats for {year} to {output_path}")

        return df

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

            for cat in ["passing", "rushing", "rushing_advanced", "receiving", "receiving_advanced"]:
                self.scrape_player_offensive_stats(year, category=cat)

            self.scrape_team_offensive_stats(year)

            logger.info(f"Completed scraping data from {year}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrapes stats from external sources"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2023,
        help="Start year to scrape (default: 2023)"
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
        default=None,
        help="Directory to save scraped data."
    )

    args = parser.parse_args()

    kwargs = {"data_dir": args.data_dir} if args.data_dir is not None else {}
    scraper = ProFootballReferenceScraper(**kwargs)

    scraper.scrape_years(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
