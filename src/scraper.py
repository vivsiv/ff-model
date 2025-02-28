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
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# Configure logging
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
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    
    def _get_soup(self, url: str, delay: float = 3.0) -> BeautifulSoup:
        """
        Get BeautifulSoup object from URL with rate limiting.
        
        Args:
            url: URL to scrape
            delay: Time to wait between requests (seconds)
            
        Returns:
            BeautifulSoup object
        """
        # Add jitter to delay to avoid detection
        time.sleep(delay + random.uniform(0.5, 1.5))
        
        try:
            logger.info(f"Requesting {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            # Save raw HTML
            url_path = url.replace(self.base_url, "").replace("/", "_").strip("_")
            if not url_path:
                url_path = "index"
            raw_path = os.path.join(self.data_dir, "raw", f"{url_path}.html")
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
        
        # Extract table headers
        headers = [th.text for th in table.find('thead').find_all('th')]
        
        # Extract table rows
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            # Skip header rows
            if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
                continue
                
            row = []
            for td in tr.find_all(['th', 'td']):
                # Extract player ID from links
                if td.find('a') and 'href' in td.find('a').attrs:
                    href = td.find('a')['href']
                    if '/players/' in href:
                        player_id = href.split('/')[-1].split('.')[0]
                        row.append(player_id)
                row.append(td.text.strip())
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Clean up column names and data
        if not df.empty:
            # Add year column
            df['Season'] = year
            
            # Save to CSV
            output_path = os.path.join(self.data_dir, "processed", f"fantasy_stats_{year}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved fantasy stats for {year} to {output_path}")
            
        return df
    
    def scrape_player_game_logs(self, player_id: str, year: int) -> pd.DataFrame:
        """
        Scrape game logs for a specific player and season.
        
        Args:
            player_id: Player ID from Pro Football Reference
            year: NFL season year
            
        Returns:
            DataFrame with player game logs
        """
        # Construct URL based on player ID format (first letter/player_id)
        first_letter = player_id[0]
        url = f"{self.base_url}/players/{first_letter}/{player_id}/gamelog/{year}/"
        
        soup = self._get_soup(url)
        
        if not soup:
            logger.error(f"Failed to get game logs for player {player_id} in {year}")
            return pd.DataFrame()
        
        # Find the game log table
        table = soup.find('table', id='gamelog')
        if not table:
            logger.warning(f"Game log table not found for player {player_id} in {year}")
            return pd.DataFrame()
        
        # Extract player name
        player_name = soup.find('h1', {'itemprop': 'name'})
        if player_name:
            player_name = player_name.text.strip()
        else:
            player_name = "Unknown"
        
        # Convert table to DataFrame
        try:
            dfs = pd.read_html(str(table))
            if not dfs:
                return pd.DataFrame()
            
            df = dfs[0]
            
            # Clean up DataFrame
            # Remove multi-level column headers if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Remove rows that are headers or have NaN in date column
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
        except Exception as e:
            logger.error(f"Error processing game logs for {player_id} in {year}: {e}")
            return pd.DataFrame()
    
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
        
        # Find the team stats table
        table = soup.find('table', id='team_stats')
        if not table:
            logger.error(f"Team stats table not found for {year}")
            return pd.DataFrame()
        
        # Convert table to DataFrame
        try:
            dfs = pd.read_html(str(table))
            if not dfs:
                return pd.DataFrame()
            
            df = dfs[0]
            
            # Add year column
            df['Season'] = year
            
            # Save to CSV
            output_path = os.path.join(self.data_dir, "processed", f"team_stats_{year}.csv")
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
        
        # Find the advanced team stats table
        table = soup.find('table', id='team_stats')
        if not table:
            logger.error(f"Advanced team stats table not found for {year}")
            return pd.DataFrame()
        
        # Convert table to DataFrame
        try:
            dfs = pd.read_html(str(table))
            if not dfs:
                return pd.DataFrame()
            
            df = dfs[0]
            
            # Add year column
            df['Season'] = year
            
            # Save to CSV
            output_path = os.path.join(self.data_dir, "processed", f"advanced_team_stats_{year}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved advanced team stats for {year} to {output_path}")
            
            return df
        except Exception as e:
            logger.error(f"Error processing advanced team stats for {year}: {e}")
            return pd.DataFrame()
    
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

    def scrape_multiple_years(self, start_year: int, end_year: int, include_game_logs: bool = True):
        """
        Scrape data for multiple years.
        
        Args:
            start_year: First year to scrape
            end_year: Last year to scrape
            include_game_logs: Whether to scrape game logs for relevant players
        """
        years = range(start_year, end_year + 1)
        
        for year in tqdm(years, desc="Scraping years"):
            logger.info(f"Scraping data for {year}")
            
            # Scrape season fantasy stats
            self.scrape_season_fantasy_stats(year)
            
            # Scrape team stats
            self.scrape_team_stats(year)
            self.scrape_advanced_team_stats(year)
            
            # Scrape game logs for relevant players
            if include_game_logs:
                player_ids = self.get_player_ids_for_year(year)
                logger.info(f"Found {len(player_ids)} relevant players for {year}")
                
                for player_id in tqdm(player_ids, desc=f"Scraping players for {year}"):
                    self.scrape_player_game_logs(player_id, year)
            
            logger.info(f"Completed scraping for {year}")


def main():
    """Main function to run the scraper."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set data directory relative to script directory
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Create scraper
    scraper = ProFootballReferenceScraper(data_dir=data_dir)
    
    # Scrape last 10 years of data
    current_year = datetime.now().year
    start_year = current_year - 10
    end_year = current_year - 1  # Last complete season
    
    logger.info(f"Starting scrape for years {start_year} to {end_year}")
    scraper.scrape_multiple_years(start_year, end_year)
    logger.info("Scraping complete")


if __name__ == "__main__":
    main() 