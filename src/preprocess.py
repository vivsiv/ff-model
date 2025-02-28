#!/usr/bin/env python3
"""
NFL Fantasy Football Data Preprocessing

This module provides functionality to clean, transform, and combine
the raw data scraped from Pro Football Reference.
"""

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FantasyDataProcessor:
    """Process and combine fantasy football data."""
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing the scraped data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.output_dir = os.path.join(data_dir, "final")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_fantasy_stats(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load fantasy stats for specified years.
        
        Args:
            years: List of years to load, or None for all available years
            
        Returns:
            DataFrame with combined fantasy stats
        """
        # Get all fantasy stats files
        pattern = os.path.join(self.processed_dir, "fantasy_stats_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"No fantasy stats files found matching {pattern}")
            return pd.DataFrame()
        
        # Filter by years if specified
        if years:
            files = [f for f in files if int(f.split('_')[-1].split('.')[0]) in years]
        
        # Load and combine all files
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                year = int(file.split('_')[-1].split('.')[0])
                if 'Season' not in df.columns:
                    df['Season'] = year
                dfs.append(df)
                logger.info(f"Loaded fantasy stats from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if not dfs:
            logger.error("No fantasy stats data loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined fantasy stats: {len(combined_df)} rows")
        
        return combined_df
    
    def load_game_logs(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load game logs for specified years.
        
        Args:
            years: List of years to load, or None for all available years
            
        Returns:
            DataFrame with combined game logs
        """
        # Get all game log files
        pattern = os.path.join(self.processed_dir, "game_logs_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"No game log files found matching {pattern}")
            return pd.DataFrame()
        
        # Filter by years if specified
        if years:
            files = [f for f in files if any(str(year) in f for year in years)]
        
        # Load and combine all files
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                logger.info(f"Loaded game logs from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if not dfs:
            logger.error("No game log data loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined game logs: {len(combined_df)} rows")
        
        return combined_df
    
    def load_team_stats(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load team stats for specified years.
        
        Args:
            years: List of years to load, or None for all available years
            
        Returns:
            DataFrame with combined team stats
        """
        # Get all team stats files
        pattern = os.path.join(self.processed_dir, "team_stats_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"No team stats files found matching {pattern}")
            return pd.DataFrame()
        
        # Filter by years if specified
        if years:
            files = [f for f in files if int(f.split('_')[-1].split('.')[0]) in years]
        
        # Load and combine all files
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                year = int(file.split('_')[-1].split('.')[0])
                if 'Season' not in df.columns:
                    df['Season'] = year
                dfs.append(df)
                logger.info(f"Loaded team stats from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if not dfs:
            logger.error("No team stats data loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined team stats: {len(combined_df)} rows")
        
        return combined_df
    
    def load_advanced_team_stats(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load advanced team stats for specified years.
        
        Args:
            years: List of years to load, or None for all available years
            
        Returns:
            DataFrame with combined advanced team stats
        """
        # Get all advanced team stats files
        pattern = os.path.join(self.processed_dir, "advanced_team_stats_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"No advanced team stats files found matching {pattern}")
            return pd.DataFrame()
        
        # Filter by years if specified
        if years:
            files = [f for f in files if int(f.split('_')[-1].split('.')[0]) in years]
        
        # Load and combine all files
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                year = int(file.split('_')[-1].split('.')[0])
                if 'Season' not in df.columns:
                    df['Season'] = year
                dfs.append(df)
                logger.info(f"Loaded advanced team stats from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if not dfs:
            logger.error("No advanced team stats data loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined advanced team stats: {len(combined_df)} rows")
        
        return combined_df
    
    def clean_fantasy_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize fantasy stats.
        
        Args:
            df: DataFrame with fantasy stats
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Standardize column names
        # This will need to be adjusted based on the actual column names in the data
        column_mapping = {
            'Rk': 'Rank',
            'Player': 'Player_Name',
            'Tm': 'Team',
            'FantPt': 'Fantasy_Points',
            'PPR': 'PPR_Points',
            'DKPt': 'DraftKings_Points',
            'FDPt': 'FanDuel_Points',
            'VBD': 'Value_Based_Draft',
            'PosRank': 'Position_Rank',
            'OvRank': 'Overall_Rank',
            'Season': 'Season'
        }
        
        # Rename columns that exist in the DataFrame
        existing_columns = set(df.columns).intersection(set(column_mapping.keys()))
        rename_dict = {col: column_mapping[col] for col in existing_columns}
        df = df.rename(columns=rename_dict)
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            try:
                # Remove commas and convert to numeric
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            except:
                pass
        
        # Ensure player_id column exists
        if 'player_id' in df.columns:
            df = df.rename(columns={'player_id': 'Player_ID'})
        
        return df
    
    def clean_game_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize game logs.
        
        Args:
            df: DataFrame with game logs
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle multi-level column names if present
        if any('_' in col for col in df.columns):
            # Already processed
            pass
        elif isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # Convert date column to datetime
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                logger.warning("Could not convert Date column to datetime")
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            try:
                # Skip non-numeric columns
                if col in ['Player_Name', 'Team', 'Opp', 'Date', 'Player_ID']:
                    continue
                # Remove commas and convert to numeric
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            except:
                pass
        
        return df
    
    def clean_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize team stats.
        
        Args:
            df: DataFrame with team stats
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            try:
                # Skip team name column
                if col in ['Tm', 'Team']:
                    continue
                # Remove commas and convert to numeric
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            except:
                pass
        
        return df
    
    def calculate_fantasy_points(self, df: pd.DataFrame, scoring_system: str = 'standard') -> pd.DataFrame:
        """
        Calculate fantasy points based on different scoring systems.
        
        Args:
            df: DataFrame with player stats
            scoring_system: Scoring system to use ('standard', 'ppr', 'half_ppr')
            
        Returns:
            DataFrame with fantasy points added
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Define scoring systems
        scoring = {
            'standard': {
                'pass_yds': 0.04,  # 1 point per 25 yards
                'pass_td': 4,
                'pass_int': -2,
                'rush_yds': 0.1,   # 1 point per 10 yards
                'rush_td': 6,
                'rec': 0,          # 0 points per reception
                'rec_yds': 0.1,    # 1 point per 10 yards
                'rec_td': 6,
                'fumbles_lost': -2,
                '2pt': 2
            },
            'ppr': {
                'pass_yds': 0.04,
                'pass_td': 4,
                'pass_int': -2,
                'rush_yds': 0.1,
                'rush_td': 6,
                'rec': 1,          # 1 point per reception
                'rec_yds': 0.1,
                'rec_td': 6,
                'fumbles_lost': -2,
                '2pt': 2
            },
            'half_ppr': {
                'pass_yds': 0.04,
                'pass_td': 4,
                'pass_int': -2,
                'rush_yds': 0.1,
                'rush_td': 6,
                'rec': 0.5,        # 0.5 points per reception
                'rec_yds': 0.1,
                'rec_td': 6,
                'fumbles_lost': -2,
                '2pt': 2
            }
        }
        
        # Get the selected scoring system
        if scoring_system not in scoring:
            logger.warning(f"Unknown scoring system: {scoring_system}, using standard")
            scoring_system = 'standard'
        
        points = scoring[scoring_system]
        
        # Calculate fantasy points based on available columns
        # This will need to be adjusted based on the actual column names in the data
        try:
            # Initialize fantasy points column
            df[f'{scoring_system.upper()}_Fantasy_Points'] = 0
            
            # Add points for passing
            if 'Passing_Yds' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Passing_Yds'] * points['pass_yds']
            if 'Passing_TD' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Passing_TD'] * points['pass_td']
            if 'Int' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Int'] * points['pass_int']
            
            # Add points for rushing
            if 'Rushing_Yds' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Rushing_Yds'] * points['rush_yds']
            if 'Rushing_TD' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Rushing_TD'] * points['rush_td']
            
            # Add points for receiving
            if 'Rec' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Rec'] * points['rec']
            if 'Receiving_Yds' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Receiving_Yds'] * points['rec_yds']
            if 'Receiving_TD' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Receiving_TD'] * points['rec_td']
            
            # Add points for miscellaneous
            if 'Fumbles_Lost' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['Fumbles_Lost'] * points['fumbles_lost']
            if '2PM' in df.columns:
                df[f'{scoring_system.upper()}_Fantasy_Points'] += df['2PM'] * points['2pt']
            
            logger.info(f"Calculated {scoring_system} fantasy points")
        except Exception as e:
            logger.error(f"Error calculating fantasy points: {e}")
        
        return df
    
    def process_all_data(self, years: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
        """
        Process all data and create combined datasets.
        
        Args:
            years: List of years to process, or None for all available years
            
        Returns:
            Dictionary of processed DataFrames
        """
        # Load all data
        fantasy_stats = self.load_fantasy_stats(years)
        game_logs = self.load_game_logs(years)
        team_stats = self.load_team_stats(years)
        advanced_team_stats = self.load_advanced_team_stats(years)
        
        # Clean data
        fantasy_stats = self.clean_fantasy_stats(fantasy_stats)
        game_logs = self.clean_game_logs(game_logs)
        team_stats = self.clean_team_stats(team_stats)
        advanced_team_stats = self.clean_team_stats(advanced_team_stats)
        
        # Calculate fantasy points for different scoring systems
        for scoring in ['standard', 'ppr', 'half_ppr']:
            fantasy_stats = self.calculate_fantasy_points(fantasy_stats, scoring)
            game_logs = self.calculate_fantasy_points(game_logs, scoring)
        
        # Save processed data
        if not fantasy_stats.empty:
            output_path = os.path.join(self.output_dir, "fantasy_stats.csv")
            fantasy_stats.to_csv(output_path, index=False)
            logger.info(f"Saved processed fantasy stats to {output_path}")
        
        if not game_logs.empty:
            output_path = os.path.join(self.output_dir, "game_logs.csv")
            game_logs.to_csv(output_path, index=False)
            logger.info(f"Saved processed game logs to {output_path}")
        
        if not team_stats.empty:
            output_path = os.path.join(self.output_dir, "team_stats.csv")
            team_stats.to_csv(output_path, index=False)
            logger.info(f"Saved processed team stats to {output_path}")
        
        if not advanced_team_stats.empty:
            output_path = os.path.join(self.output_dir, "advanced_team_stats.csv")
            advanced_team_stats.to_csv(output_path, index=False)
            logger.info(f"Saved processed advanced team stats to {output_path}")
        
        return {
            'fantasy_stats': fantasy_stats,
            'game_logs': game_logs,
            'team_stats': team_stats,
            'advanced_team_stats': advanced_team_stats
        }


def main():
    """Main function to run the data processor."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set data directory relative to script directory
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Create processor
    processor = FantasyDataProcessor(data_dir=data_dir)
    
    # Process all available data
    processor.process_all_data()
    logger.info("Data processing complete")


if __name__ == "__main__":
    main() 