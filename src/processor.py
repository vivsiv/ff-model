#!/usr/bin/env python3
"""
NFL Fantasy Football Data Preprocessing

This module provides functionality to clean, transform, and combine
the raw data scraped from Pro Football Reference.
"""

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np

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
    """Generate silver and gold layer data from bronze layer data."""
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing the scraped data
        """
        self.bronze_data_dir = os.path.join(data_dir, "bronze")
        self.silver_data_dir = os.path.join(data_dir, "silver")
        self.gold_data_dir = os.path.join(data_dir, "gold")
        
        os.makedirs(self.silver_data_dir, exist_ok=True)
        os.makedirs(self.gold_data_dir, exist_ok=True)
    
    def standardize_name(self, name: str) -> str:
        """
        Standardizes the name of a player by:
        - Removing any extra spaces
        - Making the name fully lowercase
        - Removing any dots i.e. (A.J. Brown -> aj brown)
        - Removing any suffixes suggesting awards or honors (*, +)
        - Removing any suffixes i.e. (Kenneth Walker III -> kenneth walker) or (Odell Beckham Jr. -> odell beckham)
        
        Args:
            name: The name of the player
            
        Returns:
            Standardized name
        """
        name = name.strip()
        name = name.lower()
        name = name.replace('.', '')

        suffixes = {"jr", "sr", "ii", "iii", "iv", "v", "junior", "senior", "*", "+"}
        parts = name.split()
        while parts and parts[-1] in suffixes:
            parts.pop()
        name = " ".join(parts)

        return name
    
    def parse_awards(self, awards: str) -> List[str]:
        """
        Parses the awards column and returns a list of awards.
        """
        # TODO: fix
        if pd.isna(awards):
            return []
        return awards.split(',')
    
    def combine_year_data(self,
                          file_pattern: str,
                          select_columns: List[str],
                          transformations: Dict[str, Callable],
                          rename_columns: Dict[str, str],
                          output_file_name: str) -> pd.DataFrame:
        """
        Combines data from multiple years into a single dataframe.
        """
        file_pattern = os.path.join(self.bronze_data_dir, file_pattern)
        files = glob.glob(file_pattern)

        if not files:
            logger.error(f"No files found matching {file_pattern}")
            return pd.DataFrame()
        
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            df = df[df['Tm'] != 'League Average']
            df['Year'] = int(file.split('_')[0])
            df = df.select(columns=select_columns)

            for column, func in transformations.items():
                if column in df.columns:
                    df[column] = df[column].apply(func)
                else:
                    logger.warning(f"Column '{column}' not found in dataframe, skipping transformation.")
            
            df = df.rename(columns=rename_columns)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)

        table_path = os.path.join(self.silver_data_dir, output_file_name)
        combined_df.to_csv(table_path, index=False)
        logger.info(f"Saved {output_file_name} to {table_path}")


    def build_player_fantasy_stats(self) -> None:
        """
        Reads in all years of fantasy stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.

        Returns:
            None (saves data to silver layer)
        """
        select_columns = ['Player', 'Tm', 'FantPos', 'Age', 'G', 'GS', 'FantPt', 'PPR','Year']
        transformations = {'Player': self.standardize_name}
        rename_columns = {'Player': 'name', 'Tm': 'team', 'FantPos': 'position', 'Age': 'age', 'G': 'games', 'GS': 'games_started', 'FantPt': 'standard_points', 'PPR': 'ppr_points', 'Year': 'year'}
        output_file_name = "player_fantasy_stats.csv"

        self.combine_year_data(
            file_pattern="*_player_fantasy_stats.csv",
            select_columns=select_columns,
            transformations=transformations,
            rename_columns=rename_columns,
            output_file_name=output_file_name
        )
    
    def build_player_receiving_stats(self) -> None:
        """
        Reads in all years of receiving stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        select_columns = ['Player', 'Tm', 'Age', 'Pos', 'Year', 'Tgt', 'Rec', 'Yds', 'TD', '1D', 'YBC', 'YBC/R', 'YAC', 'YAC/R', 'ADOT', 'BrkTkl', 'Rec/Br', 'Drop%', 'Int', 'Rat', 'Awards']
        transformations = {'Player': self.standardize_name, 'Awards': self.parse_awards}
        rename_columns = {
            'Player': 'name',
            'Tm': 'team',
            'Age': 'age',
            'Pos': 'position',
            'Year': 'year',
            'Tgt': 'targets',
            'Rec': 'receptions',
            'Yds': 'receiving_yards',
            'TD': 'receiving_touchdowns',
            '1D': 'receiving_first_downs',
            'YBC': 'catch_air_yards',
            'YBC/R': 'catch_air_yards_per_reception',
            'YAC': 'yards_after_catch',
            'YAC/R': 'yards_after_catch_per_reception',
            'ADOT': 'average_depth_of_target',
            'BrkTkl': 'receiving_broken_tackles',
            'Rec/Br': 'receptions_per_broken_tackle',
            'Drop%': 'drop_percentage',
            'Int': 'interceptions_when_targeted',
            'Rat': 'passer_rating_when_targeted',
            'Awards': 'awards'
        }
        output_file_name = "player_receiving_stats.csv"
        
        self.combine_year_data(
            file_pattern="*_player_receiving_stats.csv",
            select_columns=select_columns,
            transformations=transformations,
            rename_columns=rename_columns,
            output_file_name=output_file_name
        )
    
    def build_player_rushing_stats(self) -> None:
        """
        Reads in all years of rushing stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        select_columns = ['Player', 'Tm', 'Age', 'Pos', 'Year', 'Att', 'Yds', '1D', 'YBC', 'YBC/ATT', 'YAC', 'YAC/Att', 'BrkTkl', 'Att/Br', 'Awards']
        transformations = {'Player': self.standardize_name, 'Awards': self.parse_awards}
        rename_columns = {
            'Player': 'name',
            'Tm': 'team',
            'Age': 'age',
            'Pos': 'position',
            'Year': 'year',
            'Att': 'attempts',
            'Yds': 'rushing_yards',
            '1D': 'rushing_first_downs',
            'YBC': 'rush_yards_before_contact',
            'YBC/ATT': 'rush_yards_before_contact_per_attempt',
            'YAC': 'rush_yards_after_contact',
            'YAC/Att': 'rush_yards_after_contact_per_attempt',
            'BrkTkl': 'rush_broken_tackles',
            'Att/Br': 'rush_attempts_per_broken_tackle',
            'Awards': 'awards'
        }
        output_file_name = "player_rushing_stats.csv"
        
        self.combine_year_data(
            file_pattern="*_player_rushing_stats.csv",
            select_columns=select_columns,
            transformations=transformations,
            rename_columns=rename_columns,
            output_file_name=output_file_name
        )
    
    def build_player_passing_stats(self) -> None:
        """
        Reads in all years of passing stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        select_columns = ['Player', 'Tm', 'Age', 'Pos', 'Year', 'Cmp', 'Att', 'IAY', 'IAY/PA', 'CAY', 'CAY/Cmp', 'CAY/PA', 'YAC', 'YAC/Cmp', 'ThAway', 'Drop%', 'BadTh%', 'OnTgt%', 'PktTime', 'Bltz', 'Hrry', 'Hits', 'Prss%', 'Scrm', 'Yds/Scrm', 'Awards']
        transformations = {'Player': self.standardize_name}
        rename_columns = {
            'Player': 'name',
            'Tm': 'team',
            'Age': 'age',
            'Pos': 'position',
            'Year': 'year',
            'Cmp': 'completions',
            'Att': 'attempts',
            'IAY': 'incomplete_air_yards',
            'IAY/PA': 'incomplete_air_yards_per_attempt',
            'CAY': 'completed_air_yards',
            'CAY/Cmp': 'completed_air_yards_per_completion',
            'CAY/PA': 'completed_air_yards_per_attempt',
            'YAC': 'yards_after_catch',
            'YAC/Cmp': 'yards_after_catch_per_completion',
            'ThAway': 'passes_thrown_away',
            'Drop%': 'drop_percentage',
            'BadTh%': 'bad_throw_percentage',
            'OnTgt%': 'on_target_percentage',
            'PktTime': 'pocket_time',
            'Bltz': 'blitz_count',
            'Hrry': 'hurry_count',
            'Hits': 'hit_count',
            'Prss%': 'pressure_percentage',
            'Scrm': 'scramble_count',
            'Yds/Scrm': 'yards_per_scramble',
            'Awards': 'awards'
        }
        output_file_name = "player_passing_stats.csv"

        self.combine_year_data(
            file_pattern="*_player_passing_stats.csv",
            select_columns=select_columns,
            transformations=transformations,
            rename_columns=rename_columns,
            output_file_name=output_file_name
        )


    def build_team_stats(self) -> None:
        """
        Reads in all years of team stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.

        Returns:
            None (saves data to silver layer)
        """
        # TODO: fix
        select_columns = ['Tm', 'G', 'Year', 'Yds', 'Ply', 'Y/P', 'TO', 'FL',  'PassYds', 'PassTD', 'PassInt', 'RushYds', 'RushTD', 'Rec', 'RecYds', 'RecTD', 'Fmb', 'Year']
        rename_columns = {'Tm': 'team', 'PassYds': 'passing_yards', 'PassTD': 'passing_touchdowns', 'PassInt': 'passing_interceptions', 'RushYds': 'rushing_yards', 'RushTD': 'rushing_touchdowns', 'Rec': 'receptions', 'RecYds': 'receiving_yards', 'RecTD': 'receiving_touchdowns', 'Fmb': 'fumbles', 'Year': 'year'}
        output_file_name = "team_offense.csv"

        self.combine_year_data(
            file_pattern="*_team_offense.csv",
            select_columns=select_columns,
            transformations={},
            rename_columns=rename_columns,
            output_file_name=output_file_name
        )


    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process all data and create combined datasets.
        
        Args:
            years: List of years to process, or None for all available years
            
        Returns:
            Dictionary of processed DataFrames
        """
        self.build_player_fantasy_stats()
        self.build_player_receiving_stats()
        self.build_player_rushing_stats()
        self.build_player_passing_stats()
        self.build_team_stats()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    processor = FantasyDataProcessor(data_dir=data_dir)
    
    processor.process_all_data()
    logger.info("Data processing complete")


if __name__ == "__main__":
    main() 