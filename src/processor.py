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
from tqdm import tqdm

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
        name = name.replace('*', '')
        name = name.replace('+', '')
        name = name.replace('\'', '')
        name = name.replace("-", "_")

        suffixes = {"jr", "sr", "ii", "iii", "iv", "v", "junior", "senior"}
        parts = name.split()
        while parts and parts[-1] in suffixes:
            parts.pop()
        name = "_".join(parts)

        return name

    def standardize_team_name(self, team: str) -> str:
        """
        Team stats have full team names: "Philadelphia Eagles"
        Player stats have abbreviated team names: "PHI"
        Strategy:
        - In general take the city and shorten it to the first 3 letters and capitalize it
        - Use a hardcoded list of exceptions to handle cases like "New York Giants" -> "NYG"
        """
        team = team.strip()
        exceptions = {
            "Green Bay Packers": "GNB",
            "Las Vegas Raiders": "LVR",
            "Los Angeles Rams": "LAR",
            "Los Angeles Chargers": "LAC",
            "Jacksonville Jaguars": "JAX",
            "New York Giants": "NYG",
            "New York Jets": "NYJ",
            "New England Patriots": "NWE",
            "New Orleans Saints": "NOR",
            "San Francisco 49ers": "SFO",
        }
        if team in exceptions:
            team = exceptions[team]
        else:
            team = team.split()[0][:3].upper()

        return team

    def parse_awards(self, awards: str) -> int:
        """
        Parses the awards column and returns the number of awards.
        """
        if pd.isna(awards):
            return 0.0
        return len(awards.split(','))

    def combine_year_data(self,
                          file_pattern: str,
                          column_names: List[str],
                          select_columns: List[str],
                          transformations: Dict[str, Callable],
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
        for file in tqdm(files, desc=f"Processing files matching: {file_pattern.split('/')[-1]}"):
            df = pd.read_csv(file)
            assert len(column_names) == len(df.columns), "New column names and dataframe columns must have the same length"
            df.columns = column_names
            df = df[select_columns]

            if 'player' in df.columns:
                df = df[df['player'] != 'League Average']

            df['year'] = int(file.split('/')[-1].split('_')[0])

            for column, func in transformations.items():
                if column in df.columns:
                    df[column] = df[column].apply(func)
                else:
                    logger.warning(f"Column '{column}' not found in dataframe, skipping transformation.")

            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        cols = combined_df.columns.tolist()
        if 'year' in cols:
            cols.remove('year')
            cols.insert(1, 'year')
            combined_df = combined_df[cols]

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
        column_names = [
            'rank',
            'player',
            'team',
            'position',
            'age',
            'games',
            'games_started',
            'pass_completions',
            'pass_attempts',
            'pass_yards',
            'pass_touchdowns',
            'pass_interceptions',
            'rush_attempts',
            'rush_yards',
            'rush_yards_per_attempt',
            'rush_touchdowns',
            'rec_targets',
            'rec_receptions',
            'rec_yards',
            'rec_yards_per_reception',
            'rec_touchdowns',
            'fumbles',
            'fumbles_lost',
            'total_touchdowns',
            'two_point_conversions',
            'two_point_conversion_passes',
            'standard_fantasy_points',
            'ppr_fantasy_points',
            'dk_fantasy_points',
            'fd_fantasy_points',
            'value_over_replacement',
            'position_rank',
            'overall_rank'
        ]
        select_columns = [
            'player',
            'team',
            'position',
            'age',
            'games',
            'games_started',
            'standard_fantasy_points',
            'ppr_fantasy_points',
            'value_over_replacement'
        ]

        self.combine_year_data(
            file_pattern="*_player_fantasy_stats.csv",
            column_names=column_names,
            select_columns=select_columns,
            transformations={'player': self.standardize_name},
            output_file_name="player_fantasy_stats.csv"
        )

    def build_player_receiving_stats(self) -> None:
        """
        Reads in all years of receiving stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        column_names = [
            'rank',
            'player',
            'age',
            'team',
            'position',
            'games',
            'games_started',
            'rec_receptions',
            'rec_targets',
            'rec_yards',
            'rec_first_downs',
            'rec_yards_before_catch',
            'rec_yards_before_catch_per_reception',
            'rec_yards_after_catch',
            'rec_yards_after_catch_per_reception',
            'rec_average_depth_of_target',
            'rec_broken_tackles',
            'rec_receptions_per_broken_tackle',
            'rec_drops',
            'rec_drop_percentage',
            'rec_interceptions_when_targeted',
            'rec_passer_rating_when_targeted',
            'rec_awards'
        ]
        excluded_columns = {'rank', 'age', 'team', 'position', 'games', 'games_started'}

        self.combine_year_data(
            file_pattern="*_player_receiving_stats.csv",
            column_names=column_names,
            select_columns=[col for col in column_names if col not in excluded_columns],
            transformations={'player': self.standardize_name, 'rec_awards': self.parse_awards},
            output_file_name="player_receiving_stats.csv"
        )

    def build_player_rushing_stats(self) -> None:
        """
        Reads in all years of rushing stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        column_names = [
            'rank',
            'player',
            'age',
            'team',
            'position',
            'games',
            'games_started',
            'rush_attempts',
            'rush_yards',
            'rush_first_downs',
            'rush_yards_before_contact',
            'rush_yards_before_contact_per_attempt',
            'rush_yards_after_contact',
            'rush_yards_after_contact_per_attempt',
            'rush_broken_tackles',
            'rush_attempts_per_broken_tackle',
            'rush_awards'
        ]
        excluded_columns = {'rank', 'age', 'team', 'position', 'games', 'games_started'}

        self.combine_year_data(
            file_pattern="*_player_rushing_stats.csv",
            column_names=column_names,
            select_columns=[col for col in column_names if col not in excluded_columns],
            transformations={'player': self.standardize_name, 'rush_awards': self.parse_awards},
            output_file_name="player_rushing_stats.csv"
        )

    def build_player_passing_stats(self) -> None:
        """
        Reads in all years of passing stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        column_names = [
            'rank',
            'player',
            'age',
            'team',
            'position',
            'games',
            'games_started',
            'pass_completions',
            'pass_attempts',
            'pass_incomplete_air_yards',
            'pass_incomplete_air_yards_per_attempt',
            'pass_completed_air_yards',
            'pass_completed_air_yards_per_completion',
            'pass_completed_air_yards_per_attempt',
            'pass_yards_after_catch',
            'pass_yards_after_catch_per_completion',
            'pass_passes_batted',
            'pass_passes_thrown_away',
            'pass_spikes',
            'pass_drops',
            'pass_drop_percentage',
            'pass_bad_throws',
            'pass_bad_throw_percentage',
            'pass_on_target_throws',
            'pass_on_target_percentage',
            'pass_pocket_time',
            'pass_blitzes',
            'pass_hurries',
            'pass_hits',
            'pass_pressures',
            'pass_pressure_percentage',
            'pass_scrambles',
            'pass_yards_per_scramble',
            'rpo_plays',
            'rpo_yards',
            'rpo_pass_attemps',
            'rpo_pass_yards',
            'rpo_rush_attempts',
            'rpo_rush_yards',
            'play_action_pass_attempts',
            'play_action_pass_yards',
            'awards'
        ]
        select_columns = [
            'player',
            'pass_incomplete_air_yards',
            'pass_completed_air_yards',
            'pass_yards_after_catch',
            'pass_passes_thrown_away',
            'pass_drops',
            'pass_bad_throws',
            'pass_on_target_throws',
            'pass_pocket_time',
            'pass_blitzes',
            'pass_hurries',
            'pass_hits',
            'pass_pressures',
            'awards'
        ]

        self.combine_year_data(
            file_pattern="*_player_passing_stats.csv",
            column_names=column_names,
            select_columns=select_columns,
            transformations={'player': self.standardize_name, 'awards': self.parse_awards},
            output_file_name="player_passing_stats.csv"
        )

    def build_team_stats(self) -> None:
        """
        Reads in all years of team stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.

        Returns:
            None (saves data to silver layer)
        """
        column_names = [
            'rank',
            'team',
            'games',
            'points_scored',
            'yards',
            'plays',
            'team_yards_per_play',
            'team_turnovers',
            'team_fumbles_lost',
            'team_first_downs',
            'team_pass_completions',
            'team_pass_attempts',
            'team_pass_yards',
            'team_pass_touchdowns',
            'team_pass_interceptions',
            'team_pass_net_yards_per_attempt',
            'team_pass_first_downs',
            'team_rush_attempts',
            'team_rush_yards',
            'team_rush_touchdowns',
            'team_rush_yards_per_attempt',
            'team_rush_first_downs',
            'team_penalties',
            'team_penalty_yards',
            'team_penalty_first_downs',
            'team_scoring_percent',
            'team_turnover_percent',
            'team_expected_points'
        ]
        excluded_columns = {'rank', 'games'}

        self.combine_year_data(
            file_pattern="*_team_offense.csv",
            column_names=column_names,
            select_columns=[col for col in column_names if col not in excluded_columns],
            transformations={'team': self.standardize_team_name},
            output_file_name="team_offense.csv"
        )

    def join_stats(self) -> None:
        """
        Joins the player stats into a single dataframe.
        """
        fantasy_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "player_fantasy_stats.csv"))
        receiving_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "player_receiving_stats.csv"))
        rushing_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "player_rushing_stats.csv"))
        passing_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "player_passing_stats.csv"))
        team_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "team_offense.csv"))

        joined_df = pd.merge(fantasy_stats_df, receiving_stats_df, on=['player', 'year'], how='left')
        joined_df = pd.merge(joined_df, rushing_stats_df, on=['player', 'year'], how='left')
        joined_df = pd.merge(joined_df, passing_stats_df, on=['player', 'year'], how='left')
        joined_df = pd.merge(joined_df, team_stats_df, on=['team', 'year'], how='left')

        joined_df.to_csv(os.path.join(self.silver_data_dir, "all_stats.csv"), index=False)

    def create_rollup_stats(self, df: pd.DataFrame, stats_to_rollup: List[str]) -> pd.DataFrame:
        """
        Creates rolling average stats for teams over multiple years.
        For example, 2-year average points scored for each team.
        """
        team_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "team_offense.csv"))
        
        # Sort by team and year to ensure proper rolling calculation
        team_stats_df = team_stats_df.sort_values(['team', 'year'])
        
        # Create rolling averages for numeric columns
        numeric_columns = team_stats_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['year']]  # Exclude year
        
        # Create 2-year rolling averages
        for col in numeric_columns:
            rollup_col = f"2_yr_avg_{col}"
            team_stats_df[rollup_col] = team_stats_df.groupby('team')[col].rolling(
                window=2, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Create 3-year rolling averages
        for col in numeric_columns:
            rollup_col = f"3_yr_avg_{col}"
            team_stats_df[rollup_col] = team_stats_df.groupby('team')[col].rolling(
                window=3, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Save the enhanced team stats
        team_stats_df.to_csv(os.path.join(self.silver_data_dir, "team_offense_with_rollups.csv"), index=False)
        logger.info("Created rollup stats for team offense data")

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

        self.join_stats()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    processor = FantasyDataProcessor(data_dir=data_dir)

    processor.process_all_data()
    logger.info("Data processing complete")


if __name__ == "__main__":
    main()