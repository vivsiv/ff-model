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
            "Saint Louis Rams": "STL",
            "San Diego Chargers": "SDG",
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
        if awards == "":
            return 0.0
        return len(awards.split(','))

    def combine_year_data(self,
                          file_pattern: str,
                          normalized_column_names: List[str],
                          select_columns: List[str],
                          transformations: Dict[str, Callable]) -> pd.DataFrame:
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

            try:
                assert len(normalized_column_names) == len(df.columns), f"{file} columns must be the same length as provided normalized_column_names"
                df.columns = normalized_column_names
            except (AssertionError, ValueError) as e:
                if file.endswith('_player_passing_stats.csv') and 'QBR' not in df.columns and len(normalized_column_names) == len(df.columns) + 1:
                    no_qbr_columns = [col for col in normalized_column_names if col != 'pass_qbr']
                    df.columns = no_qbr_columns
                    df['pass_qbr'] = 50.0
                    logger.info(f"Added missing QBR column to {file}")
                else:
                    logger.error(f"{file} columns must be the same length as provided normalized_column_names")
                    raise e

            # Handle player stat edge cases
            if 'player' in df.columns:
                # Keep only one row for players that were on multiple teams in a season
                player_counts = df.groupby('player').size().reset_index(name='count')
                multi_team_players = player_counts[player_counts['count'] > 1]['player'].tolist()
                for multi_team_player in multi_team_players:
                    df = df[~((df['player'] == multi_team_player) & (df['team'] != '2TM'))]

                # Remove the league average row
                df = df[df['player'] != 'League Average']

            # Select only the columns we want to keep
            df = df[select_columns]

            # Add a year column as the second column
            year_value = int(file.split('/')[-1].split('_')[0])
            cols = df.columns.tolist()
            cols.insert(1, 'year')
            df = df.reindex(columns=cols)
            df['year'] = year_value

            # Apply any transformations to the columns
            for column, func in transformations.items():
                if column in df.columns:
                    df[column] = df[column].apply(func)
                else:
                    logger.warning(f"Column '{column}' not found in dataframe, skipping transformation.")

            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        return combined_df

    def create_rollup_stats(self,
                            stats_df: pd.DataFrame,
                            grouping_columns: List[str],
                            rollup_columns: List[str],
                            max_rollup_window: int = 3) -> pd.DataFrame:
        """
        Creates columns that are rolling averages of existing stats over multiple years.

        Args:
            stats_df: The dataframe to create rollup stats for
            grouping_columns: The columns to group by
            rollup_columns: The columns to rollup
            max_rollup_window: The maximum window size to rollup over

        Returns:
            The dataframe with the rollup stats added
        """

        numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
        assert all(col in numeric_cols for col in rollup_columns), "All columns to rollup must be in the input dataframe and numeric"

        stats_df_sorted = stats_df.sort_values("year")

        for window in range(2, max_rollup_window + 1):
            for col in rollup_columns:
                rollup_col = f"{col}_{window}_yr_avg"
                stats_df_sorted[rollup_col] = (
                    stats_df_sorted
                    .groupby(grouping_columns)[col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )

        return stats_df_sorted

    def write_to_silver(self, df: pd.DataFrame, file_name: str) -> None:
        """
        Writes a dataframe to the silver layer.
        """
        table_path = os.path.join(self.silver_data_dir, file_name)
        df.to_csv(table_path, index=False)
        logger.info(f"Saved {file_name} to {table_path}")

    def build_player_fantasy_stats(self) -> None:
        """
        Reads in all years of fantasy stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.

        Returns:
            None (saves data to silver layer)
        """
        normalized_column_names = [
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
            'age',
            'standard_fantasy_points',
            'ppr_fantasy_points',
            'value_over_replacement'
        ]

        fantasy_stats_df = self.combine_year_data(
            file_pattern="*_player_fantasy_stats.csv",
            normalized_column_names=normalized_column_names,
            select_columns=select_columns,
            transformations={'player': self.standardize_name},
        )

        fantasy_stats_df = fantasy_stats_df[fantasy_stats_df['ppr_fantasy_points'].notna()]

        self.write_to_silver(fantasy_stats_df, "player_fantasy_stats.csv")

    def build_player_receiving_stats(self) -> None:
        """
        Reads in all years of receiving stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        normalized_column_names = [
            'rank',
            'player',
            'age',
            'team',
            'position',
            'games',
            'games_started',
            'rec_targets',
            'rec_receptions',
            'rec_yards',
            'rec_yards_per_reception',
            'rec_touchdowns',
            'rec_first_downs',
            'rec_success_rate',
            'rec_longest_reception',
            'rec_receptions_per_game',
            'rec_yards_per_game',
            'rec_reception_percent',
            'rec_yards_per_target',
            'rec_fumbles',
            'rec_awards',
        ]
        excluded_columns = {'rank', 'team', 'position', 'games', 'games_started', 'rec_longest_reception', 'rec_fumbles'}
        select_columns = [col for col in normalized_column_names if col not in excluded_columns]

        receiving_stats_df = self.combine_year_data(
            file_pattern="*_player_receiving_stats.csv",
            normalized_column_names=normalized_column_names,
            select_columns=select_columns,
            transformations={'player': self.standardize_name, 'rec_awards': self.parse_awards},
        )

        rollup_columns = [col for col in select_columns if col not in ['player', 'rec_awards']]
        receiving_stats_df = self.create_rollup_stats(
            stats_df=receiving_stats_df,
            grouping_columns=['player'],
            rollup_columns=rollup_columns,
            max_rollup_window=2
        )

        self.write_to_silver(receiving_stats_df, "player_receiving_stats.csv")

    def build_player_receiving_advanced_stats(self) -> None:
        """
        Reads in all years of advanced receiving stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        normalized_column_names = [
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
        select_columns = [
            'player',
            'age',
            'rec_yards_before_catch',
            'rec_yards_before_catch_per_reception',
            'rec_yards_after_catch',
            'rec_yards_after_catch_per_reception',
            'rec_average_depth_of_target',
            'rec_broken_tackles',
            'rec_receptions_per_broken_tackle',
            'rec_drops',
            'rec_drop_percentage',
            'rec_passer_rating_when_targeted',
        ]

        receiving_stats_df = self.combine_year_data(
            file_pattern="*_player_receiving_advanced_stats.csv",
            normalized_column_names=normalized_column_names,
            select_columns=select_columns,
            transformations={'player': self.standardize_name},
        )

        rollup_columns = [col for col in select_columns if col != 'player']
        receiving_stats_df = self.create_rollup_stats(
            stats_df=receiving_stats_df,
            grouping_columns=['player'],
            rollup_columns=rollup_columns,
            max_rollup_window=2
        )

        self.write_to_silver(receiving_stats_df, "player_receiving_advanced_stats.csv")

    def build_player_rushing_stats(self) -> None:
        """
        Reads in all years of rushing stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        normalized_column_names = [
            'rank',
            'player',
            'age',
            'team',
            'position',
            'games',
            'games_started',
            'rush_attempts',
            'rush_yards',
            'rush_touchdowns',
            'rush_first_downs',
            'rush_success_rate',
            'rush_longest_rush',
            'rush_yards_per_attempt',
            'rush_yards_per_game',
            'rush_attempts_per_game',
            'rush_fumbles',
            'rush_awards'
        ]
        excluded_columns = {'rank', 'team', 'position', 'games', 'games_started', 'rush_longest_rush', 'rush_fumbles'}
        select_columns = [col for col in normalized_column_names if col not in excluded_columns]
        rushing_stats_df = self.combine_year_data(
            file_pattern="*_player_rushing_stats.csv",
            normalized_column_names=normalized_column_names,
            select_columns=select_columns,
            transformations={'player': self.standardize_name, 'rush_awards': self.parse_awards},
        )

        rollup_columns = [col for col in select_columns if col not in ['player', 'rush_awards']]
        rushing_stats_df = self.create_rollup_stats(
            stats_df=rushing_stats_df,
            grouping_columns=['player'],
            rollup_columns=rollup_columns,
            max_rollup_window=2
        )
        self.write_to_silver(rushing_stats_df, "player_rushing_stats.csv")

    def build_player_rushing_advanced_stats(self) -> None:
        """
        Reads in all years of advanced rushing stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        normalized_column_names = [
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
        select_columns = [
            'player',
            'age',
            'rush_yards_before_contact',
            'rush_yards_before_contact_per_attempt',
            'rush_yards_after_contact',
            'rush_yards_after_contact_per_attempt',
            'rush_broken_tackles',
            'rush_attempts_per_broken_tackle',
        ]

        rushing_stats_df = self.combine_year_data(
            file_pattern="*_player_rushing_advanced_stats.csv",
            normalized_column_names=normalized_column_names,
            select_columns=select_columns,
            transformations={'player': self.standardize_name},
        )

        rollup_columns = [col for col in select_columns if col != 'player']
        rushing_stats_df = self.create_rollup_stats(
            stats_df=rushing_stats_df,
            grouping_columns=['player'],
            rollup_columns=rollup_columns,
            max_rollup_window=2
        )

        self.write_to_silver(rushing_stats_df, "player_rushing_advanced_stats.csv")

    def build_player_passing_stats(self) -> None:
        """
        Reads in all years of passing stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.
        """
        normalized_column_names = [
            'rank',
            'player',
            'age',
            'team',
            'position',
            'games',
            'games_started',
            'pass_record',
            'pass_completions',
            'pass_attempts',
            'pass_completion_percentage',
            'pass_yards',
            'pass_touchdowns',
            'pass_touchdown_percent',
            'pass_interceptions',
            'pass_interception_percent',
            'pass_first_downs',
            'pass_success_rate',
            'pass_longest_pass',
            'pass_yards_per_attempt',
            'pass_adjusted_yards_per_attempt',
            'pass_yards_per_completion',
            'pass_yards_per_game',
            'pass_rating',
            'pass_qbr',
            'pass_sacks',
            'pass_sack_yards',
            'pass_sack_percent',
            'pass_net_yards_per_attempt',
            'pass_adjusted_net_yards_per_attempt',
            'pass_fourth_quarter_comebacks',
            'pass_game_winning_drives',
            'pass_awards'
        ]
        excluded_columns = {'rank', 'team', 'position', 'games', 'games_started', 'pass_record', 'pass_longest_pass', 'pass_fourth_quarter_comebacks', 'pass_game_winning_drives'}
        select_columns = [col for col in normalized_column_names if col not in excluded_columns]
        passing_stats_df = self.combine_year_data(
            file_pattern="*_player_passing_stats.csv",
            normalized_column_names=normalized_column_names,
            select_columns=select_columns,
            transformations={'player': self.standardize_name, 'pass_awards': self.parse_awards},
        )

        rollup_columns = [col for col in select_columns if col not in ['player', 'pass_awards']]

        passing_stats_df = self.create_rollup_stats(
            stats_df=passing_stats_df,
            grouping_columns=['player'],
            rollup_columns=rollup_columns,
            max_rollup_window=2
        )

        self.write_to_silver(passing_stats_df, "player_passing_stats.csv")

    def add_league_average_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds rows with the per year league average of each stat to the team stats dataframe.
        Note: This is only for the team stats dataframe, and must be done after the rollup stats are created.
        Players with multiple teams in a season are joined with the league average stats.
        """
        stats_columns = [col for col in df.columns if col not in ['team', 'year']]
        league_average_row = df[stats_columns + ['year']].groupby('year').mean().reset_index()
        league_average_row[stats_columns] = league_average_row[stats_columns].round(2)
        league_average_row['team'] = '2TM'
        return pd.concat([df, league_average_row], ignore_index=True).sort_values(['year', 'team']).reset_index(drop=True)

    def build_team_stats(self, rollup_window: int = 2) -> None:
        """
        Reads in all years of team stats from the bronze layer, cleans the data and merges them all into one dataframe.
        Saves the dataframe to the silver layer.

        Returns:
            None (saves data to silver layer)
        """
        normalized_column_names = [
            'rank',
            'team',
            'games',
            'team_points',
            'team_yards',
            'team_plays',
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
        select_columns = [col for col in normalized_column_names if col not in excluded_columns]

        team_offense_df = self.combine_year_data(
            file_pattern="*_team_offense.csv",
            normalized_column_names=normalized_column_names,
            select_columns=select_columns,
            transformations={'team': self.standardize_team_name},
        )

        rollup_columns = ['team_points', 'team_yards', 'team_plays', 'team_yards_per_play']
        team_offense_df = self.create_rollup_stats(
            stats_df=team_offense_df,
            grouping_columns=['team'],
            rollup_columns=rollup_columns,
            max_rollup_window=rollup_window
        )
        team_offense_df = self.add_league_average_rows(team_offense_df)

        self.write_to_silver(team_offense_df, "team_offense.csv")

    def join_stats(self, add_advanced_stats: bool = False) -> None:
        """
        Joins the player stats into a single dataframe.
        The training set must use the previous years stats to predict the current years fantasy points
        so the join needs to be: year N in fantasy stats with year N-1 in other stat tables.
        """
        fantasy_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "player_fantasy_stats.csv"))
        fantasy_stats_df['join_year'] = fantasy_stats_df['year'] - 1
        fantasy_stats_df['join_age'] = fantasy_stats_df['age'] - 1

        receiving_stats_df = (
            pd.read_csv(os.path.join(self.silver_data_dir, "player_receiving_stats.csv"))
            .rename(columns={'year': 'join_year', 'age': 'join_age'})
        )
        rushing_stats_df = (
            pd.read_csv(os.path.join(self.silver_data_dir, "player_rushing_stats.csv"))
            .rename(columns={'year': 'join_year', 'age': 'join_age'})
        )
        passing_stats_df = (
            pd.read_csv(os.path.join(self.silver_data_dir, "player_passing_stats.csv"))
            .rename(columns={'year': 'join_year', 'age': 'join_age'})
        )
        team_stats_df = (
            pd.read_csv(os.path.join(self.silver_data_dir, "team_offense.csv"))
            .rename(columns={'year': 'join_year'})
        )

        joined_df = (
            pd.merge(fantasy_stats_df, receiving_stats_df, on=['player', 'join_year', 'join_age'], how='left')
            .merge(rushing_stats_df, on=['player', 'join_year', 'join_age'], how='left')
            .merge(passing_stats_df, on=['player', 'join_year', 'join_age'], how='left')
            .merge(team_stats_df, on=['team', 'join_year'], how='left')
        )

        if add_advanced_stats:
            receiving_advanced_stats_df = (
                pd.read_csv(os.path.join(self.silver_data_dir, "player_receiving_advanced_stats.csv"))
                .rename(columns={'year': 'join_year', 'age': 'join_age'})
            )
            rushing_advanced_stats_df = (
                pd.read_csv(os.path.join(self.silver_data_dir, "player_rushing_advanced_stats.csv"))
                .rename(columns={'year': 'join_year', 'age': 'join_age'})
            )
            joined_df = (
                joined_df
                .merge(receiving_advanced_stats_df, on=['player', 'join_year', 'join_age'], how='left')
                .merge(rushing_advanced_stats_df, on=['player', 'join_year', 'join_age'], how='left')
            )

        joined_df = joined_df.drop(columns=['join_year', 'join_age'])

        return joined_df

    def clean_final_stats(self, joined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the final stats dataframe.
        """
        # Drop any rows where the player is null or an empty string
        joined_df = joined_df[joined_df['player'].notna() & (joined_df['player'] != '')]

        # Fill any numeric columns with null values with 0, and round to 2 decimal places
        numeric_columns = joined_df.select_dtypes(include=[np.number]).columns
        fill_columns = [col for col in numeric_columns if col != 'year']
        joined_df.loc[:, fill_columns] = joined_df[fill_columns].fillna(0).round(2)

        # Combine awards columns into a single awards column
        awards_columns = ['pass_awards', 'rush_awards', 'rec_awards']
        joined_df.loc[:, 'awards'] = joined_df[awards_columns].max(axis=1)
        joined_df = joined_df.drop(columns=awards_columns)

        return joined_df

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
        self.build_player_receiving_advanced_stats()
        self.build_player_rushing_stats()
        self.build_player_rushing_advanced_stats()
        self.build_player_passing_stats()
        self.build_team_stats()

        joined_df = self.join_stats()
        cleaned_df = self.clean_final_stats(joined_df)
        self.write_to_silver(cleaned_df, "final_stats.csv")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    processor = FantasyDataProcessor(data_dir=data_dir)

    processor.process_all_data()
    logger.info("Data processing complete")


if __name__ == "__main__":
    main()
