import glob
import logging
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
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


class DataProcessor:
    """Generate silver and gold layer data from bronze layer data."""

    def __init__(self, data_dir: str = "../data", current_year: int = 2025):
        self.data_dir = data_dir
        self.bronze_data_dir = os.path.join(data_dir, "bronze")
        if not os.path.exists(self.bronze_data_dir):
            raise FileNotFoundError(f"{self.bronze_data_dir} not found")

        self.silver_data_dir = os.path.join(data_dir, "silver")
        os.makedirs(self.silver_data_dir, exist_ok=True)

        self.gold_data_dir = os.path.join(data_dir, "gold")
        os.makedirs(self.gold_data_dir, exist_ok=True)

        self.current_year = current_year

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
            "St. Louis Rams": "STL",
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

    def merge_multi_player_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges rows for players that were on multiple teams in a season.
        """
        player_row_counts = df.groupby('player').size().reset_index(name='count')
        multi_team_players = player_row_counts[player_row_counts['count'] > 1]['player'].tolist()
        for multi_team_player in multi_team_players:
            # For a multi-team player the row team=2TM has their combined stats for the year.
            correct_row_mask = ~((df['player'] == multi_team_player) & (df['team'] != '2TM'))
            df = df[correct_row_mask]

        return df

    def combine_year_data(self,
                          file_pattern: str,
                          normalized_column_names: List[str],
                          select_columns: List[str] = [],
                          transformations: Dict[str, Callable] = {}) -> pd.DataFrame:
        """
        Combines data for a single stat type from multiple years into a single dataframe.
        """
        file_pattern = os.path.join(self.bronze_data_dir, file_pattern)
        files = glob.glob(file_pattern)

        if not files:
            raise FileNotFoundError(f"No files found matching {file_pattern}")

        year_dfs = []
        for file in tqdm(files, desc=f"Processing files matching: {file_pattern.split('/')[-1]}"):
            year_df = pd.read_csv(file)

            try:
                assert len(normalized_column_names) == len(year_df.columns), f"{file} columns must be the same length as provided normalized_column_names"
                year_df.columns = normalized_column_names
            except (AssertionError, ValueError) as e:
                if file.endswith('_player_passing_stats.csv') and 'QBR' not in year_df.columns and len(normalized_column_names) == len(year_df.columns) + 1:
                    non_qbr_normalized_columns = [col for col in normalized_column_names if col != 'pass_qbr']
                    year_df.columns = non_qbr_normalized_columns
                    year_df['pass_qbr'] = 50.0
                    logger.info(f"Added missing pass_qbr column to {file}")
                elif file.endswith('_team_offense.csv') and 'EXP' not in year_df.columns and len(normalized_column_names) == len(year_df.columns) + 1:
                    non_exp_normalized_columns = [col for col in normalized_column_names if col != 'team_expected_points']
                    year_df.columns = non_exp_normalized_columns
                    year_df['team_expected_points'] = 0.0
                    logger.info(f"Added missing team_expected_points column to {file}")
                else:
                    logger.error(f"{file} columns must be the same length as provided normalized_column_names")
                    raise e

            # Add a year column as the second column
            year_value = int(file.split('/')[-1].split('_')[0])
            year_df.insert(1, 'year', year_value)

            # Handle player stat edge cases
            if 'player' in year_df.columns:
                year_df = self.merge_multi_player_rows(year_df)
                year_df = year_df[year_df['player'] != 'League Average']

            # Apply any transformations to the columns
            for column, func in transformations.items():
                if column in year_df.columns:
                    year_df[column] = year_df[column].apply(func)
                else:
                    logger.warning(f"Column '{column}' not found in dataframe, skipping transformation.")

            year_dfs.append(year_df)

        return pd.concat(year_dfs, ignore_index=True)

    def add_ratio_stats(self, stats_df: pd.DataFrame, ratio_column_pairs: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Creates creates "ratio" columns that are the result of dividing the two columns in a tuple.

        Args:
            stats_df: The dataframe to add ratio stats to
            ratio_column_pairs: A list of tuples of the numerator and denominator columns to divide

        Returns:
            The dataframe with the ratio stats added, the list of ratio columns added
        """
        assert all(n in stats_df.columns and d in stats_df.columns for (n, d) in ratio_column_pairs), "All ratio columns must be in the input dataframe"

        ratio_columns = []
        for (n_col, d_col) in ratio_column_pairs:
            d_name = "games" if "games" in d_col else d_col
            ratio_col = f"{n_col}_per_{d_name.rstrip('s')}"
            stats_df[ratio_col] = stats_df[n_col] / stats_df[d_col]
            ratio_columns.append(ratio_col)
        return stats_df, ratio_columns

    def create_rollup_stats(self,
                            stats_df: pd.DataFrame,
                            grouping_columns: List[str],
                            rollup_columns: List[str],
                            max_rollup_window: int = 3) -> Tuple[pd.DataFrame, List[str]]:
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

        generated_columns = []
        for window in range(2, max_rollup_window + 1):
            for col in rollup_columns:
                rollup_col = f"{col}_{window}_yr_avg"
                stats_df_sorted[rollup_col] = (
                    stats_df_sorted
                    .groupby(grouping_columns)[col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
                generated_columns.append(rollup_col)

        return stats_df_sorted, generated_columns

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

        fantasy_stats_df = self.combine_year_data(
            file_pattern="*_player_fantasy_stats.csv",
            normalized_column_names=normalized_column_names,
            transformations={'player': self.standardize_name},
        )

        # TODO: Remove any rows where ppr_fantasy_points is null or 0
        fantasy_stats_df = fantasy_stats_df[fantasy_stats_df['ppr_fantasy_points'].notna()]

        ratio_column_pairs = [('standard_fantasy_points', 'games'), ('ppr_fantasy_points', 'games')]
        fantasy_stats_df, added_ratio_columns = self.add_ratio_stats(fantasy_stats_df, ratio_column_pairs)

        select_columns = [
            'player',
            'team',
            'year',
            'age',
            'standard_fantasy_points',
            'ppr_fantasy_points',
            'value_over_replacement',
            *added_ratio_columns
        ]

        fantasy_stats_df = fantasy_stats_df[select_columns]

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
            'rec_games',
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

        receiving_stats_df = self.combine_year_data(
            file_pattern="*_player_receiving_stats.csv",
            normalized_column_names=normalized_column_names,
            transformations={'player': self.standardize_name, 'rec_awards': self.parse_awards},
        )

        excluded_columns = {'rank', 'team', 'position', 'games_started', 'rec_longest_reception', 'rec_fumbles'}
        base_columns = [col for col in normalized_column_names if col not in excluded_columns]

        ratio_column_pairs = [('rec_targets', 'rec_games'), ('rec_touchdowns', 'rec_games'), ('rec_first_downs', 'rec_games')]
        receiving_stats_df, added_ratio_columns = self.add_ratio_stats(receiving_stats_df, ratio_column_pairs)

        rollup_columns = [col for col in base_columns if col not in ['player', 'age', 'rec_awards']] + added_ratio_columns
        receiving_stats_df, added_rollup_columns = self.create_rollup_stats(
            stats_df=receiving_stats_df,
            grouping_columns=['player'],
            rollup_columns=rollup_columns,
        )

        select_columns = base_columns + ['year'] + added_rollup_columns + added_ratio_columns
        receiving_stats_df = receiving_stats_df[select_columns]

        self.write_to_silver(receiving_stats_df, "player_receiving_stats.csv")

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
            'rush_games',
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

        rushing_stats_df = self.combine_year_data(
            file_pattern="*_player_rushing_stats.csv",
            normalized_column_names=normalized_column_names,
            transformations={'player': self.standardize_name, 'rush_awards': self.parse_awards},
        )

        excluded_columns = {'rank', 'team', 'position', 'games_started', 'rush_longest_rush', 'rush_fumbles'}
        base_columns = [col for col in normalized_column_names if col not in excluded_columns]

        ratio_column_pairs = [('rush_touchdowns', 'rush_games'), ('rush_first_downs', 'rush_games')]
        rushing_stats_df, added_ratio_columns = self.add_ratio_stats(rushing_stats_df, ratio_column_pairs)

        rollup_columns = [col for col in base_columns if col not in ['player', 'age', 'rush_awards']] + added_ratio_columns
        rushing_stats_df, added_rollup_columns = self.create_rollup_stats(
            stats_df=rushing_stats_df,
            grouping_columns=['player'],
            rollup_columns=rollup_columns,
        )

        select_columns = base_columns + ['year'] + added_rollup_columns + added_ratio_columns
        rushing_stats_df = rushing_stats_df[select_columns]

        self.write_to_silver(rushing_stats_df, "player_rushing_stats.csv")

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
            'pass_games',
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

        passing_stats_df = self.combine_year_data(
            file_pattern="*_player_passing_stats.csv",
            normalized_column_names=normalized_column_names,
            transformations={'player': self.standardize_name, 'pass_awards': self.parse_awards},
        )

        excluded_columns = {'rank', 'team', 'position', 'games_started', 'pass_record', 'pass_longest_pass', 'pass_fourth_quarter_comebacks', 'pass_game_winning_drives'}
        base_columns = [col for col in normalized_column_names if col not in excluded_columns]

        ratio_column_pairs = [('pass_touchdowns', 'pass_games'), ('pass_interceptions', 'pass_games'), ('pass_first_downs', 'pass_games'), ('pass_sacks', 'pass_games'), ('pass_sack_yards', 'pass_games')]
        passing_stats_df, added_ratio_columns = self.add_ratio_stats(passing_stats_df, ratio_column_pairs)

        rollup_columns = [col for col in base_columns if col not in ['player', 'age', 'pass_awards']] + added_ratio_columns
        passing_stats_df, added_rollup_columns = self.create_rollup_stats(
            stats_df=passing_stats_df,
            grouping_columns=['player'],
            rollup_columns=rollup_columns,
        )

        select_columns = base_columns + ['year'] + added_rollup_columns + added_ratio_columns
        passing_stats_df = passing_stats_df[select_columns]

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

    def build_team_stats(self) -> None:
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

        team_offense_df = self.combine_year_data(
            file_pattern="*_team_offense.csv",
            normalized_column_names=normalized_column_names,
            transformations={'team': self.standardize_team_name},
        )

        excluded_columns = {'rank', 'games'}
        base_columns = [col for col in normalized_column_names if col not in excluded_columns]

        rollup_columns = ['team_points', 'team_yards', 'team_plays', 'team_yards_per_play']
        team_offense_df, added_rollup_columns = self.create_rollup_stats(
            stats_df=team_offense_df,
            grouping_columns=['team'],
            rollup_columns=rollup_columns,
        )
        team_offense_df = self.add_league_average_rows(team_offense_df)

        select_columns = base_columns + ['year'] + added_rollup_columns
        team_offense_df = team_offense_df[select_columns]

        self.write_to_silver(team_offense_df, "team_offense.csv")

    def join_training_stats(self, add_advanced_stats: bool = False) -> pd.DataFrame:
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

        return joined_df.drop(columns=['join_year', 'join_age'])

    def join_live_stats(self, current_year: int) -> pd.DataFrame:
        """
        Joins the player stats into a single dataframe.
        The live set must use the current year stats to predict the next years fantasy points
        so the join needs to be: year N in fantasy stats with year N+1 in other stat tables.
        """
        player_names_df = pd.read_csv(os.path.join(self.data_dir, f"{current_year}_fantasy_players.csv"))
        join_year = current_year - 1

        receiving_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "player_receiving_stats.csv"))
        receiving_stats_df = receiving_stats_df[receiving_stats_df['year'] == join_year].drop(columns=['year'])
        receiving_stats_df = receiving_stats_df.rename(columns={'age': 'age_receiving'})

        rushing_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "player_rushing_stats.csv"))
        rushing_stats_df = rushing_stats_df[rushing_stats_df['year'] == join_year].drop(columns=['year'])
        rushing_stats_df = rushing_stats_df.rename(columns={'age': 'age_rushing'})

        passing_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "player_passing_stats.csv"))
        passing_stats_df = passing_stats_df[passing_stats_df['year'] == join_year].drop(columns=['year'])
        passing_stats_df = passing_stats_df.rename(columns={'age': 'age_passing'})

        team_stats_df = pd.read_csv(os.path.join(self.silver_data_dir, "team_offense.csv"))
        team_stats_df = team_stats_df[team_stats_df['year'] == join_year].drop(columns=['year'])

        joined_df = (
            pd.merge(player_names_df, receiving_stats_df, on=['player'], how='left')
            .merge(rushing_stats_df, on=['player'], how='left')
            .merge(passing_stats_df, on=['player'], how='left')
            .merge(team_stats_df, on=['team'], how='left')
        )

        joined_df = self.collapse_duplicate_columns(joined_df, ['age_receiving', 'age_rushing', 'age_passing'], 'age')
        joined_df['age'] = joined_df['age'] + 1

        return joined_df

    def collapse_duplicate_columns(self, df: pd.DataFrame, duplicate_columns: List[str], new_column_name: str) -> pd.DataFrame:
        """
        Collapses duplicate columns into a single column.
        """
        if all(col in df.columns for col in duplicate_columns):
            df[new_column_name] = df[duplicate_columns].max(axis=1)
            df = df.drop(columns=duplicate_columns)
        return df

    def clean_stats(self, joined_df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        final_df = joined_df.copy()

        if is_training:
            # Drop any rows where the player is null or an empty string, Does this happen? Should probably be done earlier if so.
            # final_df = final_df.loc[(final_df['player'].notna()) & (final_df['player'] != '')]

            # Drop the first year of data as it will have 0 for all stats
            final_df = final_df[final_df['year'] != final_df['year'].min()]

            # Make an id column that is player_year
            final_df.insert(0, 'id', final_df['player'].astype(str) + '_' + final_df['year'].astype(str))
            final_df = final_df.drop(columns=['player', 'year', 'team'])

        # Combine awards columns into a single awards column
        final_df = self.collapse_duplicate_columns(final_df, ['pass_awards', 'rush_awards', 'rec_awards'], 'awards')

        # Combine multiple games columns into one column.
        final_df = self.collapse_duplicate_columns(final_df, ['pass_games', 'rush_games', 'rec_games'], 'games')
        final_df = self.collapse_duplicate_columns(final_df, ['pass_games_2_yr_avg', 'rush_games_2_yr_avg', 'rec_games_2_yr_avg'], 'games_2_yr_avg')
        final_df = self.collapse_duplicate_columns(final_df, ['pass_games_3_yr_avg', 'rush_games_3_yr_avg', 'rec_games_3_yr_avg'], 'games_3_yr_avg')

        # Fill null values in feature columns with 0 and round to 2 decimal places
        feature_columns = final_df.select_dtypes(include=[np.number]).columns
        final_df[feature_columns] = final_df[feature_columns].fillna(0).astype(float).round(2)

        # Exclude rookies (rookies are players who have 0's for all games stats.)
        if all(col in final_df.columns for col in ['games', 'games_2_yr_avg', 'games_3_yr_avg']):
            final_df = final_df[(final_df['games'] > 0) | (final_df['games_2_yr_avg'] > 0) | (final_df['games_3_yr_avg'] > 0)]

        return final_df

    def build_training_set(self) -> None:
        training_df = self.join_training_stats()
        training_df = self.clean_stats(training_df, is_training=True)
        training_df.to_csv(os.path.join(self.gold_data_dir, "training_set.csv"), index=False)
        logger.info(f"Final data saved to {os.path.join(self.gold_data_dir, 'training_set.csv')}")

    def build_live_set(self) -> None:
        live_df = self.join_live_stats(current_year=self.current_year)
        live_df = self.clean_stats(live_df, is_training=False)
        live_df.to_csv(os.path.join(self.gold_data_dir, "live_set.csv"), index=False)
        logger.info(f"Final data saved to {os.path.join(self.gold_data_dir, 'live_set.csv')}")

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

        self.build_training_set()
        self.build_live_set()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    processor = DataProcessor(data_dir=data_dir)

    processor.process_all_data()


if __name__ == "__main__":
    main()
