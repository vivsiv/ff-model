import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.processor import FantasyDataProcessor


class TestFantasyDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.bronze_dir = os.path.join(self.test_dir, "bronze")
        self.silver_dir = os.path.join(self.test_dir, "silver")
        self.gold_dir = os.path.join(self.test_dir, "gold")

        os.makedirs(self.bronze_dir)
        os.makedirs(self.silver_dir)
        os.makedirs(self.gold_dir)

        self.processor = FantasyDataProcessor(data_dir=self.test_dir)

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir)

    def test_standardize_name(self):
        """Test name standardization functionality."""
        self.assertEqual(self.processor.standardize_name("A.J. Brown"), "aj_brown")
        self.assertEqual(self.processor.standardize_name("Kenneth Walker III"), "kenneth_walker")
        self.assertEqual(self.processor.standardize_name("Odell Beckham Jr."), "odell_beckham")
        self.assertEqual(self.processor.standardize_name("Calvin Ridley*"), "calvin_ridley")
        self.assertEqual(self.processor.standardize_name("DeAndre Hopkins+"), "deandre_hopkins")
        self.assertEqual(self.processor.standardize_name("D'Andre Swift"), "dandre_swift")
        self.assertEqual(self.processor.standardize_name("Ja'Marr Chase"), "jamarr_chase")

        self.assertEqual(self.processor.standardize_name("  Extra Spaces  "), "extra_spaces")
        self.assertEqual(self.processor.standardize_name("UPPERCASE NAME"), "uppercase_name")
        self.assertEqual(self.processor.standardize_name("Multi-Hyphen Name"), "multi_hyphen_name")

    def test_standardize_team_name(self):
        """Test team name standardization."""
        self.assertEqual(self.processor.standardize_team_name("Green Bay Packers"), "GNB")
        self.assertEqual(self.processor.standardize_team_name("Las Vegas Raiders"), "LVR")
        self.assertEqual(self.processor.standardize_team_name("New York Giants"), "NYG")
        self.assertEqual(self.processor.standardize_team_name("San Francisco 49ers"), "SFO")

        self.assertEqual(self.processor.standardize_team_name("Philadelphia Eagles"), "PHI")
        self.assertEqual(self.processor.standardize_team_name("Dallas Cowboys"), "DAL")
        self.assertEqual(self.processor.standardize_team_name("Kansas City Chiefs"), "KAN")

        self.assertEqual(self.processor.standardize_team_name("  Chicago Bears  "), "CHI")

    def test_parse_awards(self):
        """Test awards parsing functionality."""
        self.assertEqual(self.processor.parse_awards("PB,AP-1,AP MVP-3"), 3)
        self.assertEqual(self.processor.parse_awards("PB"), 1)
        self.assertEqual(self.processor.parse_awards("AP-1,AP-2"), 2)

        self.assertEqual(self.processor.parse_awards(np.nan), 0.0)
        self.assertEqual(self.processor.parse_awards(None), 0.0)
        self.assertEqual(self.processor.parse_awards(""), 0.0)

    def test_combine_year_data_success(self):
        """Test successful combination of year data."""
        # Create test CSV files
        test_data_2023 = pd.DataFrame({
            'Rank': [1, 2, None, 3, 4, 5],
            'Player': ['John Doe', 'Jane Smith', "League Average", "Alice Brown", "Alice Brown", "Alice Brown"],
            'Team': ['PHI', 'DAL', "", "2TM", "NYJ", "NYG"],
            'Points': [100, 90, 95, 80, 30, 50],
        })
        test_data_2024 = pd.DataFrame({
            'Rank': [1, 3, 2],
            'Player': ['Bob Johnson', 'Alice Brown+', 'John Doe'],
            'Team': ['MIA', 'NYG', 'PHI'],
            'Points': [110, 95, 105]
        })
        test_data_2023.to_csv(os.path.join(self.bronze_dir, "2023_test_stats.csv"), index=False)
        test_data_2024.to_csv(os.path.join(self.bronze_dir, "2024_test_stats.csv"), index=False)

        # Combine data
        result = self.processor.combine_year_data(
            file_pattern="*_test_stats.csv",
            normalized_column_names=['rank', 'player', 'team', 'points'],
            select_columns=['player', 'team', 'points'],
            transformations={'player': self.processor.standardize_name}
        ).sort_values(['player', 'year']).reset_index(drop=True)

        # Verify combined dataframe
        expected_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', 'alice_brown', 'bob_johnson', 'alice_brown', 'john_doe'],
            'year': [2023, 2023, 2023, 2024, 2024, 2024],
            'team': ['PHI', 'DAL', '2TM', 'MIA', 'NYG', 'PHI'],
            'points': [100, 90, 80, 110, 95, 105],
        }).sort_values(['player', 'year']).reset_index(drop=True)

        pd.testing.assert_frame_equal(result, expected_df)

    def test_combine_year_data_column_mismatch(self):
        """Test combine_year_data with column count mismatch."""
        # Create test CSV with wrong number of columns
        test_data = pd.DataFrame({
            'Col1': [1, 2],
            'Col2': ['A', 'B']
        })
        test_data.to_csv(os.path.join(self.bronze_dir, "2023_mismatch.csv"), index=False)

        with self.assertRaises(AssertionError):
            self.processor.combine_year_data(
                file_pattern="*_mismatch.csv",
                normalized_column_names=['col1', 'col2', 'col3'],
                select_columns=['col1'],
                transformations={}
            )

    def test_add_ratio_stats(self):
        """Test addition of ratio stats."""
        test_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', 'frank_west', 'eric_east', "nolan_north"],
            'yards': [100, 150, 200, 250, 300],
            'touchdowns': [10, 15, 20, 25, 30],
            'games': [5, 10, 10, 10, 5]
        })

        result_df, ratio_columns = self.processor.add_ratio_stats(
            test_df,
            [('yards', 'games'), ('touchdowns', 'games')]
        )
        result_df = result_df.sort_values(['player']).reset_index(drop=True)

        expected_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', 'frank_west', 'eric_east', "nolan_north"],
            'yards': [100, 150, 200, 250, 300],
            'touchdowns': [10, 15, 20, 25, 30],
            'games': [5, 10, 10, 10, 5],
            'yards_per_game': [20.0, 15.0, 20.0, 25.0, 60.0],
            'touchdowns_per_game': [2.0, 1.5, 2.0, 2.5, 6.0]
        }).sort_values(['player']).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_create_rollup_stats(self):
        """Test creation of rollup statistics."""
        test_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe', 'jane_smith', 'jane_smith'],
            'year': [2020, 2021, 2022, 2021, 2022],
            'yards': [100, 120, 140, 80, 90],
            'touchdowns': [5, 6, 7, 3, 4]
        })

        result = self.processor.create_rollup_stats(
            stats_df=test_df,
            grouping_columns=['player'],
            rollup_columns=['yards', 'touchdowns'],
            max_rollup_window=3
        ).sort_values(['player', 'year']).reset_index(drop=True)

        expected_df = pd.DataFrame({
            'player': ['jane_smith', 'jane_smith', 'john_doe', 'john_doe', 'john_doe'],
            'year': [2021, 2022, 2020, 2021, 2022],
            'yards': [80, 90, 100, 120, 140],
            'touchdowns': [3, 4, 5, 6, 7],
            'yards_2_yr_avg': [80.0, 85.0, 100.0, 110.0, 130.0],
            'touchdowns_2_yr_avg': [3.0, 3.5, 5.0, 5.5, 6.5],
            'yards_3_yr_avg': [80.0, 85.0, 100.0, 110.0, 120.0],
            'touchdowns_3_yr_avg': [3.0, 3.5, 5.0, 5.5, 6.0],
        }).sort_values(['player', 'year']).reset_index(drop=True)

        pd.testing.assert_frame_equal(result, expected_df)

    def test_create_rollup_stats_non_numeric_columns(self):
        """Test rollup stats with non-numeric columns raises assertion."""
        test_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe'],
            'year': [2020, 2021],
            'team': ['PHI', 'DAL']
        })

        with self.assertRaises(AssertionError):
            self.processor.create_rollup_stats(
                stats_df=test_df,
                grouping_columns=['player'],
                rollup_columns=['team'],
                max_rollup_window=2
            )

    def test_add_league_average_rows(self):
        """Test adding league average rows to the team stats dataframe."""
        test_df = pd.DataFrame({
            'year': [2023, 2023, 2022, 2022],
            'team': ['PHI', 'DAL', 'PHI', 'DAL'],
            'points': [100, 90, 120, 110],
            'yards': [5000, 4000, 6000, 5000]
        })

        result = (
            self.processor.add_league_average_rows(test_df)
            .sort_values(['year', 'team'])
            .reset_index(drop=True)
        )

        expected_df = pd.DataFrame({
            'year': [2023, 2023, 2023, 2022, 2022, 2022],
            'team': ['PHI', 'DAL', '2TM', 'PHI', 'DAL', '2TM'],
            'points': [100.0, 90.0, 95.0, 120.0, 110.0, 115.0],
            'yards': [5000.0, 4000.0, 4500.0, 6000.0, 5000.0, 5500.0]
        }).sort_values(['year', 'team']).reset_index(drop=True)

        pd.testing.assert_frame_equal(result, expected_df)

    @patch('pandas.read_csv')
    def test_join_stats(self, mock_read_csv):
        """Test joining all stats into a single dataframe."""
        fantasy_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', 'jane_smith'],
            'year': [2023, 2023, 2023],
            'age': [25, 23, 35],
            'team': ['PHI', 'DAL', 'WAS'],
            'fantasy_points': [100, 90, 10]
        })

        receiving_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe', 'jane_smith'],
            'year': [2021, 2022, 2023, 2022],
            'age': [23.0, 24.0, 25.0, 22.0],
            'rec_yards': [600, 1000, 1200, 800]
        })

        receiving_advanced_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe', 'jane_smith'],
            'year': [2021, 2022, 2023, 2022],
            'age': [23.0, 24.0, 25.0, 22.0],
            'adot': [6, 10, 12, 8]
        })

        rushing_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe'],
            'year': [2021, 2022, 2023],
            'age': [23.0, 24.0, 25.0],
            'rush_yards': [500, 600, 700]
        })

        rushing_advanced_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe'],
            'year': [2021, 2022, 2023],
            'age': [23.0, 24.0, 25.0],
            'yac': [5, 6, 7]
        })

        passing_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe', 'jane_smith', 'jane_smith'],
            'year': [2021, 2022, 2023, 2021, 2022],
            'age': [23.0, 24.0, 25.0, 21.0, 34.0],
            'pass_yards': [3000, 3500, 4000, 3200, 700]
        })

        team_df = pd.DataFrame({
            'team': ['PHI', 'PHI', 'PHI', 'DAL', 'WAS'],
            'year': [2021, 2022, 2023, 2022, 2022],
            'team_points': [400, 350, 450, 300, 250]
        })

        # Have each call to pd.read_csv return each of the test dataframes
        mock_read_csv.side_effect = [
            fantasy_df, receiving_df, rushing_df, passing_df, team_df, receiving_advanced_df, rushing_advanced_df
        ]

        joined_df = (
            self.processor.join_stats(add_advanced_stats=True)
            .sort_values(['player', 'year'])
            .reset_index(drop=True)
        )

        expected_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', 'jane_smith'],
            'year': [2023, 2023, 2023],
            'age': [25, 23, 35],
            'team': ['PHI', 'DAL', 'WAS'],
            'fantasy_points': [100, 90, 10],
            'rec_yards': [1000, 800, np.nan],
            'rush_yards': [600, np.nan, np.nan],
            'pass_yards': [3500, np.nan, 700],
            'team_points': [350, 300, 250],
            'adot': [10, 8, np.nan],
            'yac': [6, np.nan, np.nan],
        }).sort_values(['player', 'year']).reset_index(drop=True)

        pd.testing.assert_frame_equal(joined_df, expected_df)

    def test_clean_final_stats(self):
        """Test cleaning of final stats dataframe."""
        test_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', ''],
            'year': [2023, 2023, 1970],
            'team': ['PHI', 'DAL', 'HUH'],
            'fantasy_points': [100, 90, np.nan],
            'rec_yards': [1000.0, 800.233, np.nan],
            'adot': [10, 8, np.nan],
            'rush_yards': [600, 0, np.nan],
            'yac': [6, 0, np.nan],
            'pass_yards': [3500, 0, np.nan],
            'team_points': [350, 300, np.nan],
            'rec_awards': [1, 0, 0],
            'rush_awards': [1, 0, 0],
            'pass_awards': [0, 2, 0],
        })

        cleaned_df = (
            self.processor.clean_final_stats(test_df)
            .sort_values(['player', 'year'])
            .reset_index(drop=True)
        )

        expected_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith'],
            'year': [2023, 2023],
            'team': ['PHI', 'DAL'],
            'fantasy_points': [100.0, 90.0],
            'rec_yards': [1000.0, 800.23],
            'adot': [10.0, 8.0],
            'rush_yards': [600.0, 0.0],
            'yac': [6.0, 0.0],
            'pass_yards': [3500.0, 0.0],
            'team_points': [350.0, 300.0],
            'awards': [1, 2]
        }).sort_values(['player', 'year']).reset_index(drop=True)

        pd.testing.assert_frame_equal(cleaned_df, expected_df)


if __name__ == '__main__':
    unittest.main()
