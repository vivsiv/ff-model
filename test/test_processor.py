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
            'Rank': [1, 2, None],
            'Player': ['John Doe', 'Jane Smith', "League Average"],
            'Team': ['PHI', 'DAL', ""],
            'Points': [100, 90, 95]
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
            column_names=['rank', 'player', 'team', 'points'],
            select_columns=['player', 'team', 'points'],
            transformations={'player': self.processor.standardize_name}
        )

        # Verify comined dataframe
        expected_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', 'bob_johnson', 'alice_brown', 'john_doe'],
            'year': [2023, 2023, 2024, 2024, 2024],
            'team': ['PHI', 'DAL', 'MIA', 'NYG', 'PHI'],
            'points': [100, 90, 110, 95, 105],
        })

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
                column_names=['col1', 'col2', 'col3'],
                select_columns=['col1'],
                transformations={}
            )

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
        })

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

  
    @patch('pandas.read_csv')
    @patch('src.processor.FantasyDataProcessor.write_to_silver')
    def test_join_stats(self, mock_write, mock_read_csv):
        """Test joining all stats into a single dataframe."""
        fantasy_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith'],
            'year': [2023, 2023],
            'team': ['PHI', 'DAL'],
            'fantasy_points': [100, 90]
        })

        receiving_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe', 'jane_smith'],
            'year': [2021, 2022, 2023, 2022],
            'rec_yards': [600, 1000, 1200, 800]
        })

        rushing_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe'],
            'year': [2021, 2022, 2023],
            'rush_yards': [500, 600, 700]
        })

        passing_df = pd.DataFrame({
            'player': ['john_doe', 'john_doe', 'john_doe', 'jane_smith'],
            'year': [2021, 2022, 2023, 2021],
            'pass_yards': [3000, 3500, 4000, 3200]
        })

        team_df = pd.DataFrame({
            'team': ['PHI', 'PHI', 'PHI', 'DAL'],
            'year': [2021, 2022, 2023, 2022],
            'team_points': [400, 350, 450, 300]
        })

        # Have each call to pd.read_csv return each of the test dataframes
        mock_read_csv.side_effect = [fantasy_df, receiving_df, rushing_df, passing_df, team_df]
        
        self.processor.join_stats()

        expected_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith'],
            'year': [2023, 2023],
            'team': ['PHI', 'DAL'],
            'fantasy_points': [100, 90],
            'rec_yards': [1000, 800],
            'rush_yards': [600, np.nan],
            'pass_yards': [3500, np.nan],
            'team_points': [350, 300]
        })

        joined_df = mock_write.call_args[0][0]

        pd.testing.assert_frame_equal(joined_df, expected_df)


if __name__ == '__main__':
    unittest.main()
