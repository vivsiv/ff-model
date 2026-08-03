import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

from src.processing.pro_football_reference import ProFootballReferenceProcessor


class TestProFootballReferenceProcessor():
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.bronze_dir = os.path.join(cls.test_dir, "bronze", "pfr")
        os.makedirs(cls.bronze_dir)

        cls.processor = ProFootballReferenceProcessor(data_dir=cls.test_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_standardize_name(self):
        """Test name standardization functionality."""
        assert self.processor.standardize_name("A.J. Brown") == "aj_brown"
        assert self.processor.standardize_name("Kenneth Walker III") == "kenneth_walker"
        assert self.processor.standardize_name("Odell Beckham Jr.") == "odell_beckham"
        assert self.processor.standardize_name("Calvin Ridley*") == "calvin_ridley"
        assert self.processor.standardize_name("DeAndre Hopkins+") == "deandre_hopkins"
        assert self.processor.standardize_name("D'Andre Swift") == "dandre_swift"
        assert self.processor.standardize_name("Ja'Marr Chase") == "jamarr_chase"

        assert self.processor.standardize_name("  Extra Spaces  ") == "extra_spaces"
        assert self.processor.standardize_name("UPPERCASE NAME") == "uppercase_name"
        assert self.processor.standardize_name("Multi-Hyphen Name") == "multi_hyphen_name"

    def test_standardize_team_name(self):
        """Test team name standardization."""
        assert self.processor.standardize_team_name("Green Bay Packers") == "GNB"
        assert self.processor.standardize_team_name("Las Vegas Raiders") == "LVR"
        assert self.processor.standardize_team_name("New York Giants") == "NYG"
        assert self.processor.standardize_team_name("San Francisco 49ers") == "SFO"

        assert self.processor.standardize_team_name("Philadelphia Eagles") == "PHI"
        assert self.processor.standardize_team_name("Dallas Cowboys") == "DAL"
        assert self.processor.standardize_team_name("Kansas City Chiefs") == "KAN"

        assert self.processor.standardize_team_name("  Chicago Bears  ") == "CHI"

    def test_parse_awards(self):
        """Test awards parsing functionality."""
        assert self.processor.parse_awards("PB,AP-1,AP MVP-3") == 3
        assert self.processor.parse_awards("PB") == 1
        assert self.processor.parse_awards("AP-1,AP-2") == 2

        assert self.processor.parse_awards(np.nan) == 0.0
        assert self.processor.parse_awards(None) == 0.0
        assert self.processor.parse_awards("") == 0.0

    def test_merge_multi_player_rows(self):
        """Test merging of multi-player rows."""
        test_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', 'jane_smith', 'jane_smith'],
            'team': ['PHI', '2TM', 'DAL', 'WAS'],
            'points': [100, 90, 80, 70]
        })

        result = self.processor.merge_multi_player_rows(test_df)

        expected_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith'],
            'team': ['PHI', '2TM'],
            'points': [100, 90]
        })

        pd.testing.assert_frame_equal(result, expected_df)

    def test_combine_year_data_success(self):
        """Test successful combination of year data."""
        # Create test CSV files
        test_data_2023 = pd.DataFrame({
            'Player': ['John Doe', 'Jane Smith', "League Average", "Alice Brown", "Alice Brown", "Alice Brown"],
            'Team': ['PHI', 'DAL', "", "2TM", "NYJ", "NYG"],
            'Points': [100, 90, 95, 80, 30, 50],
        })
        test_data_2024 = pd.DataFrame({
            'Player': ['Bob Johnson', 'Alice Brown+', 'John Doe'],
            'Team': ['MIA', 'NYG', 'PHI'],
            'Points': [110, 95, 105]
        })
        test_data_2023.to_csv(os.path.join(self.bronze_dir, "2023_test_stats.csv"), index=False)
        test_data_2024.to_csv(os.path.join(self.bronze_dir, "2024_test_stats.csv"), index=False)

        # Combine data
        result = self.processor.combine_year_data(
            file_pattern="*_test_stats.csv",
            normalized_column_names=['player', 'team', 'points'],
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

        with pytest.raises(AssertionError):
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
            'rec_yards': [100, 150, 200, 250, 300],
            'rec_touchdowns': [10, 15, 20, 25, 30],
            'rec_games': [5, 10, 10, 10, 5]
        })

        result_df, _ = self.processor.add_ratio_stats(
            test_df,
            [('rec_yards', 'rec_games'), ('rec_touchdowns', 'rec_games')]
        )
        result_df = result_df.sort_values(['player']).reset_index(drop=True)

        expected_df = pd.DataFrame({
            'player': ['john_doe', 'jane_smith', 'frank_west', 'eric_east', "nolan_north"],
            'rec_yards': [100, 150, 200, 250, 300],
            'rec_touchdowns': [10, 15, 20, 25, 30],
            'rec_games': [5, 10, 10, 10, 5],
            'rec_yards_per_game': [20.0, 15.0, 20.0, 25.0, 60.0],
            'rec_touchdowns_per_game': [2.0, 1.5, 2.0, 2.5, 6.0]
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

        result, generated_columns = self.processor.create_rollup_stats(
            stats_df=test_df,
            grouping_columns=['player'],
            rollup_columns=['yards', 'touchdowns'],
            max_rollup_window=3
        )
        result = result.sort_values(['player', 'year']).reset_index(drop=True)

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

        with pytest.raises(AssertionError):
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

    def test_add_league_average_rows_no_team_stats(self):
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
