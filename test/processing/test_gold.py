import os
import tempfile
import shutil

import pandas as pd

from src.processing.gold import TrainingSetBuilder


class TestTrainingSetBuilder():
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.test_dir, "silver"))
        cls.builder = TrainingSetBuilder(data_dir=cls.test_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_positional_baseline__trailing_window_within_position(self):
        df = pd.DataFrame({
            "position": ["WR", "WR", "WR", "WR", "RB", "RB"],
            "season": [2020, 2021, 2022, 2023, 2020, 2021],
            "player_id": ["a", "b", "c", "d", "e", "f"],
            "fantasy_points_ppr": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0],
        })

        result = self.builder._positional_baseline(df, stat_columns=["fantasy_points_ppr"], window_years=2)

        expected = pd.DataFrame({
            "position": ["RB", "RB", "WR", "WR", "WR", "WR"],
            "season": [2020, 2021, 2020, 2021, 2022, 2023],
            "fantasy_points_ppr_positional_baseline": [100.0, 150.0, 10.0, 15.0, 25.0, 35.0],
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_positional_baseline__averages_multiple_players_in_same_season(self):
        df = pd.DataFrame({
            "position": ["WR", "WR", "WR"],
            "season": [2020, 2020, 2021],
            "player_id": ["a", "b", "c"],
            "fantasy_points_ppr": [10.0, 30.0, 100.0],
        })

        result = self.builder._positional_baseline(df, stat_columns=["fantasy_points_ppr"], window_years=5)

        expected = pd.DataFrame({
            "position": ["WR", "WR"],
            "season": [2020, 2021],
            "fantasy_points_ppr_positional_baseline": [20.0, 60.0],
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_positional_baseline__handles_multiple_stat_columns(self):
        df = pd.DataFrame({
            "position": ["QB", "QB"],
            "season": [2020, 2021],
            "player_id": ["a", "b"],
            "fantasy_points_ppr": [200.0, 300.0],
            "passing_yards": [4000.0, 5000.0],
        })

        result = self.builder._positional_baseline(
            df, stat_columns=["fantasy_points_ppr", "passing_yards"], window_years=5
        )

        expected = pd.DataFrame({
            "position": ["QB", "QB"],
            "season": [2020, 2021],
            "fantasy_points_ppr_positional_baseline": [200.0, 250.0],
            "passing_yards_positional_baseline": [4000.0, 4500.0],
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
