import os
import tempfile
import shutil

import pandas as pd
import pytest

from src.processing.gold import TrainingSetBuilder


class TestTrainingSetBuilder():
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.test_dir, "silver", "nflv"))
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

    def test_add_career_features__expanding_stats_are_inclusive_and_isolated_per_player(self):
        df = pd.DataFrame({
            "player_id": ["p1", "p1", "p1", "p2"],
            "position": ["WR", "WR", "WR", "WR"],
            "season": [2020, 2021, 2022, 2020],
            "fantasy_points_ppr": [10.0, 20.0, 30.0, 1000.0],
        })
        positional_baseline_df = pd.DataFrame({
            "position": ["WR", "WR", "WR"],
            "season": [2020, 2021, 2022],
            "fantasy_points_ppr_positional_baseline": [12.0, 18.0, 22.0],
        })

        result = self.builder._add_career_features(
            df, positional_baseline_df, stat_columns=["fantasy_points_ppr"]
        )
        p1 = result[result["player_id"] == "p1"].sort_values("season").reset_index(drop=True)

        assert list(p1["years_played"]) == [1, 2, 3]
        assert list(p1["fantasy_points_ppr_career_avg"]) == [10.0, 15.0, 20.0]
        assert list(p1["fantasy_points_ppr_career_max"]) == [10.0, 20.0, 30.0]
        assert list(p1["fantasy_points_ppr_career_min"]) == [10.0, 10.0, 10.0]
        assert list(p1["fantasy_points_ppr_trend"]) == [0.0, 5.0, 10.0]

        # career_std is undefined with 1 data point; filled with 0 rather than left NaN
        assert p1["fantasy_points_ppr_career_std"].iloc[0] == 0.0
        assert p1["fantasy_points_ppr_career_std"].iloc[1] == pytest.approx(7.0710678118654755)
        assert p1["fantasy_points_ppr_career_std"].iloc[2] == pytest.approx(10.0)

        # p2's single, very different season shouldn't leak into p1's expanding stats
        p2 = result[result["player_id"] == "p2"]
        assert p2["years_played"].iloc[0] == 1
        assert p2["fantasy_points_ppr_career_avg"].iloc[0] == 1000.0

    def test_add_career_features__shrinkage_blends_toward_positional_baseline(self):
        df = pd.DataFrame({
            "player_id": ["p1", "p1", "p1"],
            "position": ["WR", "WR", "WR"],
            "season": [2020, 2021, 2022],
            "fantasy_points_ppr": [10.0, 20.0, 30.0],
        })
        positional_baseline_df = pd.DataFrame({
            "position": ["WR", "WR", "WR"],
            "season": [2020, 2021, 2022],
            "fantasy_points_ppr_positional_baseline": [12.0, 18.0, 22.0],
        })

        result = self.builder._add_career_features(
            df, positional_baseline_df, stat_columns=["fantasy_points_ppr"], shrinkage_k=3.0
        ).sort_values("season").reset_index(drop=True)

        # weight = years_played / (years_played + 3): 0.25, 0.4, 0.5
        expected_shrunk = [
            0.25 * 10.0 + 0.75 * 12.0,
            0.4 * 15.0 + 0.6 * 18.0,
            0.5 * 20.0 + 0.5 * 22.0,
        ]
        assert list(result["fantasy_points_ppr_shrunk_avg"]) == expected_shrunk
