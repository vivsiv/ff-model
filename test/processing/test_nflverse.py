import os
import tempfile
import shutil

import pandas as pd
import pytest

from src.processing.nflverse import NflverseProcessor


class TestNflverseProcessor():
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.bronze_dir = os.path.join(cls.test_dir, "bronze", "nflv")
        os.makedirs(cls.bronze_dir)

        player_stats_df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3", "p4", "p5"],
            "player_display_name": ["Player One", "Player Two", "Player Three", "Player Four", "Player Five"],
            "position": ["QB", "RB", "K", "WR", "QB"],
            "recent_team": ["ARI", "ARI", "ARI", "DAL", "DAL"],
            "season": [2022, 2022, 2022, 2023, 2024],
            "fantasy_points": [100.0, 90.0, 10.0, 120.0, 150.0],
        })
        player_stats_df.to_csv(os.path.join(cls.bronze_dir, "player_stats.csv"), index=False)

        team_stats_df = pd.DataFrame({
            "team": ["ARI", "DAL", "ARI"],
            "season": [2022, 2023, 2024],
            "team_points": [300, 350, 320],
        })
        team_stats_df.to_csv(os.path.join(cls.bronze_dir, "team_stats.csv"), index=False)

        cls.processor = NflverseProcessor(data_dir=cls.test_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_init__raises_if_bronze_dir_missing(self):
        with pytest.raises(FileNotFoundError):
            NflverseProcessor(data_dir=os.path.join(self.test_dir, "does_not_exist"))

    def test_init__creates_silver_dir(self):
        assert os.path.exists(self.processor.silver_dir)

    def test_load_bronze__raises_if_file_missing(self):
        with pytest.raises(FileNotFoundError):
            self.processor._load_bronze("does_not_exist.csv")

    def test_load_bronze__loads_full_file_as_is(self):
        result = self.processor._load_bronze("player_stats.csv")

        expected = pd.DataFrame({
            "player_id": ["p1", "p2", "p3", "p4", "p5"],
            "player_display_name": ["Player One", "Player Two", "Player Three", "Player Four", "Player Five"],
            "position": ["QB", "RB", "K", "WR", "QB"],
            "recent_team": ["ARI", "ARI", "ARI", "DAL", "DAL"],
            "season": [2022, 2022, 2022, 2023, 2024],
            "fantasy_points": [100.0, 90.0, 10.0, 120.0, 150.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_build_player_stats__filters_by_position_and_saves_silver(self):
        result = self.processor.build_player_stats().reset_index(drop=True)

        expected = pd.DataFrame({
            "player_id": ["p1", "p2", "p4", "p5"],
            "player_display_name": ["Player One", "Player Two", "Player Four", "Player Five"],
            "position": ["QB", "RB", "WR", "QB"],
            "recent_team": ["ARI", "ARI", "DAL", "DAL"],
            "season": [2022, 2022, 2023, 2024],
            "fantasy_points": [100.0, 90.0, 120.0, 150.0],
        })
        pd.testing.assert_frame_equal(result, expected)

        silver_path = os.path.join(self.processor.silver_dir, "player_stats.csv")
        assert os.path.exists(silver_path)
        pd.testing.assert_frame_equal(pd.read_csv(silver_path), expected)

    def test_build_team_stats__saves_full_silver(self):
        result = self.processor.build_team_stats()

        expected = pd.DataFrame({
            "team": ["ARI", "DAL", "ARI"],
            "season": [2022, 2023, 2024],
            "team_points": [300, 350, 320],
        })
        pd.testing.assert_frame_equal(result, expected)

        silver_path = os.path.join(self.processor.silver_dir, "team_stats.csv")
        assert os.path.exists(silver_path)
        pd.testing.assert_frame_equal(pd.read_csv(silver_path), expected)

    def test_build_player_stats__normalizes_relocated_teams(self):
        test_dir = tempfile.mkdtemp()
        try:
            bronze_dir = os.path.join(test_dir, "bronze", "nflv")
            os.makedirs(bronze_dir)
            player_stats_df = pd.DataFrame({
                "player_id": ["p1", "p1", "p1"],
                "player_display_name": ["Player One", "Player One", "Player One"],
                "position": ["RB", "RB", "RB"],
                "recent_team": ["OAK", "LV", "SD"],
                "season": [2001, 2003, 2001],
                "fantasy_points": [50.0, 60.0, 70.0],
            })
            player_stats_df.to_csv(os.path.join(bronze_dir, "player_stats.csv"), index=False)
            processor = NflverseProcessor(data_dir=test_dir)

            result = processor.build_player_stats().reset_index(drop=True)

            assert list(result["recent_team"]) == ["LV", "LV", "LAC"]
        finally:
            shutil.rmtree(test_dir)

    def test_build_team_stats__normalizes_relocated_teams(self):
        test_dir = tempfile.mkdtemp()
        try:
            bronze_dir = os.path.join(test_dir, "bronze", "nflv")
            os.makedirs(bronze_dir)
            team_stats_df = pd.DataFrame({
                "team": ["OAK", "LV", "STL", "JAC", "JAX"],
                "season": [2002, 2003, 2002, 2001, 2003],
                "team_points": [300, 310, 320, 330, 340],
            })
            team_stats_df.to_csv(os.path.join(bronze_dir, "team_stats.csv"), index=False)
            processor = NflverseProcessor(data_dir=test_dir)

            result = processor.build_team_stats().reset_index(drop=True)

            assert list(result["team"]) == ["LV", "LV", "LA", "JAX", "JAX"]
        finally:
            shutil.rmtree(test_dir)
