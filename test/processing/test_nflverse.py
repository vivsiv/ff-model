import os
import tempfile
import shutil

import numpy as np
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
            "games": [16, 15, 16, 10, 8],
            "fantasy_points": [100.0, 90.0, 10.0, 120.0, 150.0],
            "fantasy_points_ppr": [120.0, 105.0, 10.0, 150.0, 160.0],
        })
        player_stats_df.to_csv(os.path.join(cls.bronze_dir, "player_stats.csv"), index=False)

        team_stats_df = pd.DataFrame({
            "team": ["ARI", "DAL", "ARI"],
            "season": [2022, 2023, 2024],
            "team_points": [300, 350, 320],
        })
        team_stats_df.to_csv(os.path.join(cls.bronze_dir, "team_stats.csv"), index=False)

        draft_picks_df = pd.DataFrame({
            "season": [2022, 2022, 2022, 2023],
            "round": [1, 1, 2, 1],
            "pick": [1, 2, 33, 1],
            "team": ["ARI", "ARI", "DAL", "STL"],
            "gsis_id": ["p1", "p2", "p3", "p6"],
            "position": ["QB", "K", "WR", "DE"],
            "category": ["QB", "K", "WR", "DL"],
            "side": ["O", "S", "O", "D"],
            "age": [22.0, 23.0, 21.0, 24.0],
            "college": ["State", "Tech", "U", "A&M"],
            "games": [16.0, 10.0, 40.0, 32.0],
            "hof": [False, False, False, False],
        })
        draft_picks_df.to_csv(os.path.join(cls.bronze_dir, "draft_picks.csv"), index=False)

        players_df = pd.DataFrame({
            "gsis_id": ["p1", "p2", "p4", "p5", "p_no_pfr"],
            "pfr_id": ["PfrP1", "PfrP2", "PfrP4", "PfrP5", None],
            "espn_id": [111, 222, 444, 555, 999],
        })
        players_df.to_csv(os.path.join(cls.bronze_dir, "players.csv"), index=False)

        snap_counts_df = pd.DataFrame({
            "pfr_player_id": ["PfrP1", "PfrP1", "PfrP1", "PfrP2", "PfrP4", "PfrUnknown"],
            "season": [2022, 2022, 2022, 2022, 2023, 2023],
            "game_type": ["REG", "REG", "WC", "REG", "REG", "REG"],
            "position": ["QB", "QB", "QB", "RB", "T", "WR"],
            "offense_snaps": [50.0, 60.0, 999.0, 20.0, 70.0, 30.0],
            "offense_pct": [0.8, 1.0, 1.0, 0.4, 1.0, 0.5],
        })
        snap_counts_df.to_csv(os.path.join(cls.bronze_dir, "snap_counts.csv"), index=False)

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
            "games": [16, 15, 16, 10, 8],
            "fantasy_points": [100.0, 90.0, 10.0, 120.0, 150.0],
            "fantasy_points_ppr": [120.0, 105.0, 10.0, 150.0, 160.0],
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
            "games": [16, 15, 10, 8],
            "fantasy_points": [100.0, 90.0, 120.0, 150.0],
            "fantasy_points_ppr": [120.0, 105.0, 150.0, 160.0],
            "ppr_points_per_game": [7.5, 7.0, 15.0, 20.0],
        })
        pd.testing.assert_frame_equal(result, expected)

        silver_path = os.path.join(self.processor.silver_dir, "player_stats.csv")
        assert os.path.exists(silver_path)
        pd.testing.assert_frame_equal(pd.read_csv(silver_path), expected)

    def test_build_player_stats__ppr_points_per_game_is_zero_when_games_is_zero(self):
        test_dir = tempfile.mkdtemp()
        try:
            bronze_dir = os.path.join(test_dir, "bronze", "nflv")
            os.makedirs(bronze_dir)
            player_stats_df = pd.DataFrame({
                "player_id": ["p1"],
                "player_display_name": ["Player One"],
                "position": ["WR"],
                "recent_team": ["ARI"],
                "season": [2022],
                "games": [0],
                "fantasy_points": [0.0],
                "fantasy_points_ppr": [0.0],
            })
            player_stats_df.to_csv(os.path.join(bronze_dir, "player_stats.csv"), index=False)
            processor = NflverseProcessor(data_dir=test_dir)

            result = processor.build_player_stats()

            # 0 games played -> 0 rather than NaN/inf, so the row still has a usable
            # target instead of being dropped from training/eval.
            assert result["ppr_points_per_game"].iloc[0] == 0.0
        finally:
            shutil.rmtree(test_dir)

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

    def test_build_draft_picks__filters_by_position_and_drops_unneeded_columns(self):
        result = self.processor.build_draft_picks().reset_index(drop=True)

        expected = pd.DataFrame({
            "season": [2022, 2022],
            "player_id": ["p1", "p3"],
            "age_at_draft": [22.0, 21.0],
            "draft_pick": [1, 65],
        })

        pd.testing.assert_frame_equal(result, expected)

        silver_path = os.path.join(self.processor.silver_dir, "draft_picks.csv")
        assert os.path.exists(silver_path)
        pd.testing.assert_frame_equal(pd.read_csv(silver_path), expected)

    def test_build_draft_picks__drops_team_and_position(self):
        # "team" and "position" are only used to filter to fantasy-relevant positions and
        # aren't needed downstream (player_stats has its own up-to-date versions of both).
        result = self.processor.build_draft_picks()
        assert "team" not in result.columns
        assert "position" not in result.columns
        assert "category" not in result.columns
        assert "side" not in result.columns

    def test_build_draft_picks__deduplicates_players_drafted_more_than_once(self):
        # A player drafted, not signed, and re-drafted later (e.g. Bo Jackson) should end
        # up with a single row: the most recent (highest season) draft record.
        test_dir = tempfile.mkdtemp()
        try:
            bronze_dir = os.path.join(test_dir, "bronze", "nflv")
            os.makedirs(bronze_dir)
            draft_picks_df = pd.DataFrame({
                "season": [1986, 1987, 2020],
                "round": [1, 7, 3],
                "pick": [1, 1, 10],
                "gsis_id": ["bo_jackson", "bo_jackson", "p_other"],
                "position": ["RB", "RB", "WR"],
                "age": [23.0, 24.0, 21.0],
            })
            draft_picks_df.to_csv(os.path.join(bronze_dir, "draft_picks.csv"), index=False)
            processor = NflverseProcessor(data_dir=test_dir)

            result = processor.build_draft_picks().sort_values("player_id").reset_index(drop=True)

            assert len(result) == 2
            bo_row = result[result["player_id"] == "bo_jackson"].iloc[0]
            assert bo_row["season"] == 1987
            assert bo_row["draft_pick"] == 193  # (7 - 1) * 32 + 1
        finally:
            shutil.rmtree(test_dir)

    def test_build_draft_picks__drops_rows_with_no_player_id(self):
        # Players nflverse never mapped to a gsis_id (e.g. career busts) can't be joined
        # to player_stats, so they're useless downstream and should be dropped entirely.
        test_dir = tempfile.mkdtemp()
        try:
            bronze_dir = os.path.join(test_dir, "bronze", "nflv")
            os.makedirs(bronze_dir)
            draft_picks_df = pd.DataFrame({
                "season": [1984, 1985],
                "round": [10, 11],
                "pick": [5, 6],
                "gsis_id": [None, None],
                "position": ["WR", "WR"],
                "age": [None, None],
            })
            draft_picks_df.to_csv(os.path.join(bronze_dir, "draft_picks.csv"), index=False)
            processor = NflverseProcessor(data_dir=test_dir)

            result = processor.build_draft_picks()

            assert len(result) == 0
        finally:
            shutil.rmtree(test_dir)

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
                "games": [16, 15, 16],
                "fantasy_points": [50.0, 60.0, 70.0],
                "fantasy_points_ppr": [55.0, 65.0, 75.0],
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

    def test_build_player_ids__renames_gsis_id_and_keeps_every_other_column_and_row(self):
        result = self.processor.build_player_ids().reset_index(drop=True)

        # All 5 rows are kept, including "p_no_pfr" (no pfr_id) -- this is a general ID
        # crosswalk, not filtered down to just what build_snap_counts needs today. "espn_id"
        # (standing in for any non-pfr ID column) is preserved untouched.
        expected = pd.DataFrame({
            "player_id": ["p1", "p2", "p4", "p5", "p_no_pfr"],
            "pfr_id": ["PfrP1", "PfrP2", "PfrP4", "PfrP5", np.nan],
            "espn_id": [111, 222, 444, 555, 999],
        })
        pd.testing.assert_frame_equal(result, expected)

        silver_path = os.path.join(self.processor.silver_dir, "player_ids.csv")
        assert os.path.exists(silver_path)
        pd.testing.assert_frame_equal(pd.read_csv(silver_path), expected)

    def test_build_player_ids__drops_rows_with_no_player_id(self):
        test_dir = tempfile.mkdtemp()
        try:
            bronze_dir = os.path.join(test_dir, "bronze", "nflv")
            os.makedirs(bronze_dir)
            players_df = pd.DataFrame({
                "gsis_id": ["p1", None],
                "pfr_id": ["PfrP1", "PfrOrphan"],
            })
            players_df.to_csv(os.path.join(bronze_dir, "players.csv"), index=False)
            processor = NflverseProcessor(data_dir=test_dir)

            result = processor.build_player_ids()

            assert len(result) == 1
            assert result["player_id"].iloc[0] == "p1"
        finally:
            shutil.rmtree(test_dir)

    def test_build_snap_counts__collapses_per_game_rows_to_one_row_per_player_season(self):
        player_ids_df = self.processor.build_player_ids()
        result = self.processor.build_snap_counts(player_ids_df).sort_values("player_id").reset_index(drop=True)

        # p1: two REG games (50, 60 snaps / 0.8, 1.0 pct) -- summed/averaged. Its third row
        # (WC, 999 snaps) is a playoff game and must be excluded entirely.
        # p2: single REG game, passed through as-is.
        # p4's only row (position "T", an offensive lineman) is filtered out entirely, so p4
        # doesn't appear in the output.
        # PfrUnknown has no match in player_ids, so it's dropped rather than crashing.
        expected = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "season": [2022, 2022],
            "offense_snaps": [110.0, 20.0],
            "offense_pct": [0.9, 0.4],
        })
        pd.testing.assert_frame_equal(result, expected)

        silver_path = os.path.join(self.processor.silver_dir, "snap_counts.csv")
        assert os.path.exists(silver_path)
        pd.testing.assert_frame_equal(pd.read_csv(silver_path), expected)

    def test_build_snap_counts__accepts_a_precomputed_player_ids_df(self):
        player_ids_df = pd.DataFrame({
            "player_id": ["p2"],
            "pfr_id": ["PfrP2"],
        })

        result = self.processor.build_snap_counts(player_ids_df).reset_index(drop=True)

        # Only PfrP2 is in the supplied mapping, so p1's rows (which do exist in bronze) are
        # dropped just as if they'd never matched -- confirms the passed-in df is actually
        # used instead of build_player_ids() being called internally.
        expected = pd.DataFrame({
            "player_id": ["p2"],
            "season": [2022],
            "offense_snaps": [20.0],
            "offense_pct": [0.4],
        })
        pd.testing.assert_frame_equal(result, expected)
