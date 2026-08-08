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

        draft_picks_df = pd.DataFrame({
            "season": [2022, 2022, 2022, 2023],
            "round": [1, 1, 2, 1],
            "pick": [1, 2, 33, 1],
            "team": ["ARI", "ARI", "DAL", "STL"],
            "gsis_id": ["p1", "p2", "p3", "p6"],
            "pfr_player_id": ["OneP00", "TwoP00", "ThreP00", "SixP00"],
            "cfb_player_id": ["one-1", "two-1", "three-1", "six-1"],
            "pfr_player_name": ["Player One", "Player Two", "Player Three", "Player Six"],
            "college": ["State", "Tech", "U", "A&M"],
            "position": ["QB", "K", "WR", "DE"],
            "category": ["QB", "K", "WR", "DL"],
            "side": ["O", "S", "O", "D"],
            "age": [22.0, 23.0, 21.0, 24.0],
            "hof": [False, False, False, False],
            "to": [2030.0, 2025.0, 2032.0, 2028.0],
            "allpro": [0, 0, 1, 0],
            "probowls": [0, 0, 1, 0],
            "seasons_started": [2, 0, 3, 1],
            "w_av": [10.0, 2.0, 15.0, 8.0],
            "dr_av": [10.0, 2.0, 15.0, 8.0],
            "car_av": [None, None, None, None],
            "games": [16.0, 10.0, 40.0, 32.0],
            "pass_completions": [300.0, 0.0, 0.0, 0.0],
            "pass_attempts": [500.0, 0.0, 0.0, 0.0],
            "pass_yards": [3500.0, 0.0, 0.0, 0.0],
            "pass_tds": [25.0, 0.0, 0.0, 0.0],
            "pass_ints": [10.0, 0.0, 0.0, 0.0],
            "rush_atts": [20.0, 0.0, 5.0, 0.0],
            "rush_yards": [100.0, 0.0, 20.0, 0.0],
            "rush_tds": [1.0, 0.0, 0.0, 0.0],
            "receptions": [0.0, 0.0, 60.0, 0.0],
            "rec_yards": [0.0, 0.0, 800.0, 0.0],
            "rec_tds": [0.0, 0.0, 6.0, 0.0],
            "def_solo_tackles": [None, None, None, 50.0],
            "def_ints": [None, None, None, 1.0],
            "def_sacks": [None, None, None, 5.0],
        })
        draft_picks_df.to_csv(os.path.join(cls.bronze_dir, "draft_picks.csv"), index=False)

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

    def test_build_draft_picks__filters_by_position_and_drops_unneeded_columns(self):
        result = self.processor.build_draft_picks().reset_index(drop=True)

        # Of the 4 rows in the fixture, "K" (not fantasy relevant) and "DE" (defensive,
        # also carries the dropped def_* columns) should be filtered out, leaving QB/WR.
        # "team"/"position" (only needed for the position filter), "category"/"side",
        # identifying metadata (pfr_player_id/cfb_player_id/pfr_player_name/college),
        # career counting stats (games/pass_*/rush_*/receptions/rec_*), and career-final
        # totals (hof/to/allpro/probowls/seasons_started/w_av/dr_av) are all dropped,
        # leaving only identity + draft-time-safe stat columns. "gsis_id" is renamed to
        # "player_id" and "age" to "age_at_draft", and "round"/"pick" are combined into
        # "draft_pick" ((round - 1) * 32 + pick): row0 = (1-1)*32+1 = 1,
        # row2 (round 2, pick 33) = (2-1)*32+33 = 65.
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
                "team": ["TAM", "OAK", "ARI"],
                "gsis_id": ["bo_jackson", "bo_jackson", "p_other"],
                "pfr_player_id": ["JackBo00", "JackBo00", "OtherP00"],
                "cfb_player_id": ["a", "a", "b"],
                "pfr_player_name": ["Bo Jackson", "Bo Jackson", "Other Player"],
                "college": ["Auburn", "Auburn", "State"],
                "position": ["RB", "RB", "WR"],
                "category": ["RB", "RB", "WR"],
                "side": ["O", "O", "O"],
                "age": [23.0, 24.0, 21.0],
                "hof": [True, True, False],
                "to": [1990.0, 1990.0, 2029.0],
                "allpro": [0, 1, 0],
                "probowls": [1, 1, 0],
                "seasons_started": [0, 4, 3],
                "w_av": [0.0, 45.0, 12.0],
                "dr_av": [0.0, 45.0, 12.0],
                "car_av": [None, None, None],
                "games": [0.0, 38.0, 16.0],
                "pass_completions": [0.0, 0.0, 0.0],
                "pass_attempts": [0.0, 0.0, 0.0],
                "pass_yards": [0.0, 0.0, 0.0],
                "pass_tds": [0.0, 0.0, 0.0],
                "pass_ints": [0.0, 0.0, 0.0],
                "rush_atts": [0.0, 515.0, 0.0],
                "rush_yards": [0.0, 2782.0, 0.0],
                "rush_tds": [0.0, 16.0, 0.0],
                "receptions": [0.0, 40.0, 60.0],
                "rec_yards": [0.0, 352.0, 800.0],
                "rec_tds": [0.0, 2.0, 6.0],
                "def_solo_tackles": [None, None, None],
                "def_ints": [None, None, None],
                "def_sacks": [None, None, None],
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

    def test_build_draft_picks__keeps_rows_with_no_player_id(self):
        # Players nflverse never mapped to a gsis_id (e.g. career busts) can't be
        # de-duplicated by player, so all of their rows should be kept as-is.
        test_dir = tempfile.mkdtemp()
        try:
            bronze_dir = os.path.join(test_dir, "bronze", "nflv")
            os.makedirs(bronze_dir)
            draft_picks_df = pd.DataFrame({
                "season": [1984, 1985],
                "round": [10, 11],
                "pick": [5, 6],
                "team": ["ARI", "DAL"],
                "gsis_id": [None, None],
                "pfr_player_id": ["BustA00", "BustB00"],
                "cfb_player_id": ["a", "b"],
                "pfr_player_name": ["Bust A", "Bust B"],
                "college": ["State", "Tech"],
                "position": ["WR", "WR"],
                "category": ["WR", "WR"],
                "side": ["O", "O"],
                "age": [None, None],
                "hof": [False, False],
                "to": [None, None],
                "allpro": [0, 0],
                "probowls": [0, 0],
                "seasons_started": [0, 0],
                "w_av": [None, None],
                "dr_av": [None, None],
                "car_av": [None, None],
                "games": [None, None],
                "pass_completions": [None, None],
                "pass_attempts": [None, None],
                "pass_yards": [None, None],
                "pass_tds": [None, None],
                "pass_ints": [None, None],
                "rush_atts": [None, None],
                "rush_yards": [None, None],
                "rush_tds": [None, None],
                "receptions": [None, None],
                "rec_yards": [None, None],
                "rec_tds": [None, None],
                "def_solo_tackles": [None, None],
                "def_ints": [None, None],
                "def_sacks": [None, None],
            })
            draft_picks_df.to_csv(os.path.join(bronze_dir, "draft_picks.csv"), index=False)
            processor = NflverseProcessor(data_dir=test_dir)

            result = processor.build_draft_picks()

            assert len(result) == 2
            assert result["player_id"].isna().all()
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
