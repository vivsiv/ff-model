import os
import tempfile
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.processing.gold import (
    GoldSetBuilder,
    load_gold_set,
    gold_set_filename,
    main,
)


class TestGoldSetBuilder():
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.test_dir, "silver", "nflv"))
        cls.builder = GoldSetBuilder(data_dir=cls.test_dir)

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
            player_stats_df=df, positional_baseline_df=positional_baseline_df,
            stat_columns=["fantasy_points_ppr"], shrinkage_k=3.0
        )
        expected = pd.DataFrame({
            "player_id": ["p1", "p1", "p1", "p2"],
            "position": ["WR"] * 4,
            "season": [2020, 2021, 2022, 2020],
            "fantasy_points_ppr": [10.0, 20.0, 30.0, 1000.0],
            "years_played": [1, 2, 3, 1],
            "fantasy_points_ppr_career_avg": [10.0, 15.0, 20.0, 1000.0],
            "fantasy_points_ppr_career_std": [0.0, np.sqrt(50), 10.0, 0.0],
            "fantasy_points_ppr_career_max": [10.0, 20.0, 30.0, 1000.0],
            "fantasy_points_ppr_career_min": [10.0, 10.0, 10.0, 1000.0],
            "fantasy_points_ppr_trend": [0.0, 5.0, 10.0, 0.0],
            "fantasy_points_ppr_positional_baseline": [12.0, 18.0, 22.0, 12.0],
            "fantasy_points_ppr_shrunk_avg": [11.5, 16.8, 21.0, 259.0],
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_add_career_features__zero_shrinkage_preserves_career_average(self):
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
            df, positional_baseline_df, stat_columns=["fantasy_points_ppr"], shrinkage_k=0.0
        ).sort_values("season").reset_index(drop=True)

        # k=0 gives the career average full weight regardless of career length.
        expected = pd.DataFrame({"fantasy_points_ppr_shrunk_avg": [10.0, 15.0, 20.0]})
        pd.testing.assert_frame_equal(result[expected.columns], expected)

    def test_round_significant_figures__rounds_to_requested_precision(self):
        df = pd.DataFrame({"stat": [22.35634, 0.0123123, 1234567.0, -9.87654]})

        result = self.builder._round_significant_figures(df, sig_figs=4)

        assert list(result["stat"]) == pytest.approx([22.36, 0.01231, 1235000.0, -9.877])

    def test_round_significant_figures__preserves_zero_and_nan(self):
        df = pd.DataFrame({"stat": [0.0, np.nan, 5.55555]})

        result = self.builder._round_significant_figures(df, sig_figs=4)

        assert result["stat"].iloc[0] == 0.0
        assert pd.isna(result["stat"].iloc[1])
        assert result["stat"].iloc[2] == pytest.approx(5.556)

    def test_round_significant_figures__leaves_excluded_columns_untouched(self):
        df = pd.DataFrame({"season": [1999.0], "stat": [22.35634]})

        result = self.builder._round_significant_figures(df, sig_figs=4, exclude_columns=["season"])

        assert result["season"].iloc[0] == 1999.0
        assert result["stat"].iloc[0] == pytest.approx(22.36)

    def test_round_significant_figures__leaves_non_numeric_columns_untouched(self):
        df = pd.DataFrame({"player_id": ["p1"], "stat": [22.35634]})

        result = self.builder._round_significant_figures(df, sig_figs=4)

        assert result["player_id"].iloc[0] == "p1"

    def test_round_significant_figures__does_not_mutate_input(self):
        df = pd.DataFrame({"stat": [22.35634]})

        self.builder._round_significant_figures(df, sig_figs=4)

        assert df["stat"].iloc[0] == 22.35634

    def test_add_per_game_stats__divides_counting_stats_by_games(self):
        df = pd.DataFrame({
            "games": [10, 16],
            "receiving_yards": [500.0, 1280.0],
        })

        result_df, per_game_columns = self.builder._add_per_game_stats(df, counting_stat_columns=["receiving_yards"])

        assert per_game_columns == ["receiving_yards_per_game"]
        assert list(result_df["receiving_yards_per_game"]) == [50.0, 80.0]

    def test_add_per_game_stats__zero_games_yields_zero_not_nan_or_inf(self):
        df = pd.DataFrame({
            "games": [0],
            "receiving_yards": [500.0],
        })

        result_df, _ = self.builder._add_per_game_stats(df, counting_stat_columns=["receiving_yards"])

        assert result_df["receiving_yards_per_game"].iloc[0] == 0.0

    def test_add_per_game_stats__excludes_the_games_column_itself(self):
        df = pd.DataFrame({
            "games": [10],
            "receiving_yards": [500.0],
        })

        result_df, per_game_columns = self.builder._add_per_game_stats(
            df, counting_stat_columns=["games", "receiving_yards"]
        )

        assert "games_per_game" not in result_df.columns
        assert per_game_columns == ["receiving_yards_per_game"]

    # TODO: replace these heavier weight tests with a golden set integration test
    def test_load_player_stat_features__merges_snap_counts_and_extends_career_features_to_them(self):
        # Registry lookups are mocked to a small fixed set of columns rather than the real
        # (much larger) player_stats registry, so this test doesn't need to fabricate every
        # real stat column just to exercise the merge.
        test_dir = tempfile.mkdtemp()
        try:
            silver_dir = os.path.join(test_dir, "silver", "nflv")
            os.makedirs(silver_dir)

            player_stats_df = pd.DataFrame({
                "player_id": ["p1", "p1", "p2"],
                "player_display_name": ["Player One", "Player One", "Player Two"],
                "position": ["WR", "WR", "WR"],
                "season": [2013, 2014, 2013],
                "recent_team": ["KC", "KC", "SF"],
                "fantasy_points_ppr": [100.0, 150.0, 80.0],
            })
            player_stats_df.to_csv(os.path.join(silver_dir, "player_stats.csv"), index=False)

            # Only p1/2013 has a snap count match: p1/2014 (no snap data that season) and
            # p2 (no snap count row at all) should end up NaN, not dropped or crash.
            snap_counts_df = pd.DataFrame({
                "player_id": ["p1"],
                "season": [2013],
                "offense_snaps": [500.0],
                "offense_pct": [0.75],
            })
            snap_counts_df.to_csv(os.path.join(silver_dir, "snap_counts.csv"), index=False)

            identity_columns = {
                ("nflverse", "player_stats"): ["player_id", "player_display_name", "position", "season", "recent_team"],
                ("nflverse", "snap_counts"): ["player_id", "season"],
            }
            stat_columns = {
                ("nflverse", "player_stats"): ["fantasy_points_ppr"],
                ("nflverse", "snap_counts"): ["offense_snaps", "offense_pct"],
            }
            # No counting stats for this test -- it's about the merge, not the per-game transform.
            counting_stat_columns = {
                ("nflverse", "player_stats"): [],
                ("nflverse", "snap_counts"): [],
            }

            builder = GoldSetBuilder(data_dir=test_dir)
            with patch("src.processing.gold.get_identity_columns", side_effect=lambda s, t: identity_columns[(s, t)]), \
                 patch("src.processing.gold.get_stat_columns", side_effect=lambda s, t: stat_columns[(s, t)]), \
                 patch("src.processing.gold.get_counting_stat_columns", side_effect=lambda s, t: counting_stat_columns[(s, t)]):
                result = builder.load_player_stat_features()

            p1 = result[result["player_id"] == "p1"].sort_values("season").reset_index(drop=True)
            assert p1["offense_snaps"].iloc[0] == 500.0
            assert p1["offense_pct"].iloc[0] == 0.75
            assert pd.isna(p1["offense_snaps"].iloc[1])
            # snap-count columns get the same career-average treatment as any other stat:
            # 2014's career_avg is computed from only 2013's real value, ignoring its own NaN.
            assert p1["offense_snaps_career_avg"].iloc[1] == 500.0

            p2 = result[result["player_id"] == "p2"]
            assert len(p2) == 1
            assert pd.isna(p2["offense_snaps"].iloc[0])
        finally:
            shutil.rmtree(test_dir)

    def test_load_player_stat_features__adds_per_game_career_features_for_counting_stats(self):
        test_dir = tempfile.mkdtemp()
        try:
            silver_dir = os.path.join(test_dir, "silver", "nflv")
            os.makedirs(silver_dir)

            player_stats_df = pd.DataFrame({
                "player_id": ["p1", "p1"],
                "player_display_name": ["Player One", "Player One"],
                "position": ["WR", "WR"],
                "season": [2020, 2021],
                "recent_team": ["KC", "KC"],
                "games": [10, 16],
                "receiving_yards": [500.0, 1280.0],
                "target_share": [0.2, 0.25],  # rate stat -- should not get a per-game variant
            })
            player_stats_df.to_csv(os.path.join(silver_dir, "player_stats.csv"), index=False)
            pd.DataFrame({"player_id": [], "season": []}).to_csv(os.path.join(silver_dir, "snap_counts.csv"), index=False)

            identity_columns = {
                ("nflverse", "player_stats"): ["player_id", "player_display_name", "position", "season", "recent_team"],
                ("nflverse", "snap_counts"): ["player_id", "season"],
            }
            stat_columns = {
                ("nflverse", "player_stats"): ["games", "receiving_yards", "target_share"],
                ("nflverse", "snap_counts"): [],
            }
            counting_stat_columns = {
                ("nflverse", "player_stats"): ["games", "receiving_yards"],
                ("nflverse", "snap_counts"): [],
            }

            builder = GoldSetBuilder(data_dir=test_dir)
            with patch("src.processing.gold.get_identity_columns", side_effect=lambda s, t: identity_columns[(s, t)]), \
                 patch("src.processing.gold.get_stat_columns", side_effect=lambda s, t: stat_columns[(s, t)]), \
                 patch("src.processing.gold.get_counting_stat_columns", side_effect=lambda s, t: counting_stat_columns[(s, t)]):
                result = builder.load_player_stat_features().sort_values("season").reset_index(drop=True)

            # raw per-game column kept as a real feature (per the "keep it, simpler codewise"
            # decision), not just used transiently to derive career features.
            assert list(result["receiving_yards_per_game"]) == [50.0, 80.0]
            # 2021's career_avg is the expanding mean of per-game rates (50, then (50+80)/2),
            # not of season totals -- a shortened prior season doesn't drag it down as hard.
            assert list(result["receiving_yards_per_game_career_avg"]) == [50.0, 65.0]
            assert result["receiving_yards_per_game_trend"].iloc[1] == pytest.approx(15.0)
            # "games" itself never gets a "games_per_game" column -- dividing it by itself is meaningless.
            assert "games_per_game" not in result.columns
            # rate stats (target_share) don't get a per-game variant at all.
            assert "target_share_per_game" not in result.columns
        finally:
            shutil.rmtree(test_dir)

    def test_join_draft_features__computes_age_as_of_target_season(self):
        df = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "target_season": [2021, 2022],
        })
        draft_features_df = pd.DataFrame({
            "player_id": ["p1"],
            "draft_season": [2018],
            "age_at_draft": [21],
            "draft_pick": [15],
        })

        result = self.builder._join_draft_features(df, draft_features_df)

        # age = age_at_draft + (target_season - draft_season): 21 + (2021-2018) = 24,
        # 21 + (2022-2018) = 25 -- age changes year over year, draft_pick never does.
        assert list(result["age"]) == [24, 25]
        assert list(result["draft_pick"]) == [15, 15]
        assert "draft_season" not in result.columns
        assert "age_at_draft" not in result.columns

    def test_join_draft_features__draft_pick_is_static_across_a_players_rows(self):
        # Explicitly guards against draft_pick ever being run through career-averaging --
        # it must come through untouched and identical for every row of the same player.
        df = pd.DataFrame({
            "player_id": ["p1", "p1", "p1"],
            "target_season": [2019, 2020, 2021],
        })
        draft_features_df = pd.DataFrame({
            "player_id": ["p1"],
            "draft_season": [2017],
            "age_at_draft": [22],
            "draft_pick": [88],
        })

        result = self.builder._join_draft_features(df, draft_features_df)

        assert result["draft_pick"].nunique() == 1
        assert list(result["draft_pick"]) == [88, 88, 88]

    # TODO: is this right? shouldnt undrafted players have a pick higher than the last pick in the draft.
    def test_join_draft_features__undrafted_players_get_nan(self):
        df = pd.DataFrame({
            "player_id": ["p1"],
            "target_season": [2021],
        })
        draft_features_df = pd.DataFrame({
            "player_id": ["p_other"],
            "draft_season": [2018],
            "age_at_draft": [21],
            "draft_pick": [15],
        })

        result = self.builder._join_draft_features(df, draft_features_df)

        assert pd.isna(result["draft_pick"].iloc[0])
        assert pd.isna(result["age"].iloc[0])

    def test_join_draft_features__does_not_mix_players(self):
        df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "target_season": [2021, 2021],
        })
        draft_features_df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "draft_season": [2018, 2015],
            "age_at_draft": [21, 23],
            "draft_pick": [15, 200],
        })

        result = self.builder._join_draft_features(df, draft_features_df).sort_values("player_id").reset_index(drop=True)

        assert list(result["draft_pick"]) == [15, 200]
        assert list(result["age"]) == [24, 29]  # 21+(2021-2018)=24, 23+(2021-2015)=29

    def test_league_average_team_stats__single_season_mean_not_trailing_window(self):
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF", "DAL"],
            "season": [2020, 2020, 2021],
            "passing_yards": [4000.0, 3000.0, 5000.0],
        })

        result = self.builder._league_average_team_stats(team_features_df, ["passing_yards"])

        expected = pd.DataFrame({
            "season": [2020, 2021],
            "passing_yards_league_avg": [3500.0, 5000.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__no_team_change_has_zero_shift(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"],
            "season": [2020],
            "target_season": [2021],
            "recent_team": ["KC"],
        })
        player_team_by_season = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "season": [2020, 2021],
            "recent_team": ["KC", "KC"],
        })
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF"],
            "season": [2020, 2020],
            "passing_yards": [4000.0, 3500.0],
        })

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        # league avg for 2020 = mean(4000, 3500) = 3750; destination (KC) = 4000
        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2020], "target_season": [2021], "recent_team": ["KC"],
            "team_passing_yards": [250.0], "team_shift_passing_yards": [0.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__team_change_looks_up_destination_team_from_origin_season(self):
        # Mirrors the Davante Adams motivation: player's own feature season (2022) was with
        # the Raiders, but they're actually on the Rams by target_season (2023). Destination
        # stats must come from the Rams' 2022 (origin season) performance, not 2023's.
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"],
            "season": [2022],
            "target_season": [2023],
            "recent_team": ["LV"],
        })
        player_team_by_season = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "season": [2022, 2023],
            "recent_team": ["LV", "LA"],
        })
        team_features_df = pd.DataFrame({
            "team": ["LV", "LA", "LA"],
            "season": [2022, 2022, 2023],
            "passing_yards": [3000.0, 4200.0, 9999.0],
        })

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        # league avg for the *origin* season (2022) = mean(3000, 4200) = 3600, using only
        # 2022 rows -- LA's 9999 in 2023 must not leak into it. destination (LA, 2022) = 4200.
        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2022], "target_season": [2023], "recent_team": ["LV"],
            "team_passing_yards": [600.0], "team_shift_passing_yards": [1200.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__falls_back_to_origin_team_when_current_lookup_is_missing(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"],
            "season": [2025],
            "target_season": [2026],
            "recent_team": ["KC"],
        })
        player_team_by_season = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "recent_team": ["KC"],
        })
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF"],
            "season": [2025, 2025],
            "passing_yards": [4000.0, 3000.0],
        })

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
            "team_passing_yards": [500.0], "team_shift_passing_yards": [0.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__free_agent_falls_back_to_origin_team(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
        })
        player_team_by_season = pd.DataFrame({
            "player_id": ["p1", "p1"], "season": [2025, 2026], "recent_team": ["KC", np.nan],
        })
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF"], "season": [2025, 2025], "passing_yards": [4000.0, 3000.0],
        })

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
            "team_passing_yards": [500.0], "team_shift_passing_yards": [0.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__missing_destination_stats_use_feature_season_league_average(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
        })
        player_team_by_season = pd.DataFrame({"player_id": ["p1"], "season": [2026], "recent_team": ["NE"]})
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF", "NE"], "season": [2025, 2025, 2026],
            "passing_yards": [4000.0, 2000.0, 9999.0],
        })

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        # NE has no 2025 stats: use the 2025 mean (3000), never NE's 2026 result.
        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
            "team_passing_yards": [0.0], "team_shift_passing_yards": [-1000.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__missing_origin_stats_use_feature_season_league_average(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["NE"],
        })
        player_team_by_season = pd.DataFrame({"player_id": ["p1"], "season": [2026], "recent_team": ["SF"]})
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF", "NE"], "season": [2025, 2025, 2026],
            "passing_yards": [4000.0, 2000.0, 9999.0],
        })

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["NE"],
            "team_passing_yards": [-1000.0], "team_shift_passing_yards": [-1000.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__missing_both_teams_produces_average_level_and_zero_shift(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["LV"],
        })
        player_team_by_season = pd.DataFrame({"player_id": ["p1"], "season": [2026], "recent_team": ["LA"]})
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF"], "season": [2025, 2025], "passing_yards": [4000.0, 2000.0],
        })

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["LV"],
            "team_passing_yards": [0.0], "team_shift_passing_yards": [0.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__missing_stat_on_existing_team_uses_league_average(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
        })
        player_team_by_season = pd.DataFrame({"player_id": ["p1"], "season": [2026], "recent_team": ["NE"]})
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF", "NE"], "season": [2025, 2025, 2025],
            "passing_yards": [4000.0, 2000.0, np.nan],
        })

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
            "team_passing_yards": [0.0], "team_shift_passing_yards": [-1000.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__unavailable_feature_season_average_remains_nan(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
        })
        player_team_by_season = pd.DataFrame({"player_id": ["p1"], "season": [2026], "recent_team": ["NE"]})
        team_features_df = pd.DataFrame({"team": ["NE"], "season": [2026], "passing_yards": [9999.0]})

        result = self.builder._join_team_features(player_rows_df, player_team_by_season, team_features_df, ["passing_yards"])

        expected = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "target_season": [2026], "recent_team": ["KC"],
            "team_passing_yards": [np.nan], "team_shift_passing_yards": [np.nan],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_team_features__does_not_mix_players(self):
        player_rows_df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "season": [2020, 2020],
            "target_season": [2021, 2021],
            "recent_team": ["KC", "SF"],
        })
        player_team_by_season = pd.DataFrame({
            "player_id": ["p1", "p1", "p2", "p2"],
            "season": [2020, 2021, 2020, 2021],
            "recent_team": ["KC", "KC", "SF", "DAL"],
        })
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF", "DAL"],
            "season": [2020, 2020, 2020],
            "passing_yards": [4000.0, 3500.0, 3000.0],
        })

        result = self.builder._join_team_features(
            player_rows_df, player_team_by_season, team_features_df, ["passing_yards"]
        ).sort_values("player_id").reset_index(drop=True)

        # league avg for 2020 = mean(4000, 3500, 3000) = 3500.
        # p1 didn't change teams -> shift 0; p2 (SF -> DAL) shouldn't pick up p1's KC lookup.
        expected = pd.DataFrame({
            "player_id": ["p1", "p2"], "season": [2020, 2020], "target_season": [2021, 2021],
            "recent_team": ["KC", "SF"], "team_passing_yards": [500.0, -500.0],
            "team_shift_passing_yards": [0.0, -500.0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_with_target__pairs_each_season_with_the_prior_seasons_features(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p1"],
            "season": [2020, 2021, 2022],
            "fantasy_points_ppr": [10.0, 20.0, 30.0],
            "fantasy_points_ppr_career_avg": [10.0, 15.0, 20.0],
        })

        result = self.builder._join_with_target(
            features_df, target_col="fantasy_points_ppr"
        ).sort_values("season").reset_index(drop=True)

        # 2020's row has no prior season to be a feature source, so it only shows up as
        # a target (paired with 2019 features, which don't exist) -> excluded entirely.
        # 2020 features -> 2021 target; 2021 features -> 2022 target.
        expected = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "season": [2020, 2021],
            "fantasy_points_ppr": [10.0, 20.0],
            "fantasy_points_ppr_career_avg": [10.0, 15.0],
            "target_season": [2021, 2022],
            "target": [20.0, 30.0],
            "seasons_since_played": [0, 0],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_join_with_target__bridges_gap_seasons_using_most_recent_prior_data(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p1"],
            "season": [2019, 2020, 2022],  # 2021 missing entirely (e.g. hurt/out of league)
            "fantasy_points_ppr": [10.0, 20.0, 30.0],
        })

        result = self.builder._join_with_target(
            features_df, target_col="fantasy_points_ppr"
        ).sort_values("season").reset_index(drop=True)

        # 2019 -> 2020 pairs normally (seasons_since_played=0, no gap). 2020's features still
        # get used to predict 2022's target even though 2021 is missing entirely
        # (seasons_since_played=1, one season missed) -- a player who missed a season should
        # still be predictable from their last active one.
        assert list(result["season"]) == [2019, 2020]
        assert list(result["target_season"]) == [2020, 2022]
        assert list(result["target"]) == [20.0, 30.0]
        assert list(result["seasons_since_played"]) == [0, 1]

    def test_join_with_target__does_not_mix_players(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p2", "p2"],
            "season": [2020, 2021, 2020, 2021],
            "fantasy_points_ppr": [10.0, 20.0, 100.0, 200.0],
        })

        result = self.builder._join_with_target(
            features_df, target_col="fantasy_points_ppr"
        ).sort_values("player_id").reset_index(drop=True)

        assert list(result["player_id"]) == ["p1", "p2"]
        assert list(result["target"]) == [20.0, 200.0]
        # p2's 2020 feature row must never match against p1's target, even though they share
        # a season value -- merge_asof's `by` grouping needs to actually be respected.
        assert list(result["seasons_since_played"]) == [0, 0]

    def test_join_with_target__rejects_a_target_not_in_the_registry(self):
        with pytest.raises(AssertionError):
            self.builder._join_with_target(pd.DataFrame(), target_col="not_a_real_target")

    def test_prepare_player_rows__joins_historical_and_blank_current_targets_without_mutation(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p2", "retired", "retired", "p1"],
            "season": [2020, 2021, 2020, 2019, 2020, 2023],
            "recent_team": ["KC", "SF", "LV", "KC", "KC", "NE"],
            "fantasy_points_ppr": [10.0, 20.0, 100.0, 1.0, 2.0, 999.0],
            "fantasy_points_ppr_career_avg": [10.0, 15.0, 100.0, 1.0, 1.5, 999.0],
        })
        current_players_df = pd.DataFrame({
            "player_id": ["p1", "p2", "rookie"], "season": [2023] * 3,
            "recent_team": ["NE", "LA", "SF"],
        })
        original_features = features_df.copy(deep=True)
        original_players = current_players_df.copy(deep=True)

        rows = self.builder.prepare_player_rows(features_df, current_players_df, 2023, "fantasy_points_ppr")

        expected_rows = pd.DataFrame({
            "player_id": ["p1", "p1", "p2", "retired"],
            "season": [2020, 2021, 2020, 2019],
            "recent_team": ["KC", "SF", "LV", "KC"],
            "fantasy_points_ppr": [10.0, 20.0, 100.0, 1.0],
            "fantasy_points_ppr_career_avg": [10.0, 15.0, 100.0, 1.0],
            "target_season": [2021, 2023, 2023, 2020],
            "target": [20.0, np.nan, np.nan, 2.0],
            "seasons_since_played": [0, 1, 2, 0],
        })
        pd.testing.assert_frame_equal(rows.sort_values(["player_id", "target_season"]).reset_index(drop=True), expected_rows)
        pd.testing.assert_frame_equal(features_df, original_features)
        pd.testing.assert_frame_equal(current_players_df, original_players)

    def test_prepare_player_rows__current_rows_use_the_most_recent_season_per_player(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p1"],
            "season": [2020, 2021, 2022],
            "fantasy_points_ppr_career_avg": [10.0, 15.0, 20.0],
            "fantasy_points_ppr": [10.0, 20.0, 30.0],
            "recent_team": ["KC"] * 3,
        })
        players_df = pd.DataFrame({"player_id": ["p1"], "season": [2023], "recent_team": ["KC"]})

        result = self.builder.prepare_player_rows(features_df, players_df, 2023, "fantasy_points_ppr")
        expected = pd.DataFrame({
            "player_id": ["p1"] * 3,
            "season": [2020, 2021, 2022],
            "fantasy_points_ppr_career_avg": [10.0, 15.0, 20.0],
            "fantasy_points_ppr": [10.0, 20.0, 30.0],
            "recent_team": ["KC"] * 3,
            "target_season": [2021, 2022, 2023],
            "target": [20.0, 30.0, np.nan],
            "seasons_since_played": [0, 0, 0],
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_prepare_player_rows__ignores_current_and_future_season_stats(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p1", "rookie"],
            "season": [2025, 2026, 2027, 2026],
            "stat": [1.0, 999.0, 1000.0, 50.0],
            "fantasy_points_ppr": [1.0, 999.0, 1000.0, 50.0],
            "recent_team": ["KC"] * 4,
        })
        players_df = pd.DataFrame({
            "player_id": ["p1", "rookie"], "season": [2026, 2026], "recent_team": ["KC", "SF"],
        })
        result = self.builder.prepare_player_rows(features_df, players_df, 2026, "fantasy_points_ppr")
        expected = pd.DataFrame({
            "player_id": ["p1"],
            "season": [2025],
            "stat": [1.0],
            "fantasy_points_ppr": [1.0],
            "recent_team": ["KC"],
            "target_season": [2026],
            "target": [np.nan],
            "seasons_since_played": [0],
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_prepare_player_rows__listed_player_with_a_gap_still_gets_a_current_row(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "season": [2019, 2020],
            "fantasy_points_ppr_career_avg": [10.0, 15.0],
            "fantasy_points_ppr": [10.0, 20.0],
            "recent_team": ["KC", "KC"],
        })
        players_df = pd.DataFrame({"player_id": ["p1"], "season": [2023], "recent_team": ["KC"]})

        rows = self.builder.prepare_player_rows(features_df, players_df, 2023, "fantasy_points_ppr")
        result = rows[rows["target_season"] == 2023].reset_index(drop=True)

        # last played 2020, predicting 2023 -> 2021 and 2022 were missed (2 seasons)
        assert result["season"].iloc[0] == 2020
        assert result["target_season"].iloc[0] == 2023
        assert result["seasons_since_played"].iloc[0] == 2

    def test_prepare_player_rows__only_includes_listed_players_in_current_rows(self):
        features_df = pd.DataFrame({
            "player_id": ["retired", "active", "rookie_not_in_features"],
            "season": [2015, 2022, 2022],
            "fantasy_points_ppr_career_avg": [30.0, 20.0, 5.0],
            "fantasy_points_ppr": [30.0, 20.0, 5.0],
            "recent_team": ["KC"] * 3,
        })
        players_df = pd.DataFrame({
            "player_id": ["active", "rookie_with_no_stats"], "season": [2023, 2023], "recent_team": ["KC", "SF"]
        })

        rows = self.builder.prepare_player_rows(features_df, players_df, 2023, "fantasy_points_ppr")
        result = rows[rows["target_season"] == 2023]

        # The retired player isn't in players_df, the listed rookie has no prior season to build a row from
        assert list(result["player_id"]) == ["active"]

    def test_prepare_player_rows__does_not_mix_players(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p2", "p2", "p2"],
            "season": [2020, 2021, 2019, 2020, 2022],
            "fantasy_points_ppr_career_avg": [10.0, 20.0, 1.0, 2.0, 30.0],
            "fantasy_points_ppr": [10.0, 20.0, 1.0, 2.0, 30.0],
            "recent_team": ["KC"] * 5,
        })
        players_df = pd.DataFrame({
            "player_id": ["p1", "p2"], "season": [2023, 2023], "recent_team": ["KC", "SF"]
        })

        rows = self.builder.prepare_player_rows(features_df, players_df, 2023, "fantasy_points_ppr")
        result = rows[rows["target_season"] == 2023].sort_values("player_id").reset_index(drop=True)

        assert list(result["player_id"]) == ["p1", "p2"]
        assert list(result["season"]) == [2021, 2022]
        assert list(result["fantasy_points_ppr_career_avg"]) == [20.0, 30.0]
        assert list(result["seasons_since_played"]) == [1, 0]

    def test_build_player_team_by_season__uses_prior_and_current_teams_without_mutation(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "retired", "p1", "p1"],
            "season": [2020, 2021, 2019, 2023, 2024],
            "recent_team": ["KC", "SF", "LV", "CHI", "DAL"],
            "stat": [10.0, 20.0, 1.0, 999.0, 1000.0],
        })
        current_players_df = pd.DataFrame({
            "player_id": ["p1", "rookie", "free_agent"], "season": [2023] * 3,
            "recent_team": ["NE", "LA", np.nan], "extra": [100, 200, 300],
        })
        original_features = features_df.copy(deep=True)
        original_players = current_players_df.copy(deep=True)

        result = self.builder.build_player_team_by_season(features_df, current_players_df, 2023)

        expected = pd.DataFrame({
            "player_id": ["p1", "p1", "retired", "p1", "rookie", "free_agent"],
            "season": [2020, 2021, 2019, 2023, 2023, 2023],
            "recent_team": ["KC", "SF", "LV", "NE", "LA", np.nan],
        })
        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(features_df, original_features)
        pd.testing.assert_frame_equal(current_players_df, original_players)

    def test_build_player_team_by_season__works_without_performance_stats_or_targets(self):
        features_df = pd.DataFrame({"player_id": ["p1"], "season": [2025], "recent_team": ["KC"]})
        current_players_df = pd.DataFrame({"player_id": ["p1"], "season": [2026], "recent_team": ["SF"]})
        result = self.builder.build_player_team_by_season(features_df, current_players_df, 2026)
        expected = pd.DataFrame({"player_id": ["p1", "p1"], "season": [2025, 2026], "recent_team": ["KC", "SF"]})
        pd.testing.assert_frame_equal(result, expected)

    def test_build_gold_set__enriches_unrounded_historical_rows_without_saving(self, tmp_path):
        (tmp_path / "silver" / "nflv").mkdir(parents=True)
        builder = GoldSetBuilder(data_dir=str(tmp_path))
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "season": [2020, 2021],
            "recent_team": ["KC", "KC"],
            "fantasy_points_ppr": [10.12345, 20.12345],
        })
        team_features_df = pd.DataFrame({"team": ["KC"], "season": [2020], "passing_yards": [100.0]})
        draft_features_df = pd.DataFrame({
            "player_id": ["p1"], "draft_season": [2019], "draft_pick": [10], "age_at_draft": [21],
        })

        rows_df = builder._join_with_target(features_df, "fantasy_points_ppr")
        original = rows_df.copy(deep=True)
        player_team_by_season = features_df[["player_id", "season", "recent_team"]]
        with patch("src.processing.gold.get_stat_columns", return_value=["passing_yards"]):
            result = builder.build_gold_set(
                player_rows_df=rows_df, team_features_df=team_features_df,
                draft_features_df=draft_features_df, player_team_by_season=player_team_by_season,
            )

        assert list((tmp_path / "gold").iterdir()) == []
        assert result["season"].tolist() == [2020]
        assert result["target_season"].tolist() == [2021]
        assert result["target"].tolist() == [20.12345]
        assert result["fantasy_points_ppr"].tolist() == [10.12345]
        assert result["age"].tolist() == [23]
        assert result["team_shift_passing_yards"].tolist() == [0.0]
        pd.testing.assert_frame_equal(rows_df, original)

    def test_build_gold_set__enriches_unrounded_current_rows_without_saving(self, tmp_path):
        (tmp_path / "silver" / "nflv").mkdir(parents=True)
        builder = GoldSetBuilder(data_dir=str(tmp_path))
        features_df = pd.DataFrame({
            "player_id": ["p1"], "season": [2025], "recent_team": ["KC"],
            "fantasy_points_ppr": [10.12345],
        })
        team_df = pd.DataFrame({"team": ["KC"], "season": [2025], "passing_yards": [100.0]})
        draft_df = pd.DataFrame({
            "player_id": ["p1"], "draft_season": [2020], "draft_pick": [10], "age_at_draft": [21],
        })
        players_df = pd.DataFrame({"player_id": ["p1"], "season": [2026], "recent_team": ["KC"]})

        rows_df = builder.prepare_player_rows(features_df, players_df, 2026, "fantasy_points_ppr")
        player_team_by_season = builder.build_player_team_by_season(features_df, players_df, 2026)
        with patch("src.processing.gold.get_stat_columns", return_value=["passing_yards"]):
            result = builder.build_gold_set(rows_df, team_df, draft_df, player_team_by_season)

        assert list((tmp_path / "gold").iterdir()) == []
        assert result["fantasy_points_ppr"].tolist() == [10.12345]
        assert result["target"].isna().all()
        assert result["target_season"].tolist() == [2026]

    def test_save_to_gold__rounds_without_mutating_input_and_preserves_excluded_columns(self, tmp_path):
        (tmp_path / "silver" / "nflv").mkdir(parents=True)
        builder = GoldSetBuilder(data_dir=str(tmp_path))
        data = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "season": [2020, 2021],
            "target_season": [2021, 2022],
            "draft_pick": [12345, 12346],
            "target": [20.12345, np.nan],
            "stat": [10.12345, 0.0],
        })
        original = data.copy(deep=True)

        builder.save_to_gold(data, gold_set_filename("fantasy_points_ppr", 2022))

        saved = load_gold_set(builder.gold_dir, "fantasy_points_ppr", 2022)
        expected = data.copy()
        expected["target"] = [20.12, np.nan]
        expected["stat"] = [10.12, 0.0]
        pd.testing.assert_frame_equal(saved, expected)
        pd.testing.assert_frame_equal(data, original)

    @pytest.mark.parametrize("current_season", [2026, 2027])
    def test_load_current_players__sets_current_season_and_renames_team(self, current_season):
        test_dir = tempfile.mkdtemp()
        try:
            silver_dir = os.path.join(test_dir, "silver", "nflv")
            os.makedirs(silver_dir)
            pd.DataFrame({
                "player_id": ["p1", "p2"],
                "player_name": ["One", "Two"],
                "position": ["QB", "WR"],
                "team": ["KC", np.nan],
                "ecr": [1.0, 5.0],
            }).to_csv(os.path.join(silver_dir, "ff_rankings.csv"), index=False)

            builder = GoldSetBuilder(data_dir=test_dir)
            with patch("src.processing.gold.get_current_season", return_value=current_season) as current_season_mock:
                result = builder.load_current_players()
            current_season_mock.assert_called_once_with(roster=True)

            expected = pd.DataFrame({
                "player_id": ["p1", "p2"], "season": [current_season, current_season], "recent_team": ["KC", np.nan]
            })
            pd.testing.assert_frame_equal(result, expected)
        finally:
            shutil.rmtree(test_dir)


class TestValidateGoldData:
    @staticmethod
    def _historical_data():
        return pd.DataFrame({
            "player_id": ["p1", "p1", "p2"],
            "player_display_name": ["Player One", "Player One", "Player Two"],
            "position": ["WR", "WR", "RB"],
            "recent_team": ["KC", "KC", None],
            "season": [2020, 2021, 2020],
            "target_season": [2021, 2023, 2021],
            "target": [10.0, 20.0, 30.0],
            "feature": [1.0, np.nan, 3.0],
        })

    @classmethod
    def _current_data(cls):
        data = cls._historical_data().iloc[[0, 2]].copy().reset_index(drop=True)
        data["season"] = [2023, 2021]
        data["target_season"] = 2024
        data["target"] = np.nan
        return data

    def test_validates_combined_table_without_mutation(self):
        data = pd.concat([self._historical_data(), self._current_data()], ignore_index=True)
        original = data.copy(deep=True)
        GoldSetBuilder.validate_data(data, 2024)
        pd.testing.assert_frame_equal(data, original)

    def test_accepts_current_only_players_and_reordered_columns(self):
        historical_df = self._historical_data()
        current_df = self._current_data()
        current_df.loc[1, "player_id"] = "new_player"
        current_df = current_df[current_df.columns[::-1]]
        GoldSetBuilder.validate_data(pd.concat([historical_df, current_df], ignore_index=True), 2024)

    @pytest.mark.parametrize("is_current", [False, True])
    def test_rejects_non_numeric_features(self, is_current):
        data = self._current_data() if is_current else self._historical_data()
        data["team_name"] = "KC"
        with pytest.raises(ValueError, match="team_name"):
            GoldSetBuilder.validate_data(data, 2024)

    def test_rejects_missing_historical_targets(self):
        data = self._historical_data()
        data.loc[0, "target"] = np.nan
        with pytest.raises(ValueError, match="null target"):
            GoldSetBuilder.validate_data(data, 2024)

    @pytest.mark.parametrize("is_current", [False, True])
    def test_rejects_repeated_player_feature_season(self, is_current):
        data = self._current_data() if is_current else self._historical_data()
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="More than one row"):
            GoldSetBuilder.validate_data(data, 2024)

    def test_rejects_duplicate_player_feature_season(self):
        data = self._historical_data()
        data.loc[1, "season"] = data.loc[0, "season"]
        with pytest.raises(ValueError, match="More than one row"):
            GoldSetBuilder.validate_data(data, 2024)

    @pytest.mark.parametrize("feature_season", [2021, 2022], ids=["same-season", "future-season"])
    def test_rejects_historical_features_not_before_their_target_season(self, feature_season):
        # Both feature seasons are before the overall current season (2024), so
        # this must exercise the per-row target boundary rather than that cutoff.
        data = self._historical_data().iloc[[0]].copy()
        data["season"] = feature_season
        assert data["target_season"].tolist() == [2021]
        with pytest.raises(ValueError, match="strictly before target_season"):
            GoldSetBuilder.validate_data(data, 2024)

    def test_rejects_labeled_current_rows(self):
        data = self._current_data()
        data.loc[0, "target"] = 10.0
        with pytest.raises(ValueError, match="Current targets must be blank"):
            GoldSetBuilder.validate_data(data, 2024)

    def test_rejects_target_seasons_after_the_current_season(self):
        data = self._current_data()
        data.loc[0, "target_season"] = 2025
        with pytest.raises(ValueError, match="no later than current_season"):
            GoldSetBuilder.validate_data(data, 2024)

    def test_rejects_features_from_the_current_season(self):
        data = self._current_data()
        data.loc[0, "season"] = 2024
        with pytest.raises(ValueError, match="feature seasons must be before"):
            GoldSetBuilder.validate_data(data, 2024)

    @pytest.mark.parametrize("season", [2019, 2021])
    def test_rejects_stale_or_repeated_current_feature_season(self, season):
        current_df = self._current_data()
        current_df.loc[0, "season"] = season
        message = "More than one row" if season == 2021 else "later than all historical"
        with pytest.raises(ValueError, match=message):
            GoldSetBuilder.validate_data(pd.concat([self._historical_data(), current_df], ignore_index=True), 2024)


class TestGoldMain:
    def test_main__writes_combined_rows_and_preserves_an_older_year_file(self, tmp_path):
        silver_dir = tmp_path / "silver" / "nflv"
        silver_dir.mkdir(parents=True)
        pd.DataFrame({"player_id": ["p1"], "team": ["SF"]}).to_csv(silver_dir / "ff_rankings.csv", index=False)
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        old_file = gold_dir / "fantasy_points_ppr__2025.csv"
        old_file.write_text("previous year's snapshot\n")
        original_bytes = old_file.read_bytes()
        features_df = pd.DataFrame({
            "player_id": ["p1"] * 4, "season": [2023, 2024, 2025, 2026],
            "player_display_name": ["Player One"] * 4, "position": ["WR"] * 4,
            "recent_team": ["KC", "SF", "KC", "KC"], "fantasy_points_ppr": [10.0, 20.12345, 30.12345, 999.0],
        })
        team_df = pd.DataFrame({
            "team": ["KC"] * 3 + ["SF"] * 3,
            "season": [2023, 2024, 2025] * 2, "passing_yards": [100.0] * 3 + [300.0] * 3,
        })
        draft_df = pd.DataFrame({
            "player_id": ["p1"], "draft_season": [2020], "draft_pick": [10], "age_at_draft": [21],
        })
        with (
            patch("sys.argv", ["gold", "--data-dir", str(tmp_path)]),
            patch("src.processing.gold.get_current_season", return_value=2026),
            patch.object(GoldSetBuilder, "load_player_stat_features", return_value=features_df),
            patch.object(GoldSetBuilder, "load_team_features", return_value=team_df),
            patch.object(GoldSetBuilder, "load_draft_features", return_value=draft_df),
            patch("src.processing.gold.get_stat_columns", return_value=["passing_yards"]),
        ):
            main()

        saved = load_gold_set(str(gold_dir), "fantasy_points_ppr", 2026)
        assert saved["target_season"].tolist() == [2024, 2025, 2026]
        assert saved["season"].tolist() == [2023, 2024, 2025]
        assert saved["target"].iloc[:2].tolist() == [20.12, 30.12]
        assert pd.isna(saved["target"].iloc[2])
        assert saved["team_shift_passing_yards"].tolist() == [200.0, -200.0, 200.0]
        assert saved["team_passing_yards"].tolist() == [100.0, -100.0, 100.0]
        assert old_file.read_bytes() == original_bytes
        assert sorted(path.name for path in gold_dir.iterdir()) == [
            "fantasy_points_ppr__2025.csv", "fantasy_points_ppr__2026.csv",
        ]

    def test_main__prepares_player_rows_then_saves_one_combined_table(self):
        historical_df = pd.DataFrame({"target_season": [2025], "target": [20.12345]})
        current_df = pd.DataFrame({"target_season": [2026], "target": [np.nan]})
        features_df = pd.DataFrame({
            "player_id": ["p1"] * 4, "season": [2024, 2025, 2026, 2027], "recent_team": ["KC"] * 4,
        })
        with (
            patch("src.processing.gold.GoldSetBuilder") as builder_class,
            patch("sys.argv", ["gold", "--target-col", "ppr_points_per_game", "--data-dir", "test-data"]),
        ):
            builder = builder_class.return_value
            current_players_df = pd.DataFrame({"player_id": ["p1"], "season": [2026], "recent_team": ["SF"]})
            builder.load_current_players.return_value = current_players_df
            builder.load_player_stat_features.return_value = features_df
            prepared_rows = pd.concat([historical_df, current_df], ignore_index=True)
            prepared_teams = pd.concat([features_df.iloc[:2], current_players_df], ignore_index=True)
            builder.prepare_player_rows.return_value = prepared_rows
            builder.build_player_team_by_season.return_value = prepared_teams
            gold_df = prepared_rows.assign(team_stat=100.0)
            builder.build_gold_set.return_value = gold_df

            main()

            builder_class.assert_called_once_with(data_dir="test-data")
            builder.load_current_players.assert_called_once_with()
            builder.prepare_player_rows.assert_called_once_with(
                features_df, current_players_df, 2026, target_col="ppr_points_per_game",
            )
            builder.build_player_team_by_season.assert_called_once_with(features_df, current_players_df, 2026)
            builder.build_gold_set.assert_called_once()
            rows_df, _, _, teams_df = builder.build_gold_set.call_args.args
            assert rows_df is prepared_rows
            assert teams_df is prepared_teams
            builder.validate_data.assert_called_once()
            validated_df, year = builder.validate_data.call_args.args
            assert year == 2026
            builder.save_to_gold.assert_called_once()
            saved_df, filename = builder.save_to_gold.call_args.args
            assert saved_df is validated_df
            assert saved_df is gold_df
            assert filename == "ppr_points_per_game__2026.csv"
            method_names = [call[0] for call in builder.method_calls]
            assert method_names.index("validate_data") < method_names.index("save_to_gold")

    def test_main__does_not_save_either_table_when_validation_fails(self):
        with (
            patch("src.processing.gold.GoldSetBuilder") as builder_class,
            patch("sys.argv", ["gold"]),
        ):
            builder = builder_class.return_value
            builder.load_current_players.return_value = pd.DataFrame({
                "player_id": ["p1"], "season": [2026], "recent_team": ["KC"],
            })
            builder.load_player_stat_features.return_value = pd.DataFrame({
                "player_id": ["p1"], "season": [2025], "recent_team": ["KC"],
            })
            builder.prepare_player_rows.return_value = pd.DataFrame({"target": [1.0, np.nan]})
            builder.build_player_team_by_season.return_value = builder.load_current_players.return_value
            builder.validate_data.side_effect = ValueError("Invalid gold data")

            with pytest.raises(ValueError, match="Invalid gold data"):
                main()

            builder.prepare_player_rows.assert_called_once()
            builder.build_player_team_by_season.assert_called_once()
            builder.build_gold_set.assert_called_once()
            builder.save_to_gold.assert_not_called()


class TestLoadGoldSets:
    def test_gold_set_filename__uses_target_and_current_season(self):
        assert gold_set_filename("ppr_points_per_game", 2026) == "ppr_points_per_game__2026.csv"

    def test_load_gold_set__reads_the_target_named_file(self):
        gold_dir = tempfile.mkdtemp()
        try:
            expected = pd.DataFrame({"player_id": ["p1", "p2"], "target_season": [2021, 2022], "target": [1.5, 2.5]})
            expected.to_csv(os.path.join(gold_dir, "target_1__2026.csv"), index=False)

            pd.testing.assert_frame_equal(load_gold_set(gold_dir, "target_1", 2026), expected)
        finally:
            shutil.rmtree(gold_dir)

    def test_load_gold_set__reads_combined_file_with_blank_current_targets(self):
        gold_dir = tempfile.mkdtemp()
        try:
            expected = pd.DataFrame({"player_id": ["p1", "p1"], "target_season": [2025, 2026], "target": [10.0, np.nan]})
            expected.to_csv(os.path.join(gold_dir, gold_set_filename("target_1", 2026)), index=False)

            result = load_gold_set(gold_dir, "target_1", 2026)

            pd.testing.assert_frame_equal(result, expected)
            assert result["target"].isna().tolist() == [False, True]
        finally:
            shutil.rmtree(gold_dir)

    def test_loaders__read_each_targets_own_file(self):
        gold_dir = tempfile.mkdtemp()
        try:
            for target, value in [("target_1", 1), ("target_2", 2)]:
                for year in [2026, 2027]:
                    pd.DataFrame({"x": [value * year]}).to_csv(os.path.join(gold_dir, gold_set_filename(target, year)), index=False)

            assert load_gold_set(gold_dir, "target_2", 2026)["x"].tolist() == [4052]
            assert load_gold_set(gold_dir, "target_1", 2027)["x"].tolist() == [2027]
        finally:
            shutil.rmtree(gold_dir)

    def test_loaders__raise_when_the_file_is_missing(self):
        gold_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(FileNotFoundError):
                load_gold_set(gold_dir, "target_1", 2026)
        finally:
            shutil.rmtree(gold_dir)
