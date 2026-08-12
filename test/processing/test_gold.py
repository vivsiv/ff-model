import os
import tempfile
import shutil
from unittest.mock import patch

import numpy as np
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

    def test_load_player_features__merges_snap_counts_and_extends_career_features_to_them(self):
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

            builder = TrainingSetBuilder(data_dir=test_dir)
            with patch("src.processing.gold.get_identity_columns", side_effect=lambda s, t: identity_columns[(s, t)]), \
                 patch("src.processing.gold.get_stat_columns", side_effect=lambda s, t: stat_columns[(s, t)]):
                result = builder.load_player_features()

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

    def test_build_training_set__rejects_a_target_not_in_the_registry(self):
        with pytest.raises(AssertionError):
            self.builder.build_training_set(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), target_col="not_a_real_target"
            )

    def test_build_prediction_rows__keeps_only_the_most_recent_season_per_player(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p1"],
            "season": [2020, 2021, 2022],
            "fantasy_points_ppr_career_avg": [10.0, 15.0, 20.0],
        })

        result = self.builder._build_prediction_rows(features_df, prediction_season=2023)

        assert len(result) == 1
        assert result["season"].iloc[0] == 2022
        assert result["fantasy_points_ppr_career_avg"].iloc[0] == 20.0
        assert result["target_season"].iloc[0] == 2023
        assert pd.isna(result["target"].iloc[0])
        # 2022 -> predicting 2023 is a normal adjacent-year gap (no seasons missed)
        assert result["seasons_since_played"].iloc[0] == 0

    def test_build_prediction_rows__reflects_a_real_gap_since_last_played(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "season": [2019, 2020],
            "fantasy_points_ppr_career_avg": [10.0, 15.0],
        })

        result = self.builder._build_prediction_rows(features_df, prediction_season=2023)

        # last played 2020, predicting 2023 -> 2021 and 2022 were missed (2 seasons)
        assert result["season"].iloc[0] == 2020
        assert result["target_season"].iloc[0] == 2023
        assert result["seasons_since_played"].iloc[0] == 2

    def test_build_prediction_rows__does_not_mix_players(self):
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1", "p2", "p2", "p2"],
            "season": [2020, 2021, 2019, 2020, 2022],
            "fantasy_points_ppr_career_avg": [10.0, 20.0, 1.0, 2.0, 30.0],
        })

        result = self.builder._build_prediction_rows(
            features_df, prediction_season=2023
        ).sort_values("player_id").reset_index(drop=True)

        assert list(result["player_id"]) == ["p1", "p2"]
        assert list(result["season"]) == [2021, 2022]
        assert list(result["fantasy_points_ppr_career_avg"]) == [20.0, 30.0]
        assert list(result["seasons_since_played"]) == [1, 0]

    def test_build_prediction_set__rejects_a_target_not_in_the_registry(self):
        with pytest.raises(AssertionError):
            self.builder.build_prediction_set(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                target_col="not_a_real_target", prediction_season=2026
            )

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
        df = pd.DataFrame({
            "player_id": ["p1"],
            "season": [2020],
            "target_season": [2021],
            "recent_team": ["KC"],
        })
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "season": [2020, 2021],
            "recent_team": ["KC", "KC"],
        })
        team_features_df = pd.DataFrame({
            "team": ["KC", "SF"],
            "season": [2020, 2020],
            "passing_yards": [4000.0, 3500.0],
        })

        result = self.builder._join_team_features(df, features_df, team_features_df, ["passing_yards"])

        # league avg for 2020 = mean(4000, 3500) = 3750; destination (KC) = 4000
        assert "origin_team" not in result.columns
        assert "destination_team" not in result.columns
        assert result["team_passing_yards"].iloc[0] == 250.0
        assert result["team_shift_passing_yards"].iloc[0] == 0.0

    def test_join_team_features__team_change_looks_up_destination_team_from_origin_season(self):
        # Mirrors the Davante Adams motivation: player's own feature season (2022) was with
        # the Raiders, but they're actually on the Rams by target_season (2023). Destination
        # stats must come from the Rams' 2022 (origin season) performance, not 2023's.
        df = pd.DataFrame({
            "player_id": ["p1"],
            "season": [2022],
            "target_season": [2023],
            "recent_team": ["LV"],
        })
        features_df = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "season": [2022, 2023],
            "recent_team": ["LV", "LA"],
        })
        team_features_df = pd.DataFrame({
            "team": ["LV", "LA", "LA"],
            "season": [2022, 2022, 2023],
            "passing_yards": [3000.0, 4200.0, 9999.0],
        })

        result = self.builder._join_team_features(df, features_df, team_features_df, ["passing_yards"])

        # league avg for the *origin* season (2022) = mean(3000, 4200) = 3600, using only
        # 2022 rows -- LA's 9999 in 2023 must not leak into it. destination (LA, 2022) = 4200.
        assert "origin_team" not in result.columns
        assert "destination_team" not in result.columns
        assert result["team_passing_yards"].iloc[0] == 600.0
        assert result["team_shift_passing_yards"].iloc[0] == 1200.0

    def test_join_team_features__falls_back_to_origin_team_when_target_season_is_unmatched(self):
        # Prediction rows: target_season hasn't happened yet, so there's no features_df row
        # for it -- destination_team should fall back to the player's own current team
        # (assume no team change) rather than producing a NaN shift.
        df = pd.DataFrame({
            "player_id": ["p1"],
            "season": [2025],
            "target_season": [2026],
            "recent_team": ["KC"],
        })
        features_df = pd.DataFrame({
            "player_id": ["p1"],
            "season": [2025],
            "recent_team": ["KC"],
        })
        team_features_df = pd.DataFrame({
            "team": ["KC"],
            "season": [2025],
            "passing_yards": [4000.0],
        })

        result = self.builder._join_team_features(df, features_df, team_features_df, ["passing_yards"])

        assert "destination_team" not in result.columns
        assert result["team_shift_passing_yards"].iloc[0] == 0.0

    def test_join_team_features__does_not_mix_players(self):
        df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "season": [2020, 2020],
            "target_season": [2021, 2021],
            "recent_team": ["KC", "SF"],
        })
        features_df = pd.DataFrame({
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
            df, features_df, team_features_df, ["passing_yards"]
        ).sort_values("player_id").reset_index(drop=True)

        # league avg for 2020 = mean(4000, 3500, 3000) = 3500.
        # p1 didn't change teams -> shift 0; p2 (SF -> DAL) shouldn't pick up p1's KC lookup.
        assert list(result["team_passing_yards"]) == [500.0, -500.0]
        assert list(result["team_shift_passing_yards"]) == [0.0, -500.0]

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
