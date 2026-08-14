from src.processing.column_registry import (
    get_identity_columns,
    get_stat_columns,
    get_counting_stat_columns,
    get_targets,
)


class TestColumnRegistry():
    def test_nflverse_player_stats__identity_and_stats_have_no_overlap(self):
        identity = get_identity_columns("nflverse", "player_stats")
        stats = get_stat_columns("nflverse", "player_stats")

        assert set(identity) & set(stats) == set()

    def test_nflverse_player_stats__no_duplicates_within_any_list(self):
        identity = get_identity_columns("nflverse", "player_stats")
        stats = get_stat_columns("nflverse", "player_stats")
        targets = get_targets("nflverse", "player_stats")

        assert len(identity) == len(set(identity))
        assert len(stats) == len(set(stats))
        assert len(targets) == len(set(targets))

    def test_nflverse_player_stats__stats_has_known_fantasy_relevant_columns(self):
        stats = get_stat_columns("nflverse", "player_stats")

        for col in ["fantasy_points", "fantasy_points_ppr", "passing_epa", "target_share", "games"]:
            assert col in stats

    def test_nflverse_player_stats__identity_has_known_identity_columns(self):
        identity = get_identity_columns("nflverse", "player_stats")

        for col in ["player_id", "player_display_name", "position", "season", "recent_team"]:
            assert col in identity

    def test_nflverse_player_stats__targets_are_a_subset_of_stats(self):
        # targets aren't an alternative to being in "stats" -- a prior season's value of a
        # target is a legitimate feature for a later season (autoregression, not leakage), so
        # every target must also appear in "stats".
        stats = get_stat_columns("nflverse", "player_stats")
        targets = get_targets("nflverse", "player_stats")

        assert set(targets) <= set(stats)

    def test_nflverse_player_stats__targets_has_known_target_columns(self):
        targets = get_targets("nflverse", "player_stats")

        assert set(targets) == {"fantasy_points", "fantasy_points_ppr", "ppr_points_per_game"}

    def test_nflverse_player_stats__counting_stats_are_a_subset_of_stats(self):
        stats = get_stat_columns("nflverse", "player_stats")
        counting = get_counting_stat_columns("nflverse", "player_stats")

        assert set(counting) <= set(stats)

    def test_nflverse_player_stats__counting_and_rate_stats_are_mutually_exclusive_and_exhaustive(self):
        # Every stat should be classified as exactly one of "counting"/"rate" -- get_stat_columns
        # flattens both sub-lists, so their union must equal the full stats list with no overlap.
        import yaml
        from src.processing.column_registry import _REGISTRY_PATH

        with open(_REGISTRY_PATH) as f:
            raw_stats = yaml.safe_load(f)["nflverse"]["player_stats"]["stats"]

        counting = set(raw_stats["counting"])
        rate = set(raw_stats["rate"])

        assert counting & rate == set()
        assert counting | rate == set(get_stat_columns("nflverse", "player_stats"))

    def test_nflverse_player_stats__counting_stats_has_known_counting_columns(self):
        counting = get_counting_stat_columns("nflverse", "player_stats")

        for col in ["games", "receiving_yards", "passing_epa", "fantasy_points_ppr"]:
            assert col in counting

    def test_nflverse_player_stats__rate_stats_are_not_counting_stats(self):
        counting = get_counting_stat_columns("nflverse", "player_stats")

        for col in ["passing_cpoe", "pacr", "racr", "target_share", "air_yards_share", "wopr", "ppr_points_per_game"]:
            assert col not in counting

    def test_nflverse_team_stats__identity_and_stats_have_no_overlap(self):
        identity = get_identity_columns("nflverse", "team_stats")
        stats = get_stat_columns("nflverse", "team_stats")

        assert set(identity) & set(stats) == set()

    def test_nflverse_team_stats__no_duplicates_within_any_list(self):
        identity = get_identity_columns("nflverse", "team_stats")
        stats = get_stat_columns("nflverse", "team_stats")

        assert len(identity) == len(set(identity))
        assert len(stats) == len(set(stats))

    def test_nflverse_team_stats__stats_has_known_offense_relevant_columns(self):
        stats = get_stat_columns("nflverse", "team_stats")

        for col in ["passing_yards", "rushing_yards", "receiving_yards", "passing_epa"]:
            assert col in stats

    def test_nflverse_team_stats__identity_has_known_identity_columns(self):
        identity = get_identity_columns("nflverse", "team_stats")

        for col in ["team", "season"]:
            assert col in identity

    def test_nflverse_draft_picks__identity_and_stats_have_no_overlap(self):
        identity = get_identity_columns("nflverse", "draft_picks")
        stats = get_stat_columns("nflverse", "draft_picks")

        assert set(identity) & set(stats) == set()

    def test_nflverse_draft_picks__no_duplicates_within_any_list(self):
        identity = get_identity_columns("nflverse", "draft_picks")
        stats = get_stat_columns("nflverse", "draft_picks")

        assert len(identity) == len(set(identity))
        assert len(stats) == len(set(stats))

    def test_nflverse_draft_picks__identity_has_exactly_the_expected_columns(self):
        identity = get_identity_columns("nflverse", "draft_picks")

        assert set(identity) == {"season", "player_id"}

    def test_nflverse_draft_picks__stats_has_exactly_the_expected_columns(self):
        stats = get_stat_columns("nflverse", "draft_picks")

        assert set(stats) == {"draft_pick", "age_at_draft"}

    def test_nflverse_snap_counts__identity_and_stats_have_no_overlap(self):
        identity = get_identity_columns("nflverse", "snap_counts")
        stats = get_stat_columns("nflverse", "snap_counts")

        assert set(identity) & set(stats) == set()

    def test_nflverse_snap_counts__no_duplicates_within_any_list(self):
        identity = get_identity_columns("nflverse", "snap_counts")
        stats = get_stat_columns("nflverse", "snap_counts")

        assert len(identity) == len(set(identity))
        assert len(stats) == len(set(stats))

    def test_nflverse_snap_counts__identity_has_exactly_the_expected_columns(self):
        identity = get_identity_columns("nflverse", "snap_counts")

        assert set(identity) == {"player_id", "season"}

    def test_nflverse_snap_counts__stats_has_exactly_the_expected_columns(self):
        stats = get_stat_columns("nflverse", "snap_counts")

        assert set(stats) == {"offense_snaps", "offense_pct"}

    def test_nflverse_snap_counts__counting_stats_has_exactly_the_expected_columns(self):
        counting = get_counting_stat_columns("nflverse", "snap_counts")

        assert set(counting) == {"offense_snaps"}
