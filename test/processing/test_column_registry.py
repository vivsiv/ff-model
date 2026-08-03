from src.processing.column_registry import get_included_columns, get_excluded_columns, get_targets


class TestColumnRegistry():
    def test_player_stats__included_and_excluded_have_no_overlap(self):
        included = get_included_columns("player_stats")
        excluded = get_excluded_columns("player_stats")

        assert set(included) & set(excluded) == set()

    def test_player_stats__no_duplicates_within_any_list(self):
        included = get_included_columns("player_stats")
        excluded = get_excluded_columns("player_stats")
        targets = get_targets("player_stats")

        assert len(included) == len(set(included))
        assert len(excluded) == len(set(excluded))
        assert len(targets) == len(set(targets))

    def test_player_stats__included_has_known_fantasy_relevant_columns(self):
        included = get_included_columns("player_stats")

        for col in ["fantasy_points", "fantasy_points_ppr", "passing_epa", "target_share"]:
            assert col in included

    def test_player_stats__excluded_has_known_identity_and_irrelevant_columns(self):
        excluded = get_excluded_columns("player_stats")

        for col in ["player_id", "position", "season", "def_sacks", "fg_made"]:
            assert col in excluded

    def test_player_stats__targets_are_a_subset_of_included(self):
        # targets aren't an alternative to being in "included" -- a prior season's value of a
        # target is a legitimate feature for a later season (autoregression, not leakage), so
        # every target must also appear in "included".
        included = get_included_columns("player_stats")
        targets = get_targets("player_stats")

        assert set(targets) <= set(included)

    def test_player_stats__targets_has_known_target_columns(self):
        targets = get_targets("player_stats")

        assert set(targets) == {"fantasy_points", "fantasy_points_ppr"}
