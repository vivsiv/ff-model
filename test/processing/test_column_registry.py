from src.processing.column_registry import get_included_columns, get_excluded_columns, get_labels


class TestColumnRegistry():
    def test_player_stats__included_and_excluded_have_no_overlap(self):
        included = get_included_columns("player_stats")
        excluded = get_excluded_columns("player_stats")

        assert set(included) & set(excluded) == set()

    def test_player_stats__no_duplicates_within_any_list(self):
        included = get_included_columns("player_stats")
        excluded = get_excluded_columns("player_stats")
        labels = get_labels("player_stats")

        assert len(included) == len(set(included))
        assert len(excluded) == len(set(excluded))
        assert len(labels) == len(set(labels))

    def test_player_stats__included_has_known_fantasy_relevant_columns(self):
        included = get_included_columns("player_stats")

        for col in ["fantasy_points", "fantasy_points_ppr", "passing_epa", "target_share"]:
            assert col in included

    def test_player_stats__excluded_has_known_identity_and_irrelevant_columns(self):
        excluded = get_excluded_columns("player_stats")

        for col in ["player_id", "position", "season", "def_sacks", "fg_made"]:
            assert col in excluded

    def test_player_stats__labels_are_a_subset_of_included(self):
        # labels aren't an alternative to being in "included" -- a prior season's value of a
        # label is a legitimate feature for a later season (autoregression, not leakage), so
        # every label must also appear in "included".
        included = get_included_columns("player_stats")
        labels = get_labels("player_stats")

        assert set(labels) <= set(included)

    def test_player_stats__labels_has_known_target_columns(self):
        labels = get_labels("player_stats")

        assert set(labels) == {"fantasy_points", "fantasy_points_ppr"}
