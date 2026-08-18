"""
Sanity checks for the gold-layer training/prediction sets built by
src/processing/gold.py's TrainingSetBuilder.
"""
from typing import List

import pandas as pd

# The earliest season present in nflverse's historical range as of writing -- it can never
# appear as a target_season, since there's no prior season to source features from. Hardcoded
# rather than derived from the data itself (which would make the check tautological): if this
# ever fails, either the pipeline has a real bug, or nflverse's historical range has grown
# further back and this constant is due for an update.
EARLIEST_SEASON = 1999


def run_training_data_quality_checks(training_data: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Runs sanity checks against a gold-layer training set (gold_dir/{target}__training_set.csv):
    shape, identity-column completeness, duplicates, degenerate (all/mostly-zero) rows,
    per-target_season row counts, the earliest-season-can't-be-a-target invariant, rookie
    exclusion, and a handful of known-value spot checks on specific player-seasons.

    Args:
        training_data: Gold-layer training set (one row per player-season, identity columns
            player_id/player_display_name/position/season/recent_team/target_season, a
            "target" column, feature_cols)
        feature_cols: Columns of training_data to treat as features for the degenerate-row
            checks

    Raises:
        AssertionError: If any check fails.
    """
    training_shape = training_data.shape
    assert training_shape[0] > 8000, f"Training data must have at least 8000 rows, got {training_shape[0]}"
    assert training_shape[1] >= 150, f"Training data must have at least 150 columns, got {training_shape[1]}"

    for col in ["player_id", "player_display_name", "season", "target_season"]:
        assert training_data[col].notna().all(), f"Training data must not have null {col} values"

    assert training_data["target"].notna().all(), "Training data must not have null target values"

    duplicates = training_data.duplicated()
    assert not duplicates.any(), f"Training data must not have duplicate rows, got {duplicates.sum()}"

    key_duplicates = training_data.duplicated(subset=["player_id", "target_season"])
    assert not key_duplicates.any(), (
        "Training data must have exactly one row per (player_id, target_season), got "
        f"{key_duplicates.sum()} duplicate(s)"
    )

    all_zero_rows = training_data[feature_cols].eq(0).all(axis=1)
    assert not all_zero_rows.any(), f"Training data must not have rows that are 0 for all features, got {all_zero_rows.sum()}"

    mostly_zero_rows = training_data[feature_cols].eq(0).sum(axis=1) / len(feature_cols) > 0.95
    assert not mostly_zero_rows.any(), f"Training data must not have rows that are 0 for 95% of features, got {mostly_zero_rows.sum()}"

    # Spot check some known rookies: their rookie season should never be a target_season -- a
    # rookie has no prior season to source features from, so _join_with_target's backward-only
    # join can never produce a row targeting their first season.
    rookie_seasons = {
        "Malik Nabers": 2024,
        "Ja'Marr Chase": 2021,
        "Saquon Barkley": 2018,
        "Dak Prescott": 2016,
        "Aaron Rodgers": 2005,
    }
    for player, rookie_season in rookie_seasons.items():
        rookie_target_rows = training_data[
            (training_data["player_display_name"] == player) & (training_data["target_season"] == rookie_season)
        ]
        assert len(rookie_target_rows) == 0, (
            f"Training data must not target a rookie season, got {player} {rookie_season}"
        )

    # Spot check some known real season totals -- these are each row's own season stats
    # (raw, not shifted/joined), so they're a real, independently-verifiable fact about the
    # underlying data, unrelated to which data source produced it.
    known_values = [
        ("Aaron Rodgers", 2011, "passing_tds", 45),
        ("Christian McCaffrey", 2019, "receptions", 116),
        ("Saquon Barkley", 2023, "rushing_yards", 962),
        ("Priest Holmes", 2001, "rushing_tds", 8),
        ("Terrell Owens", 2004, "receiving_yards", 1200),
        ("Terrell Owens", 2004, "games", 14),
    ]
    for player, season, col, expected in known_values:
        rows = training_data[(training_data["player_display_name"] == player) & (training_data["season"] == season)]
        assert len(rows) == 1, f"Expected exactly one {player} {season} row, got {len(rows)}"

        actual = rows[col].iloc[0]
        assert actual == expected, f"{player}'s {season} {col} should be {expected}, got {actual}"

    # The earliest season in the data can never be targeted (see EARLIEST_SEASON).
    earliest_season_targets = training_data[training_data["target_season"] == EARLIEST_SEASON]
    assert len(earliest_season_targets) == 0, (
        f"Training data must not target {EARLIEST_SEASON} (the earliest season in the data, "
        f"with no prior season to source features from), got {len(earliest_season_targets)} rows"
    )

    # Every target_season should have a reasonable number of rows (guards against a silent
    # partial join/filter bug for a specific season).
    season_counts = training_data["target_season"].value_counts()
    sparse_seasons = season_counts[season_counts < 350]
    assert len(sparse_seasons) == 0, (
        f"Every target_season must have at least 350 rows, got {sparse_seasons.to_dict()}"
    )


def run_prediction_data_quality_checks(prediction_data: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Runs sanity checks against a gold-layer prediction set
    (gold_dir/{target}__prediction_set.csv): shape, identity-column completeness, duplicates,
    degenerate (all/mostly-zero) rows, and a handful of known-value spot checks on specific
    players' most recent season.

    Args:
        prediction_data: Gold-layer prediction set (one row per player, identity columns as
            above, "target" expected to be entirely blank, feature_cols)
        feature_cols: Columns of prediction_data to treat as features for the degenerate-row checks

    Raises:
        AssertionError: If any check fails.
    """
    prediction_shape = prediction_data.shape
    assert prediction_shape[0] > 200, f"Prediction data must have at least 200 rows, got {prediction_shape[0]}"
    assert prediction_shape[1] >= 150, f"Prediction data must have at least 150 columns, got {prediction_shape[1]}"

    for col in ["player_id", "player_display_name", "season", "target_season"]:
        assert prediction_data[col].notna().all(), f"Prediction data must not have null {col} values"

    assert prediction_data["target"].isna().all(), "Prediction data's target must be entirely blank (nothing to predict against yet)"

    duplicates = prediction_data.duplicated()
    assert not duplicates.any(), f"Prediction data must not have duplicate rows, got {duplicates.sum()}"

    key_duplicates = prediction_data.duplicated(subset=["player_id", "target_season"])
    assert not key_duplicates.any(), (
        "Prediction data must have exactly one row per (player_id, target_season), got "
        f"{key_duplicates.sum()} duplicate(s)"
    )

    all_zero_rows = prediction_data[feature_cols].eq(0).all(axis=1)
    assert not all_zero_rows.any(), f"Prediction data must not have rows that are 0 for all features, got {all_zero_rows.sum()}"

    mostly_zero_rows = prediction_data[feature_cols].eq(0).sum(axis=1) / len(feature_cols) > 0.90
    assert not mostly_zero_rows.any(), f"Prediction data must not have rows that are 0 for 90% of features, got {mostly_zero_rows.sum()}"

    # Spot check some known players' most recent season on record (2025, as of writing).
    known_values = [
        ("Saquon Barkley", "rushing_yards", 1140),
        ("Ja'Marr Chase", "receiving_yards", 1412),
        ("Josh Allen", "passing_tds", 25),
        ("Josh Allen", "rushing_tds", 14),
        ("Christian McCaffrey", "games", 17),
    ]
    for player, col, expected in known_values:
        rows = prediction_data[prediction_data["player_display_name"] == player]
        assert len(rows) == 1, f"Expected exactly one {player} row, got {len(rows)}"

        actual = rows[col].iloc[0]
        assert actual == expected, f"{player}'s {col} should be {expected}, got {actual}"
