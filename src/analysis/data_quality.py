"""
Sanity checks for the gold-layer training/live sets.

NOTE: These checks were written against the old pro-football-reference-based gold schema
(id = "{name}_{year}" strings, column names like "pass_touchdowns"/"rec_receptions"/
"rush_yards") and predate the nflverse-based `TrainingSetBuilder` in `src/processing/gold.py`
(player_id-based rows, current nflverse column names, `{target}__training_set.csv` output).
They will not pass as-is against current gold output -- the known-value spot checks (specific
players/seasons/stat values) need to be re-derived against the current schema before these are
usable again.
"""
from typing import List

import pandas as pd


def run_training_data_quality_checks(training_data: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Runs sanity checks against a gold-layer training set: shape, dtypes, duplicates,
    nulls, degenerate (all/mostly-zero) rows, rookie exclusion, and a handful of
    known-value spot checks on specific player-seasons.

    Args:
        training_data: Gold-layer training set (one row per player-season, "id" identity
            column, feature_cols + target columns)
        feature_cols: Columns of training_data to treat as features for the degenerate-row
            checks

    Raises:
        AssertionError: If any check fails.
    """
    training_shape = training_data.shape
    assert training_shape[0] > 8000, f"Training data must have at least 8000 rows, got {training_shape[0]}"
    assert training_shape[1] >= 150, f"Training data must have at least 150 columns, got {training_shape[1]}"

    non_float_columns = training_data.select_dtypes(exclude=['float64']).columns
    assert len(non_float_columns) == 1, f"Id should be the only non float column, got {non_float_columns}"
    assert training_data['id'].dtype == 'object', "Id should be a string"

    # Check for rows where the id is only: _YYYY (the name is missing)
    year_only_rows = training_data[training_data['id'].str.startswith('_')]
    assert len(year_only_rows) == 0, f"Training data must not have rows where the id is missing the name, got {len(year_only_rows)}"

    duplicates = training_data.duplicated()
    assert not duplicates.any(), f"Training data must not have duplicates, got {duplicates.sum()}"

    null_value_rows = training_data.isnull().any(axis=1)
    assert not null_value_rows.any(), f"Training data must not have rows with null values, got {null_value_rows.sum()}"

    all_zero_rows = training_data[feature_cols].eq(0).all(axis=1)
    assert not all_zero_rows.any(), f"Training data must not have rows that are 0 for all features, got {all_zero_rows.sum()}"

    mostly_zero_rows = training_data[feature_cols].eq(0).sum(axis=1) / len(feature_cols) > 0.95
    assert not mostly_zero_rows.any(), f"Training data must not have rows that are 0 for 95% of features, got {mostly_zero_rows.sum()}"

    # Spot check some known rookies to see that they are not there
    rookies = ['malik_nabers_2024', 'jamarr_chase_2021', 'saquon_barkley_2018', 'dak_prescott_2016', 'aaron_rodgers_2005']
    for rookie in rookies:
        rookie_rows = training_data[training_data['id'] == rookie]
        assert len(rookie_rows) == 0, f"Training data must not have rookies, got {rookie}"

    # Spot check some known joins to see that they are correct
    aaron_rodgers_2012_pass_touchdowns = training_data[training_data['id'] == 'aaron_rodgers_2012']['pass_touchdowns'].iloc[0]
    assert aaron_rodgers_2012_pass_touchdowns == 45, f"Aaron Rodgers' 2012 row should have 2011's passing touchdowns (45), got {aaron_rodgers_2012_pass_touchdowns}"

    christian_mccaffrey_2020_rec_receptions = training_data[training_data['id'] == 'christian_mccaffrey_2020']['rec_receptions'].iloc[0]
    assert christian_mccaffrey_2020_rec_receptions == 116, f"Christian McCaffrey's 2020 row should have 2019's receptions (116), got {christian_mccaffrey_2020_rec_receptions}"

    saquon_barkley_2024_rushing_yards = training_data[training_data['id'] == 'saquon_barkley_2024']['rush_yards'].iloc[0]
    assert saquon_barkley_2024_rushing_yards == 962, f"Saquon Barkley's 2024 row should have 2023's rushing yards (962), got {saquon_barkley_2024_rushing_yards}"

    priest_holmes_2002_rush_touchdowns = training_data[training_data['id'] == 'priest_holmes_2002']['rush_touchdowns'].iloc[0]
    assert priest_holmes_2002_rush_touchdowns == 8, f"Priest Holmes' 2002 row should have 2001's rushing touchdowns (8), got {priest_holmes_2002_rush_touchdowns}"

    terrell_owens_2005_rec_yards = training_data[training_data['id'] == 'terrell_owens_2005']['rec_yards'].iloc[0]
    assert terrell_owens_2005_rec_yards == 1200, f"Terrell Owens' 2005 row should have 2004's rec yards (1200), got {terrell_owens_2005_rec_yards}"

    terrell_owens_2005_games = training_data[training_data['id'] == 'terrell_owens_2005']['games'].iloc[0]
    assert terrell_owens_2005_games == 14, f"Terrell Owens' 2005 row should have 2004's games (14), got {terrell_owens_2005_games}"

    years = training_data['id'].str.split('_').str[-1].astype(int)

    # Should have at least 350 rows for each year
    year_counts = years.value_counts()
    year_counts_below_350 = year_counts[year_counts < 350]
    assert len(year_counts_below_350) == 0, f"Training data must have at least 350 rows for each year, got {year_counts_below_350.index.tolist()}"

    # Check that the last year of data is dropped
    last_year_data = training_data[years == 2000]
    assert len(last_year_data) == 0, f"Training data must not have the last year of data, got {len(last_year_data)}"


def run_live_data_quality_checks(live_data: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Runs sanity checks against a gold-layer live (prediction) set: shape, dtypes,
    duplicates, nulls, degenerate (all/mostly-zero) rows, and a handful of known-value spot
    checks on specific players.

    Args:
        live_data: Gold-layer live/prediction set (one row per player, "id" identity
            column, feature_cols)
        feature_cols: Columns of live_data to treat as features for the degenerate-row
            checks

    Raises:
        AssertionError: If any check fails.
    """
    live_shape = live_data.shape
    assert live_shape[0] > 200, f"Live data must have at least 200 rows, got {live_shape[0]}"
    assert live_shape[1] >= 150, f"Live data must have at least 150 columns, got {live_shape[1]}"

    non_float_columns = live_data.select_dtypes(exclude=['float64']).columns
    assert len(non_float_columns) == 1, f"Id should be the only non float column, got {non_float_columns}"

    duplicates = live_data.duplicated()
    assert not duplicates.any(), f"Live data must not have duplicates, got {duplicates.sum()}"

    null_value_rows = live_data.isnull().any(axis=1)
    assert not null_value_rows.any(), f"Live data must not have rows with null values, got {null_value_rows.sum()}"

    all_zero_rows = live_data[feature_cols].eq(0).all(axis=1)
    assert not all_zero_rows.any(), f"Live data must not have rows that are 0 for all features, got {all_zero_rows.sum()}"

    mostly_zero_rows = live_data[feature_cols].eq(0).sum(axis=1) / len(feature_cols) > 0.90
    assert not mostly_zero_rows.any(), f"Live data must not have rows that are 0 for 95% of features, got {mostly_zero_rows.sum()}"

    # Spot check some known rows
    saquon_barkley_rushing_yards = live_data[live_data['id'] == 'saquon_barkley_rb']['rush_yards'].iloc[0]
    assert saquon_barkley_rushing_yards == 2005, f"Saquon Barkley's rushing yards should be 2005, got {saquon_barkley_rushing_yards}"

    jamarr_chase_rec_yards = live_data[live_data['id'] == 'jamarr_chase_wr']['rec_yards'].iloc[0]
    assert jamarr_chase_rec_yards == 1708, f"Jamarr Chase's rec yards should be 1708, got {jamarr_chase_rec_yards}"

    josh_allen_pass_touchdowns = live_data[live_data['id'] == 'josh_allen_qb']['pass_touchdowns'].iloc[0]
    assert josh_allen_pass_touchdowns == 28, f"Josh Allen's pass touchdowns should be 28, got {josh_allen_pass_touchdowns}"

    josh_allen_rush_touchdowns = live_data[live_data['id'] == 'josh_allen_qb']['rush_touchdowns'].iloc[0]
    assert josh_allen_rush_touchdowns == 12, f"Josh Allen's rush touchdowns should be 12, got {josh_allen_rush_touchdowns}"

    christian_mccaffrey_games = live_data[live_data['id'] == 'christian_mccaffrey_rb']['games'].iloc[0]
    assert christian_mccaffrey_games == 4, f"Christian McCaffrey's games should be 4, got {christian_mccaffrey_games}"
