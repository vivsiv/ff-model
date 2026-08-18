import numpy as np
import pandas as pd
import pytest

from src.analysis.data_quality import (
    run_training_data_quality_checks,
    run_prediction_data_quality_checks,
    EARLIEST_SEASON,
)

# Stat columns referenced by name in the known-value spot checks.
SPOT_CHECK_STAT_COLUMNS = ["passing_tds", "receptions", "rushing_yards", "rushing_tds", "receiving_yards", "games"]
NUM_FILLER_COLUMNS = 149 - len(SPOT_CHECK_STAT_COLUMNS)
FEATURE_COLS = SPOT_CHECK_STAT_COLUMNS + [f"filler_stat_{i}" for i in range(NUM_FILLER_COLUMNS)]
IDENTITY_COLS = ["player_id", "player_display_name", "position", "season", "recent_team"]

# player_display_name/season -> target_season/overrides, keyed so each spot check lands in a
# distinct target_season group (>= 350 rows required per target_season).
TRAINING_SPOT_CHECK_ROWS = {
    ("Aaron Rodgers", 2011): {"target_season": 2012, "passing_tds": 45.0},
    ("Christian McCaffrey", 2019): {"target_season": 2020, "receptions": 116.0},
    ("Saquon Barkley", 2023): {"target_season": 2024, "rushing_yards": 962.0},
    ("Priest Holmes", 2001): {"target_season": 2002, "rushing_tds": 8.0},
    ("Terrell Owens", 2004): {"target_season": 2005, "receiving_yards": 1200.0, "games": 14.0},
}
ROWS_PER_TARGET_SEASON = 1700  # >= 350/target_season required, comfortably clears >8000 total rows


def _filler_row(rng: np.random.Generator) -> dict:
    # Non-zero-ish values across every feature column so no row trips the all/mostly-zero checks.
    return {col: float(rng.integers(1, 500)) for col in FEATURE_COLS}


def _base_row(rng: np.random.Generator, player_id: str, player_display_name: str, season: int, target_season: int) -> dict:
    return {
        "player_id": player_id,
        "player_display_name": player_display_name,
        "position": "WR",
        "season": season,
        "recent_team": "KC",
        "target_season": target_season,
        "target": float(rng.integers(1, 300)),
        **_filler_row(rng),
    }


def _valid_training_df(rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows = []

    for (player, season), overrides in TRAINING_SPOT_CHECK_ROWS.items():
        target_season = overrides["target_season"]

        row = {**_base_row(rng, f"pid_{player}", player, season, target_season), **overrides}
        rows.append(row)

        for i in range(ROWS_PER_TARGET_SEASON - 1):
            rows.append(_base_row(rng, f"pid_filler_{target_season}_{i}", f"Filler Player {i}", season, target_season))

    return pd.DataFrame(rows)[IDENTITY_COLS + ["target_season", "target"] + FEATURE_COLS]


def _valid_prediction_df(rng_seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)

    prediction_spot_check_rows = {
        "Saquon Barkley": {"rushing_yards": 1140.0},
        "Ja'Marr Chase": {"receiving_yards": 1412.0},
        "Josh Allen": {"passing_tds": 25.0, "rushing_tds": 14.0},
        "Christian McCaffrey": {"games": 17.0},
    }

    rows = []
    for player, overrides in prediction_spot_check_rows.items():
        row = {**_base_row(rng, f"pid_{player}", player, 2025, 2026), **overrides}
        row["target"] = np.nan
        rows.append(row)

    for i in range(250):
        row = _base_row(rng, f"pid_filler_{i}", f"Filler Player {i}", 2025, 2026)
        row["target"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)[IDENTITY_COLS + ["target_season", "target"] + FEATURE_COLS]


class TestRunTrainingDataQualityChecks:
    def test_passes_for_a_fully_valid_training_set(self):
        run_training_data_quality_checks(_valid_training_df(), FEATURE_COLS)

    def test_raises_when_too_few_rows(self):
        df = _valid_training_df().iloc[:100].copy()
        with pytest.raises(AssertionError, match="at least 8000 rows"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_too_few_columns(self):
        df = _valid_training_df().drop(columns=FEATURE_COLS[-10:])
        with pytest.raises(AssertionError, match="at least 150 columns"):
            run_training_data_quality_checks(df, FEATURE_COLS[:-10])

    def test_raises_when_player_id_is_null(self):
        df = _valid_training_df()
        df.loc[df.index[0], "player_id"] = None
        with pytest.raises(AssertionError, match="null player_id values"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_target_is_null(self):
        df = _valid_training_df()
        df.loc[df.index[0], "target"] = None
        with pytest.raises(AssertionError, match="null target values"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_duplicate_rows_present(self):
        df = _valid_training_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(AssertionError, match="must not have duplicate rows"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_same_player_and_target_season_appears_twice_with_different_values(self):
        # A more targeted duplicate than a literal full-row copy -- two different feature rows
        # both claiming to be the (player_id, target_season) key.
        df = _valid_training_df()
        second_row = df.iloc[[1]].copy()
        second_row["player_id"] = df["player_id"].iloc[0]
        second_row["target_season"] = df["target_season"].iloc[0]
        df = pd.concat([df, second_row], ignore_index=True)

        with pytest.raises(AssertionError, match=r"exactly one row per \(player_id, target_season\)"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_row_is_all_zero_across_features(self):
        df = _valid_training_df()
        df.loc[df.index[0], FEATURE_COLS] = 0.0
        with pytest.raises(AssertionError, match="0 for all features"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_rookie_season_is_targeted(self):
        df = _valid_training_df()
        df.loc[df.index[0], "player_display_name"] = "Malik Nabers"
        df.loc[df.index[0], "target_season"] = 2024
        with pytest.raises(AssertionError, match="must not target a rookie season"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_known_spot_check_value_is_wrong(self):
        df = _valid_training_df()
        df.loc[
            (df["player_display_name"] == "Aaron Rodgers") & (df["season"] == 2011), "passing_tds"
        ] = 0.0
        with pytest.raises(AssertionError, match="passing_tds should be 45"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_earliest_season_is_targeted(self):
        # Needs >= 350 rows targeting EARLIEST_SEASON too, so the sparse-target_season check
        # (which runs first) doesn't mask the check this test is actually targeting.
        rng = np.random.default_rng(99)
        df = _valid_training_df()
        earliest_season_rows = pd.DataFrame([
            _base_row(rng, f"pid_old_{i}", f"Old Player {i}", EARLIEST_SEASON - 1, EARLIEST_SEASON)
            for i in range(ROWS_PER_TARGET_SEASON)
        ])[IDENTITY_COLS + ["target_season", "target"] + FEATURE_COLS]
        df = pd.concat([df, earliest_season_rows], ignore_index=True)

        with pytest.raises(AssertionError, match="must not target"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_target_season_is_sparse(self):
        # Adds a brand new, deliberately-sparse target_season group (well under 350 rows)
        # rather than thinning an existing one, so the overall row-count check (which runs
        # first) doesn't mask the check this test is actually targeting.
        rng = np.random.default_rng(7)
        df = _valid_training_df()
        sparse_rows = pd.DataFrame([
            _base_row(rng, f"pid_sparse_{i}", f"Sparse Player {i}", 2009, 2010)
            for i in range(10)
        ])[IDENTITY_COLS + ["target_season", "target"] + FEATURE_COLS]
        df = pd.concat([df, sparse_rows], ignore_index=True)

        with pytest.raises(AssertionError, match="at least 350 rows"):
            run_training_data_quality_checks(df, FEATURE_COLS)


class TestRunPredictionDataQualityChecks:
    def test_passes_for_a_fully_valid_prediction_set(self):
        run_prediction_data_quality_checks(_valid_prediction_df(), FEATURE_COLS)

    def test_raises_when_too_few_rows(self):
        df = _valid_prediction_df().iloc[:5].copy()
        with pytest.raises(AssertionError, match="at least 200 rows"):
            run_prediction_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_target_is_not_entirely_blank(self):
        df = _valid_prediction_df()
        df.loc[df.index[0], "target"] = 10.0
        with pytest.raises(AssertionError, match="entirely blank"):
            run_prediction_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_duplicate_rows_present(self):
        df = _valid_prediction_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(AssertionError, match="must not have duplicate rows"):
            run_prediction_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_known_spot_check_value_is_wrong(self):
        df = _valid_prediction_df()
        df.loc[df["player_display_name"] == "Josh Allen", "passing_tds"] = 0.0
        with pytest.raises(AssertionError, match="passing_tds should be 25"):
            run_prediction_data_quality_checks(df, FEATURE_COLS)
