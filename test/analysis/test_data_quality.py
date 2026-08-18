import numpy as np
import pandas as pd
import pytest

from src.analysis.data_quality import (
    run_training_data_quality_checks,
    run_live_data_quality_checks,
)

# Stat columns referenced by name in the known-value spot checks.
SPOT_CHECK_STAT_COLUMNS = [
    "pass_touchdowns", "rec_receptions", "rush_yards", "rush_touchdowns", "rec_yards", "games",
]
NUM_FILLER_COLUMNS = 149 - len(SPOT_CHECK_STAT_COLUMNS)
FEATURE_COLS = SPOT_CHECK_STAT_COLUMNS + [f"filler_stat_{i}" for i in range(NUM_FILLER_COLUMNS)]

# One real player-season/player id per required spot check, grouped by the year each row's
# id encodes.
TRAINING_SPOT_CHECK_ROWS = {
    "aaron_rodgers_2012": {"pass_touchdowns": 45.0},
    "christian_mccaffrey_2020": {"rec_receptions": 116.0},
    "saquon_barkley_2024": {"rush_yards": 962.0},
    "priest_holmes_2002": {"rush_touchdowns": 8.0},
    "terrell_owens_2005": {"rec_yards": 1200.0, "games": 14.0},
}
ROWS_PER_YEAR = 1700  # >= 350/year required, comfortably clears the >8000 total rows required


def _filler_row(rng: np.random.Generator) -> dict:
    # Non-zero-ish values across every feature column so no row trips the all/mostly-zero checks.
    row = {col: float(rng.integers(1, 500)) for col in FEATURE_COLS}
    return row


def _valid_training_df(rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows = []

    for player_id, overrides in TRAINING_SPOT_CHECK_ROWS.items():
        year = int(player_id.rsplit("_", 1)[1])

        row = {"id": player_id, **_filler_row(rng), **overrides}
        rows.append(row)

        for i in range(ROWS_PER_YEAR - 1):
            rows.append({"id": f"filler_player_{i}_{year}", **_filler_row(rng)})

    return pd.DataFrame(rows)[["id"] + FEATURE_COLS]


def _valid_live_df(rng_seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)

    spot_check_rows = {
        "saquon_barkley_rb": {"rush_yards": 2005.0},
        "jamarr_chase_wr": {"rec_yards": 1708.0},
        "josh_allen_qb": {"pass_touchdowns": 28.0, "rush_touchdowns": 12.0},
        "christian_mccaffrey_rb": {"games": 4.0},
    }

    rows = [{"id": player_id, **_filler_row(rng), **overrides} for player_id, overrides in spot_check_rows.items()]
    rows += [{"id": f"filler_player_{i}", **_filler_row(rng)} for i in range(250)]

    return pd.DataFrame(rows)[["id"] + FEATURE_COLS]


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

    def test_raises_when_id_is_missing_the_player_name(self):
        df = _valid_training_df()
        df.loc[df.index[0], "id"] = "_2012"
        with pytest.raises(AssertionError, match="id is missing the name"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_duplicate_rows_present(self):
        df = _valid_training_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(AssertionError, match="must not have duplicates"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_null_values_present(self):
        df = _valid_training_df()
        df.loc[df.index[0], FEATURE_COLS[0]] = None
        with pytest.raises(AssertionError, match="null values"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_row_is_all_zero_across_features(self):
        df = _valid_training_df()
        df.loc[df.index[0], FEATURE_COLS] = 0.0
        with pytest.raises(AssertionError, match="0 for all features"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_rookie_is_present(self):
        df = _valid_training_df()
        df.loc[df.index[0], "id"] = "malik_nabers_2024"
        with pytest.raises(AssertionError, match="must not have rookies"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_known_spot_check_value_is_wrong(self):
        df = _valid_training_df()
        df.loc[df["id"] == "aaron_rodgers_2012", "pass_touchdowns"] = 0.0
        with pytest.raises(AssertionError, match="passing touchdowns"):
            run_training_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_last_year_of_data_is_present(self):
        # Needs >= 350 rows in year 2000 too, so the year-count check (which runs first)
        # doesn't mask the last-year check this test is actually targeting.
        rng = np.random.default_rng(99)
        df = _valid_training_df()
        year_2000_rows = pd.DataFrame(
            [{"id": f"old_player_{i}_2000", **_filler_row(rng)} for i in range(ROWS_PER_YEAR)]
        )[["id"] + FEATURE_COLS]
        df = pd.concat([df, year_2000_rows], ignore_index=True)

        with pytest.raises(AssertionError, match="not have the last year"):
            run_training_data_quality_checks(df, FEATURE_COLS)


class TestRunLiveDataQualityChecks:
    def test_passes_for_a_fully_valid_live_set(self):
        run_live_data_quality_checks(_valid_live_df(), FEATURE_COLS)

    def test_raises_when_too_few_rows(self):
        df = _valid_live_df().iloc[:5].copy()
        with pytest.raises(AssertionError, match="at least 200 rows"):
            run_live_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_duplicate_rows_present(self):
        df = _valid_live_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(AssertionError, match="must not have duplicates"):
            run_live_data_quality_checks(df, FEATURE_COLS)

    def test_raises_when_a_known_spot_check_value_is_wrong(self):
        df = _valid_live_df()
        df.loc[df["id"] == "josh_allen_qb", "pass_touchdowns"] = 0.0
        with pytest.raises(AssertionError, match="pass touchdowns"):
            run_live_data_quality_checks(df, FEATURE_COLS)
