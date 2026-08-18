"""
Integration tests that run the real data_quality checks against real, on-disk gold-layer
output (data/gold/{target}__training_set.csv / __prediction_set.csv).

Excluded from the default test run (see pyproject.toml's `addopts`/`markers`) since they:
  - depend on real generated data existing on disk (gitignored, not present in a fresh
    checkout/CI), and
  - are tied to a specific snapshot of nflverse's data (the known-value spot checks are only
    valid for the season(s) that were current when they were written).

Run explicitly with: pytest -m integration
Regenerate the gold data first if it's missing: python -m src.processing.gold --target-col <target>
"""
import os

import pandas as pd
import pytest

from src.processing.column_registry import get_identity_columns
from src.analysis.data_quality import run_training_data_quality_checks, run_prediction_data_quality_checks

pytestmark = pytest.mark.integration

DATA_DIR = "data"
GOLD_DIR = os.path.join(DATA_DIR, "gold")
TARGETS = ["fantasy_points_ppr", "ppr_points_per_game"]

NON_FEATURE_COLS = get_identity_columns("nflverse", "player_stats") + ["target_season", "target"]


def _load_gold_csv(filename: str) -> pd.DataFrame:
    """Loads gold_dir/filename, skipping (not failing) the test if it doesn't exist on disk."""
    path = os.path.join(GOLD_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"{path} not found -- run `python -m src.processing.gold --target-col ...` first")

    return pd.read_csv(path, low_memory=False)


class TestTrainingDataQuality:
    # ids= makes the actual gold_dir/ filename being read show up directly in the test ID
    # (e.g. `pytest -v` output), rather than just the bare target name.
    @pytest.mark.parametrize("target", TARGETS, ids=[f"{t}__training_set.csv" for t in TARGETS])
    def test_training_set_passes_quality_checks(self, target):
        filename = f"{target}__training_set.csv"
        training_data = _load_gold_csv(filename)
        feature_cols = [col for col in training_data.columns if col not in NON_FEATURE_COLS]

        run_training_data_quality_checks(training_data, feature_cols)


class TestPredictionDataQuality:
    @pytest.mark.parametrize("target", TARGETS, ids=[f"{t}__prediction_set.csv" for t in TARGETS])
    def test_prediction_set_passes_quality_checks(self, target):
        filename = f"{target}__prediction_set.csv"
        prediction_data = _load_gold_csv(filename)
        feature_cols = [col for col in prediction_data.columns if col not in NON_FEATURE_COLS]

        run_prediction_data_quality_checks(prediction_data, feature_cols)
