import os
import logging
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("gold_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingSetBuilder:
    """Builds gold layer training/live data from nflverse silver layer data."""

    def __init__(self, data_dir: str = "../data/nflv"):
        """
        Initialize the builder.

        Args:
            data_dir: Root directory the nflverse silver data lives in and gold data will be saved to.
        """
        self.silver_dir = os.path.join(data_dir, "silver")
        if not os.path.exists(self.silver_dir):
            raise FileNotFoundError(f"{self.silver_dir} not found")

        self.gold_dir = os.path.join(data_dir, "gold")
        os.makedirs(self.gold_dir, exist_ok=True)

    def _positional_baseline(
        self,
        df: pd.DataFrame,
        stat_columns: List[str],
        window_years: int = 5,
    ) -> pd.DataFrame:
        """
        Computes, for each (position, season), the trailing `window_years`-season league-wide
        average of each stat, using seasons up to and including that season. Used as the
        shrinkage target for the career-average features, instead of an all-time positional
        average, since league-wide offensive output has drifted over the nflverse history
        (1999-present) and an all-time average would be a stale reference for recent seasons.

        Args:
            df: Player-season stats, must contain "position" and "season" columns
            stat_columns: The stat columns to compute a positional baseline for
            window_years: Trailing window size in seasons (default: 5)

        Returns:
            DataFrame with one row per (position, season) and one
            "{stat}_positional_baseline" column per stat in stat_columns
        """
        season_position_means = (
            df.groupby(["position", "season"])[stat_columns]
            .mean()
            .reset_index()
            .sort_values(["position", "season"])
        )

        baseline_columns = ["position", "season"]
        for stat in stat_columns:
            baseline_col = f"{stat}_positional_baseline"
            season_position_means[baseline_col] = (
                season_position_means
                .groupby("position")[stat]
                .transform(lambda x: x.rolling(window=window_years, min_periods=1).mean())
            )
            baseline_columns.append(baseline_col)

        return season_position_means[baseline_columns]
