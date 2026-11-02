import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.modeling.data_builder import SplitYears


class TestSplitYears:
    def test_default_prediction_season__uses_current_roster_year_at_construction(self):
        with patch("src.modeling.data_builder.get_current_season", side_effect=[2026, 2027]) as current_season:
            first = SplitYears()
            second = SplitYears()
        assert first.prediction_season == 2026
        assert second.prediction_season == 2027
        assert current_season.call_count == 2
        current_season.assert_called_with(roster=True)

    def test_from_config__uses_explicit_snapshot_year_without_resolving_current_year(self):
        with patch("src.modeling.data_builder.get_current_season") as current_season:
            split_years = SplitYears.from_config({"test_data_years": 2}, prediction_season=2025)
        current_season.assert_not_called()
        assert split_years.prediction_season == 2025
        assert split_years.test_data_years == 2

    def test_from_config__applies_defaults_for_omitted_fields(self):
        assert SplitYears.from_config({}) == SplitYears(eval_data_years=1, test_data_years=1, num_training_seasons=None)

    def test_from_config__reads_every_field(self):
        split_years = SplitYears.from_config({"eval_data_years": 2, "test_data_years": 3, "num_training_seasons": 10})

        assert split_years == SplitYears(eval_data_years=2, test_data_years=3, num_training_seasons=10)


class TestSplitYearsToMasks:
    def test_to_masks__excludes_prediction_and_future_seasons_without_shifting_holdouts(self):
        seasons = pd.Series([2020, 2021, 2022, 2023, 2024, 2025, 2026], index=[8, 2, 5, 1, 9, 0, 3])
        masks = SplitYears(prediction_season=2025).to_masks(seasons)

        assert masks["train"].tolist() == [True, True, True, False, False, False, False]
        assert masks["eval"].tolist() == [False, False, False, True, False, False, False]
        assert masks["test"].tolist() == [False, False, False, False, True, False, False]
        for mask in masks.values():
            pd.testing.assert_index_equal(mask.index, seasons.index)

    def test_to_masks__counts_only_historical_seasons_when_validating_requested_window(self):
        seasons = pd.Series([2021, 2022, 2023, 2024, 2025])
        with pytest.raises(ValueError, match="exceeds the 3 distinct"):
            SplitYears(prediction_season=2024, num_training_seasons=2).to_masks(seasons)

    def test_to_masks__no_historical_seasons_produces_empty_scored_masks(self):
        seasons = pd.Series([2025, 2026, np.nan])
        masks = SplitYears(prediction_season=2025).to_masks(seasons)
        assert all(not mask.any() for mask in masks.values())

    def test_to_masks__partitions_seasons_chronologically(self):
        target_season = pd.Series([2019, 2020, 2021, 2022, 2023, 2024])

        masks = SplitYears(eval_data_years=1, test_data_years=2).to_masks(target_season)

        assert list(masks) == ["train", "eval", "test"]
        assert masks["test"].tolist() == [False, False, False, False, True, True]
        assert masks["eval"].tolist() == [False, False, False, True, False, False]
        assert masks["train"].tolist() == [True, True, True, False, False, False]

    def test_to_masks__preserves_index_and_boolean_dtype(self):
        target_season = pd.Series([2020, 2021, 2022], index=[8, 2, 5])

        masks = SplitYears().to_masks(target_season)

        for mask in masks.values():
            pd.testing.assert_index_equal(mask.index, target_season.index)
            assert mask.dtype == bool

    def test_to_masks__num_training_seasons_keeps_only_the_most_recent_training_seasons(self):
        target_season = pd.Series([2019, 2020, 2021, 2022, 2023, 2024])

        masks = SplitYears(eval_data_years=1, test_data_years=2, num_training_seasons=2).to_masks(target_season)

        assert masks["train"].tolist() == [False, True, True, False, False, False]

    def test_to_masks__missing_target_seasons_are_in_no_split(self):
        target_season = pd.Series([2020, 2021, 2022, np.nan])

        masks = SplitYears().to_masks(target_season)

        assert [mask[3] for mask in masks.values()] == [False, False, False]
        assert masks["test"].tolist() == [False, False, True, False]

    def test_to_masks__masks_are_disjoint(self):
        target_season = pd.Series([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2024])

        masks = SplitYears(eval_data_years=2, test_data_years=1).to_masks(target_season)

        assert sum(mask.astype(int) for mask in masks.values()).max() == 1

    def test_to_masks__raises_when_requested_seasons_exceed_available_seasons(self):
        target_season = pd.Series([2020, 2021, 2022, 2022])

        with pytest.raises(ValueError, match="exceeds the 3 distinct"):
            SplitYears(eval_data_years=1, test_data_years=1, num_training_seasons=2).to_masks(target_season)

    def test_to_masks__accepts_requested_seasons_equal_to_available_seasons(self):
        target_season = pd.Series([2020, 2021, 2022])

        masks = SplitYears(eval_data_years=1, test_data_years=1, num_training_seasons=1).to_masks(target_season)

        assert masks["train"].tolist() == [True, False, False]
