import numpy as np
import pandas as pd
import pytest

from src.modeling.data_builder import SplitYears


class TestSplitYears:
    def test_from_config__applies_defaults_for_omitted_fields(self):
        assert SplitYears.from_config({}) == SplitYears(eval_data_years=1, test_data_years=1, num_training_seasons=None)

    def test_from_config__reads_every_field(self):
        split_years = SplitYears.from_config({"eval_data_years": 2, "test_data_years": 3, "num_training_seasons": 10})

        assert split_years == SplitYears(eval_data_years=2, test_data_years=3, num_training_seasons=10)


class TestSplitYearsToMasks:
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
