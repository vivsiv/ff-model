import os
import tempfile
import shutil

import pandas as pd

from src.data.nflverse import NflverseDataScraper


class TestNflverseDataScraper():
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.scraper = NflverseDataScraper(data_dir=cls.test_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_save_by_season__splits_into_one_file_per_season(self):
        df = pd.DataFrame({
            "season": [2022, 2022, 2023],
            "player": ["Player A", "Player B", "Player C"],
            "fantasy_points": [10.0, 20.0, 30.0],
        })

        self.scraper._save_by_season(df, "test_stats")

        path_2022 = os.path.join(self.scraper.bronze_dir, "2022_test_stats.csv")
        path_2023 = os.path.join(self.scraper.bronze_dir, "2023_test_stats.csv")
        assert os.path.exists(path_2022)
        assert os.path.exists(path_2023)

        result_2022 = pd.read_csv(path_2022)
        expected_2022 = pd.DataFrame({
            "season": [2022, 2022],
            "player": ["Player A", "Player B"],
            "fantasy_points": [10.0, 20.0],
        })
        pd.testing.assert_frame_equal(result_2022, expected_2022)

        result_2023 = pd.read_csv(path_2023)
        expected_2023 = pd.DataFrame({
            "season": [2023],
            "player": ["Player C"],
            "fantasy_points": [30.0],
        })
        pd.testing.assert_frame_equal(result_2023, expected_2023)
