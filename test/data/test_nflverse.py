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

    def test_save__writes_a_single_file_to_bronze(self):
        df = pd.DataFrame({
            "season": [2022, 2022, 2023],
            "player": ["Player A", "Player B", "Player C"],
            "fantasy_points": [10.0, 20.0, 30.0],
        })

        self.scraper._save(df, "test_stats.csv")

        path = os.path.join(self.scraper.bronze_dir, "test_stats.csv")
        assert os.path.exists(path)

        result = pd.read_csv(path)
        pd.testing.assert_frame_equal(result, df)
