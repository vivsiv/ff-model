import os
import tempfile
import shutil
import pytest
import pandas as pd
from unittest.mock import patch

from src.feature_engineering import FantasyFeatureEngineer


class TestFantasyFeatureEngineer:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.gold_dir = os.path.join(cls.test_dir, "gold")

        os.makedirs(cls.gold_dir)

        gold_data = pd.DataFrame({
            'id': ['x', 'y', 'z'],
            'f1': [1, 2, 3],
            'f2': [100, 50, 0],
            'f3': [12, 0, 8],
            'target': [10, 11, 12]
        })
        gold_data.to_csv(os.path.join(cls.gold_dir, "final_stats.csv"), index=False)

        cls.feature_eng = FantasyFeatureEngineer(
            data_dir=cls.test_dir,
            metadata_cols=['id'],
            target_cols=['target'],
            redundancy_threshold=0.5)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    @pytest.fixture
    def feature_corr_matrix(self):
        corr_data = {
            'f1': {
                'f1': 1.0,
                'f2': 0.45,
                'f3': 0.50,
                'f4': 0.6,
                'f5': 0.65,
                'f6': 0,
            },
            'f2': {
                'f1': 0.45,
                'f2': 1.0,
                'f3': 0.51,
                'f4': 0.2,
                'f5': 0.1,
                'f6': 0,
            },
            'f3': {
                'f1': 0.50,
                'f2': 0.51,
                'f3': 1.0,
                'f4': 0.3,
                'f5': 0.1,
                'f6': 0,
            },
            'f4': {
                'f1': 0.6,
                'f2': 0.2,
                'f3': 0.3,
                'f4': 1.0,
                'f5': 0.8,
                'f6': 0,
            },
            'f5': {
                'f1': 0.65,
                'f2': 0.1,
                'f3': 0.1,
                'f4': 0.8,
                'f5': 1.0,
                'f6': 0,
            },
            'f6': {
                'f1': 0,
                'f2': 0,
                'f3': 0,
                'f4': 0,
                'f5': 0,
                'f6': 1.0,
            }
        }
        return pd.DataFrame(corr_data)

    @pytest.fixture
    def test_pearsons_correlation_with_target(self):
        corr_matrix = self.feature_eng.pearsons_correlation_with_target('target')
        assert corr_matrix.columns.tolist() == ['feature', 'p_corr']
        assert sorted(corr_matrix['feature'].tolist()) == ['f1', 'f2', 'f3']

    def test_mutual_information_with_target(self):
        with patch('src.feature_engineering.mutual_info_regression') as mock_mi:
            mock_mi.return_value = [0.8, 0.3, 0.9]

            mi_df = self.feature_eng.mutual_information_with_target('target').sort_values(by='feature')

            expected_df = pd.DataFrame({
                'feature': ['f1', 'f2', 'f3'],
                'mi': [0.8, 0.3, 0.9]
            }).sort_values(by='feature')

            pd.testing.assert_frame_equal(mi_df, expected_df)
