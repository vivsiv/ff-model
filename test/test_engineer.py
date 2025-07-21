import pytest
import pandas as pd

from src.feature_engineering import FantasyFeatureEngineer


class TestFantasyFeatureEngineer:

    @pytest.fixture
    def feature_eng(self):
        return FantasyFeatureEngineer()

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
    def target_relevance_df(self):
        return pd.DataFrame({
            'feature': ['f1', 'f2', 'f3', 'f4', 'f5', 'f6'],
            'p_corr': [0.80, 0.10, 0.95, 0.90, 0.95, 0.05],
            'mi': [0.80, 0.10, 0.95, 0.90, 0.95, 0.05],
        })

    def test_get_redundant_features(self, feature_eng, feature_corr_matrix):
        redundant_features = feature_eng.get_redundant_features(feature_corr_matrix, 0.5)

        expected_redundant_features = {'f1': {'f4', 'f5'}, 'f2': {'f3'}, 'f3': {'f2'}, 'f4': {'f1', 'f5'}, 'f5': {'f1', 'f4'}}

        assert redundant_features == expected_redundant_features

    def test_select_features_for_target(self, feature_eng, target_relevance_df, feature_corr_matrix):
        redundant_features = feature_eng.get_redundant_features(feature_corr_matrix, 0.5)

        selected_features = feature_eng.select_features_for_target(target_relevance_df, redundant_features)
        assert selected_features == {'f3', 'f5'}
