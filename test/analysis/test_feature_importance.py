from unittest.mock import patch

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.analysis.feature_importance import (
    pearsons_correlation_between_features,
    pearsons_correlation_with_target,
    mutual_information_with_target,
    get_feature_importance,
    plot_feature_importance,
)


class TestPearsonsCorrelationBetweenFeatures:
    def test_returns_a_square_matrix_rounded_to_two_decimals(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [4.0, 3.0, 2.0, 1.0],
            "target": [10.0, 20.0, 30.0, 40.0],
        })

        result = pearsons_correlation_between_features(df, feature_cols=["f1", "f2"])

        assert list(result.columns) == ["f1", "f2"]
        assert list(result.index) == ["f1", "f2"]
        assert result.loc["f1", "f2"] == -1.0


class TestPearsonsCorrelationWithTarget:
    def test_returns_one_row_per_feature_sorted_descending(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],  # perfectly correlated with target
            "f2": [4.0, 3.0, 2.0, 1.0],  # perfectly anti-correlated with target
            "target": [10.0, 20.0, 30.0, 40.0],
        })

        result = pearsons_correlation_with_target(df, feature_cols=["f1", "f2"], target="target")

        assert list(result.index) == ["f1", "f2"]
        assert result.loc["f1", "p_corr"] == 1.0
        assert result.loc["f2", "p_corr"] == -1.0

    def test_target_column_is_not_included_as_a_feature_row(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "target": [1.0, 2.0, 3.0],
        })

        result = pearsons_correlation_with_target(df, feature_cols=["f1"], target="target")

        assert "target" not in result.index


class TestMutualInformationWithTarget:
    def test_returns_one_row_per_feature_sorted_descending(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [5.0, 5.0, 5.0],
            "target": [1.0, 2.0, 3.0],
        })

        with patch("src.analysis.feature_importance.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.9, 0.1]

            result = mutual_information_with_target(df, feature_cols=["f1", "f2"], target="target")

        expected = pd.DataFrame({
            "feature": ["f1", "f2"],
            "mi": [0.9, 0.1],
        }).set_index("feature")
        pd.testing.assert_frame_equal(result, expected)

    def test_passes_random_state_through_for_reproducibility(self):
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "target": [1.0, 2.0, 3.0]})

        with patch("src.analysis.feature_importance.mutual_info_regression") as mock_mi:
            mock_mi.return_value = [0.5]
            mutual_information_with_target(df, feature_cols=["f1"], target="target", random_state=42)

            _, kwargs = mock_mi.call_args
            assert kwargs["random_state"] == 42


class TestGetFeatureImportance:
    def test_pairs_feature_names_with_importances_sorted_descending(self):
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [4.0, 3.0, 2.0, 1.0]})
        y = pd.Series([1.0, 2.0, 3.0, 4.0])

        pipeline = Pipeline([
            ("imputer", SimpleImputer()),
            ("model", RandomForestRegressor(n_estimators=5, random_state=0)),
        ])
        pipeline.fit(X, y)

        result = get_feature_importance(pipeline)

        assert set(result.columns) == {"feature", "importance"}
        assert set(result["feature"]) == {"f1", "f2"}
        assert list(result["importance"]) == sorted(result["importance"], reverse=True)


class TestPlotFeatureImportance:
    def test_plots_only_the_requested_number_of_top_features(self):
        feature_importance_df = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "importance": [0.5, 0.3, 0.2],
        })

        ax = plot_feature_importance(feature_importance_df, num_features=2)

        assert len(ax.patches) == 2
