import pandas as pd

from src.modeling.data_builder import EVAL_SPLIT, TEST_SPLIT, TRAIN_SPLIT, ModelDataBuilder, SplitYears
from src.processing.gold import TARGET_COL, load_prediction_set, load_training_set


class TabularModelDataBuilder(ModelDataBuilder):
    """Build tabular datasets from gold tables, including features, targets, identities, and sample weights.

    See ModelDataBuilder for the shared config schema.
    """

    def _build_data_splits(self) -> dict[str, pd.DataFrame]:
        """Build train/eval/test datasets from the gold source table.

        Returns:
            X_train/X_eval/X_test, y_train/y_eval/y_test,
            identity_train/identity_eval/identity_test, and sample_weight_train.
            Always builds fresh data; use get_data_splits() for cached access.

        Raises:
            ValueError: If requested split seasons exceed the distinct seasons available.
        """
        self.source_data = load_training_set(self.gold_data_dir, self.target)

        data = self.source_data
        if self.positions:
            data = self._filter_positions(data, self.positions)

        feature_cols = self._resolve_feature_columns(data.columns)
        target_df = data[TARGET_COL]
        target_season = data["target_season"]

        return self.split(
            split_years=SplitYears.from_config(self.config.get("split", {})),
            target_season=target_season,
            features_df=data[feature_cols],
            target_df=target_df,
            identity_df=data[self.identity_cols],
            sample_weights=self._compute_sample_weights(target_df),
        )

    def build_prediction_data(self) -> dict[str, pd.DataFrame]:
        """Build live prediction data as row-aligned identity and features DataFrames."""
        data = load_prediction_set(self.gold_data_dir, self.target)

        if self.positions:
            data = self._filter_positions(data, self.positions)

        feature_cols = self._resolve_feature_columns(data.columns)

        return {
            "identity": data[self.identity_cols],
            "features": data[feature_cols],
        }

    @staticmethod
    def split(
        split_years: SplitYears,
        target_season: pd.Series,
        features_df: pd.DataFrame,
        target_df: pd.Series,
        identity_df: pd.DataFrame,
        sample_weights: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        """Slice aligned tabular inputs into chronological train/eval/test splits.

        See SplitYears.to_masks for season allocation. Sample weights are returned
        only for training rows.
        """
        masks = split_years.to_masks(target_season)
        is_train, is_eval, is_test = masks[TRAIN_SPLIT], masks[EVAL_SPLIT], masks[TEST_SPLIT]

        return {
            "X_train": features_df[is_train],
            "X_eval": features_df[is_eval],
            "X_test": features_df[is_test],
            "y_train": target_df[is_train],
            "y_eval": target_df[is_eval],
            "y_test": target_df[is_test],
            "identity_train": identity_df[is_train],
            "identity_eval": identity_df[is_eval],
            "identity_test": identity_df[is_test],
            "sample_weight_train": sample_weights[is_train],
        }
