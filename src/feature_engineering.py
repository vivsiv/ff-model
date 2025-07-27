#!/usr/bin/env python3
"""
NFL Fantasy Football Feature Engineering

This module combines player and team statistics and creates advanced
features for fantasy football prediction.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
from typing import Dict, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FantasyFeatureEngineer:
    """Engineer features for fantasy football prediction."""

    def __init__(
        self,
        data_dir: str = "../data",
        metadata_cols: List[str] = [],
        target_cols: List[str] = [],
        must_include_features: List[str] = [],
        redundancy_threshold: float = 0.75,
    ):
        """
        Initialize the feature engineer.

        Args:
            data_dir: Directory containing the processed data
        """
        self.data_dir = data_dir
        try:
            # We assume the silver table already exists.
            self.silver_data_dir = os.path.join(data_dir, "silver")
            self.silver_data = self.load_silver_table()
        except Exception as e:
            logger.error(f"Silver table not found at {self.silver_data_dir}")
            raise e

        self.gold_data_dir = os.path.join(data_dir, "gold")
        os.makedirs(self.gold_data_dir, exist_ok=True)

        self.discovery_data_dir = os.path.join(data_dir, "discovery")
        os.makedirs(self.discovery_data_dir, exist_ok=True)

        self.metadata_cols = metadata_cols
        self.target_cols = target_cols
        self.feature_cols = [col for col in self.silver_data.columns if col not in self.metadata_cols + self.target_cols]
        self.must_include_features = must_include_features
        self.redundancy_threshold = redundancy_threshold

    def load_silver_table(self) -> pd.DataFrame:
        """
        Loads the silver table(s).

        Returns:
            DataFrame of the silver table(s)
        """
        data = pd.DataFrame()

        silver_path = os.path.join(self.silver_data_dir, "final_stats.csv")
        if os.path.exists(silver_path):
            data = pd.read_csv(silver_path)
            logger.info(f"Loaded silver table: {len(data)} rows")

        return data

    def pearsons_correlation_between_features(self, output_file_name: str = "feature_corr_matrix.csv") -> pd.DataFrame:
        """
        Runs a Pearson correlation analysis between all features.
        """
        corr_matrix = self.silver_data[self.feature_cols].corr(method='pearson')
        corr_matrix = corr_matrix.round(2)
        corr_matrix.to_csv(os.path.join(self.discovery_data_dir, output_file_name))
        return corr_matrix

    def pearsons_correlation_with_target(self, target: str, output_file_name: str = "target_corr.csv") -> pd.DataFrame:
        """
        Runs a Pearson correlation analysis between all features and a single target.
        """

        silver_data_single_target = self.silver_data[self.feature_cols].copy()
        silver_data_single_target[target] = self.silver_data[target]

        corr_matrix = silver_data_single_target.corr(method='pearson')

        # Double brackets returns a dataframe as opposed to a series
        # The .drop() call drops the row with index label exam_score (axis is set to 0 by default)
        corr_matrix = corr_matrix[[target]].drop(axis=0, labels=[target])
        corr_matrix = corr_matrix.reset_index().rename(columns={'index': 'feature', target: 'p_corr'})
        corr_matrix['p_corr'] = corr_matrix['p_corr'].round(2)
        corr_matrix = corr_matrix.sort_values(by='p_corr', ascending=False)

        corr_matrix.to_csv(os.path.join(self.discovery_data_dir, output_file_name), index=False)

        return corr_matrix

    def mutual_information_with_target(self, target: str, output_file_name: str = "target_mi.csv") -> pd.DataFrame:
        """
        Runs a mutual information analysis between all features and a single target.
        """
        mutual_info_values = mutual_info_regression(self.silver_data[self.feature_cols], self.silver_data[target], random_state=68)
        mutual_info = pd.DataFrame({
            'feature': self.feature_cols,
            'mi': [round(mi, 2) for mi in mutual_info_values]
        })
        mutual_info = mutual_info.sort_values(by="mi", ascending=False)

        mutual_info.to_csv(os.path.join(self.discovery_data_dir, output_file_name), index=False)
        return mutual_info

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
        output_file_name: str = "correlation_matrix.png",
        font_size: int = 12,
        annot_size: int = 8,
    ):
        """
        Plot correlation matrix with dynamically sized figure based on number of features.

        Args:
            corr_matrix: Correlation matrix DataFrame
            title: Plot title
            output_file_name: Output file name
            font_size: Base font size for labels
            annot_size: Font size for correlation values
        """
        n_features = len(corr_matrix)

        # Calculate figure size based on number of features - much larger now
        base_size = 0.6  # Increased from 0.3 to 0.6 inches per feature
        min_size = 12    # Increased from 8 to 12
        max_size = 60    # Increased from 40 to 60

        fig_size = min(max(n_features * base_size, min_size), max_size)

        plt.figure(figsize=(fig_size, fig_size))

        # Use larger fonts for better readability
        label_font_size = max(8, font_size - (n_features // 30))
        value_font_size = max(6, annot_size - (n_features // 40))

        sns.heatmap(corr_matrix,
                   annot=True,
                   cmap="coolwarm",
                   center=0,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   annot_kws={'size': value_font_size})

        plt.title(title, fontsize=label_font_size + 4)
        plt.xticks(rotation=45, ha='right', fontsize=label_font_size)
        plt.yticks(rotation=0, fontsize=label_font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(self.discovery_data_dir, output_file_name),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def get_redundant_features(self) -> Dict[str, Set[str]]:
        """
        Returns a dictionary of features to the set of features it is redundant with.
        """
        feature_corr_matrix = self.pearsons_correlation_between_features()
        np.fill_diagonal(feature_corr_matrix.values, 0)

        redundant_features = {}
        for feature in feature_corr_matrix.columns:
            correlated = feature_corr_matrix.loc[feature_corr_matrix[feature].abs() > self.redundancy_threshold].index.tolist()
            if correlated:
                redundant_features[feature] = set(correlated)

        return redundant_features

    def select_features_for_target(
        self,
        feature_target_scores_df: pd.DataFrame,
        redundant_features: Dict[str, Set[str]],
        max_features: int = 10
    ) -> Set[str]:
        """
        Selects the most relevant features for a given target.
        """

        candidate_features = (
            feature_target_scores_df
            .sort_values(by=["score", "feature"], ascending=False)['feature']
            .tolist()
        )

        selection_round = set()
        for feature in candidate_features:
            if feature not in redundant_features or all(rf not in selection_round for rf in redundant_features[feature]):
                selection_round.add(feature)
            if len(selection_round) >= max_features:
                break

        return selection_round

    def select_features(
        self,
        redundant_features: Dict[str, Set[str]],
        must_include_features: List[str] = []
    ) -> Set[str]:
        """
        Selects the most relevant features for all targets.
        """
        selected_features = set(must_include_features)

        for target in tqdm(self.target_cols, desc=f"Selecting features for all targets"):
            corr = self.pearsons_correlation_with_target(target, f"{target}_corr.csv")
            mi = self.mutual_information_with_target(target, f"{target}_mi.csv")

            feature_target_score_dfs = [corr.rename(columns={'p_corr': 'score'}), mi.rename(columns={'mi': 'score'})]
            for score_df in tqdm(feature_target_score_dfs, desc=f"Selecting features for {target}"):
                selection_round = self.select_features_for_target(score_df, redundant_features)
                selected_features.update(selection_round)

        selected_features_df = pd.DataFrame(list(selected_features), columns=["feature"]).sort_values(by="feature")
        selected_features_df.to_csv(
            os.path.join(self.gold_data_dir, "selected_features.csv"),
            index=False,
            header=False
        )

        return selected_features

    def build_gold_table(self):
        """
        Creates a gold table from the silver table.
        """

        redundant_features = self.get_redundant_features()
        selected_features = self.select_features(redundant_features, self.must_include_features)

        gold_data = self.silver_data.copy()
        gold_data = gold_data[self.metadata_cols + list(selected_features) + self.target_cols]

        gold_data.insert(0, 'id', gold_data[self.metadata_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1))
        gold_data = gold_data.drop(columns=self.metadata_cols)

        gold_data.to_csv(os.path.join(self.gold_data_dir, "final_data.csv"), index=False)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    metadata_cols = ["player", "year"]
    target_cols = ["standard_fantasy_points", "standard_fantasy_points_per_game", "ppr_fantasy_points", "ppr_fantasy_points_per_game", "value_over_replacement"]
    must_include_features = ['age']
    redundancy_threshold = 0.75

    feature_eng = FantasyFeatureEngineer(
        data_dir=data_dir,
        metadata_cols=metadata_cols,
        target_cols=target_cols,
        must_include_features=must_include_features,
        redundancy_threshold=redundancy_threshold)

    feature_eng.build_gold_table()


if __name__ == "__main__":
    main()
