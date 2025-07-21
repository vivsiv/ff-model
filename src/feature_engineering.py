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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the feature engineer.

        Args:
            data_dir: Directory containing the processed data
        """
        self.data_dir = data_dir
        self.silver_data_dir = os.path.join(data_dir, "silver")
        self.gold_data_dir = os.path.join(data_dir, "gold")
        self.discovery_data_dir = os.path.join(data_dir, "discovery")

        os.makedirs(self.discovery_data_dir, exist_ok=True)
        os.makedirs(self.gold_data_dir, exist_ok=True)

    def load_silver_table(self) -> Dict[str, pd.DataFrame]:
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

    def create_feature_pipeline(self) -> Pipeline:
        """
        Creates a feature engineering pipeline.
        """
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])

    def pearsons_correlation_between_features(self, df: pd.DataFrame, features: List[str], output_file_name: str = "feature_corr_matrix.csv") -> pd.DataFrame:
        """
        Runs a Pearson correlation analysis on the features and target.
        """
        corr_matrix = df[features].corr(method='pearson')
        corr_matrix = corr_matrix.round(2)
        corr_matrix.to_csv(os.path.join(self.discovery_data_dir, output_file_name))
        return corr_matrix

    def pearsons_correlation_with_target(self, df: pd.DataFrame, features: List[str], target: str, output_file_name: str = "feature_target_corr_matrix.csv") -> pd.DataFrame:
        """
        Runs a Pearson correlation analysis on the features and target.
        """

        df_with_target = df[features].copy()
        df_with_target[target] = df[target]

        corr_matrix = df_with_target.corr()

        # Double brackets returns a dataframe as opposed to a series
        # The .drop() call drops the row with index label exam_score (axis is set to 0 by default)
        corr_matrix = corr_matrix[[target]].drop(axis=0, labels=[target])
        corr_matrix = corr_matrix.reset_index().rename(columns={'index': 'feature', target: 'p_corr'})
        corr_matrix['p_corr'] = corr_matrix['p_corr'].round(2)
        corr_matrix = corr_matrix.sort_values(by='p_corr', ascending=False)

        corr_matrix.to_csv(os.path.join(self.discovery_data_dir, output_file_name), index=False)

        return corr_matrix

    def mutual_information_with_target(self, df: pd.DataFrame, features: List[str], target: str, output_file_name: str = "feature_target_mutual_info.csv") -> pd.DataFrame:
        """
        Runs a mutual information analysis on the features and target.
        """
        mutual_info_values = mutual_info_regression(df[features], df[target], random_state=68)
        mutual_info = pd.DataFrame({
            'feature': features,
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

    def get_redundant_features(self, feature_corr_matrix: pd.DataFrame, redundancy_threshold: float = 0.95) -> Dict[str, Set[str]]:
        """
        Returns a set of features who's correlation with another feature exceeds the redudancy threshold.
        """

        corr_matrix = feature_corr_matrix.copy()
        np.fill_diagonal(corr_matrix.values, 0)

        redundant_features = {}
        for feature in corr_matrix.columns:
            correlated = corr_matrix.loc[corr_matrix[feature].abs() > redundancy_threshold].index.tolist()
            if correlated:
                redundant_features[feature] = set(correlated)

        return redundant_features

    def select_features_for_target(self, target_score_df: pd.DataFrame, redundant_features: Dict[str, Set[str]], auto_add_top_10: bool = False) -> Set[str]:

        """
        Selects the most relevant features for a given target.
        """

        p50_features_scores = target_score_df.sort_values(by="score", ascending=False).head(len(target_score_df) // 2)

        candidate_features = p50_features_scores['feature'].tolist()
        selection_round = set(candidate_features[:10]) if auto_add_top_10 else set()
        for feature in candidate_features:
            if feature not in redundant_features or all(redundancy not in selection_round for redundancy in redundant_features[feature]):
                selection_round.add(feature)

        return selection_round

    def select_all_features(self, targets: List[str], features: List[str], silver_df: pd.DataFrame, redundant_features: Dict[str, Set[str]]) -> Set[str]:
        """
        Selects the most relevant features for all targets.
        """
        selected_features = set()
        for target in tqdm(targets, desc=f"Selecting features for targets {targets}"):
            corr = self.pearsons_correlation_with_target(silver_df, features, target, f"{target}_corr.csv")
            mi = self.mutual_information_with_target(silver_df, features, target, f"{target}_mi.csv")

            target_score_dfs = [corr.rename(columns={'p_corr': 'score'}), mi.rename(columns={'mi': 'score'})]
            for target_score_df in target_score_dfs:
                selection_round = self.select_features_for_target(target_score_df, redundant_features, auto_add_top_10=True)
                selected_features.update(selection_round)

        selected_features_df = pd.DataFrame(list(selected_features), columns=["feature"])
        selected_features_df.sort_values(by="feature").to_csv(os.path.join(self.gold_data_dir, "selected_features.csv"), index=False)

        return selected_features

    def create_gold_table(self, silver_data: pd.DataFrame, metadata_cols: List[str], select_cols: List[str], target_cols: List[str]):
        """
        Creates a gold table from the silver table.
        """
        gold_data = silver_data[metadata_cols + select_cols + target_cols]
        gold_data.to_csv(os.path.join(self.gold_data_dir, "final_data.csv"), index=False)

        return gold_data


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    feature_eng = FantasyFeatureEngineer(data_dir=data_dir)

    silver_data = feature_eng.load_silver_table()
    metadata_cols = ["player", "year", "team"]
    target_cols = ["standard_fantasy_points", "standard_fantasy_points_per_game", "ppr_fantasy_points", "ppr_fantasy_points_per_game", "value_over_replacement"]

    non_feature_cols = metadata_cols + target_cols
    features = [col for col in silver_data.columns if col not in non_feature_cols]

    feature_corr = feature_eng.pearsons_correlation_between_features(silver_data, features, "feature_corr_matrix.csv")
    redundant_features = feature_eng.get_redundant_features(feature_corr, 0.95)

    selected_features = feature_eng.select_all_features(target_cols, features, silver_data, redundant_features)

    feature_eng.create_gold_table(metadata_cols, list(selected_features), target_cols, silver_data)


if __name__ == "__main__":
    main()
