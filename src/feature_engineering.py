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
            self.gold_data_dir = os.path.join(data_dir, "gold")
            self.gold_data = self.load_gold_table()
        except Exception as e:
            logger.error(f"Gold table not found at {self.gold_data_dir}")
            raise e

        self.discovery_data_dir = os.path.join(data_dir, "discovery")
        os.makedirs(self.discovery_data_dir, exist_ok=True)

        self.metadata_cols = metadata_cols
        self.target_cols = target_cols
        self.feature_cols = [col for col in self.gold_data.columns if col not in self.metadata_cols + self.target_cols]
        self.must_include_features = must_include_features
        self.redundancy_threshold = redundancy_threshold

    def load_gold_table(self) -> pd.DataFrame:
        """
        Loads the gold table(s).

        Returns:
            DataFrame of the gold table(s)
        """
        data = pd.DataFrame()

        gold_path = os.path.join(self.gold_data_dir, "final_stats.csv")
        if os.path.exists(gold_path):
            data = pd.read_csv(gold_path)
            logger.info(f"Loaded gold table: {len(data)} rows")

        return data

    def pearsons_correlation_between_features(self, output_file_name: str = "feature_corr_matrix.csv") -> pd.DataFrame:
        """
        Runs a Pearson correlation analysis between all features.
        """
        corr_matrix = self.gold_data[self.feature_cols].corr(method='pearson')
        corr_matrix = corr_matrix.round(2)

        corr_matrix.to_csv(os.path.join(self.discovery_data_dir, output_file_name))

        return corr_matrix

    def pearsons_correlation_with_target(self, target: str, output_file_name: str = "target_corr.csv") -> pd.DataFrame:
        """
        Runs a Pearson correlation analysis between all features and a single target.
        """

        gold_data_single_target = self.gold_data[self.feature_cols].copy()
        gold_data_single_target[target] = self.gold_data[target]

        corr_matrix = gold_data_single_target.corr(method='pearson')

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
        mutual_info_values = mutual_info_regression(self.gold_data[self.feature_cols], self.gold_data[target], random_state=68)
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

    def generate_feature_analysis(self):
        """
        Generates feature analysis and saves to discovery directory.
        """

        feature_corr_matrix = self.pearsons_correlation_between_features()
        self.plot_correlation_matrix(feature_corr_matrix, title="Correlation Matrix")

        for target in self.target_cols:
            target_corr = self.pearsons_correlation_with_target(target)
            target_mi = self.mutual_information_with_target(target)

            self.plot_correlation_matrix(feature_corr_matrix, title=f"Correlation Matrix for {target}")
            self.plot_correlation_matrix(target_corr, title=f"Pearson Correlation with {target}")
            self.plot_correlation_matrix(target_mi, title=f"Mutual Information with {target}")

        self.gold_data.to_csv(os.path.join(self.gold_data_dir, "final_data.csv"), index=False)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    metadata_cols = ["player", "year"]
    target_cols = ["standard_fantasy_points", "standard_fantasy_points_per_game", "ppr_fantasy_points", "ppr_fantasy_points_per_game", "value_over_replacement"]

    feature_eng = FantasyFeatureEngineer(
        data_dir=data_dir,
        metadata_cols=metadata_cols,
        target_cols=target_cols,
    )

    feature_eng.generate_feature_analysis()


if __name__ == "__main__":
    main()
