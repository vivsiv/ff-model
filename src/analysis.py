import os
import logging
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataAnalysis:
    def __init__(
        self,
        data_dir: str = "../data",
        metadata_cols: List[str] = ["id"],
        target_cols: List[str] = ["standard_fantasy_points", "standard_fantasy_points_per_game", "ppr_fantasy_points", "ppr_fantasy_points_per_game", "value_over_replacement"],
    ):

        self.data_dir = data_dir
        try:
            self.gold_data_dir = os.path.join(data_dir, "gold")
            self.training_data, self.live_data = self.load_data()
        except Exception as e:
            logger.error(f"Error loading data at: {self.gold_data_dir}: {e}")
            raise e

        self.discovery_data_dir = os.path.join(data_dir, "discovery")
        os.makedirs(self.discovery_data_dir, exist_ok=True)

        self.metadata_cols = metadata_cols
        self.target_cols = target_cols
        self.feature_cols = [col for col in self.training_data.columns if col not in self.metadata_cols + self.target_cols]

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        training_data_path = os.path.join(self.gold_data_dir, "training_set.csv")
        training_data = pd.read_csv(training_data_path)
        logger.info(f"Loaded training data: {len(training_data)} rows")

        live_data_path = os.path.join(self.gold_data_dir, "live_set.csv")
        live_data = pd.read_csv(live_data_path)
        logger.info(f"Loaded live data: {len(live_data)} rows")

        return training_data, live_data

    def run_training_data_quality_checks(self) -> None:
        training_shape = self.training_data.shape
        assert training_shape[0] > 8000, f"Training data must have at least 8000 rows, got {training_shape[0]}"
        assert training_shape[1] >= 150, f"Training data must have at least 150 columns, got {training_shape[1]}"

        non_float_columns = self.training_data.select_dtypes(exclude=['float64']).columns
        assert len(non_float_columns) == 1, f"Id should be the only non float column, got {non_float_columns}"
        assert self.training_data['id'].dtype == 'object', "Id should be a string"

        # Check for rows where the id is only: _YYYY (the name is missing)
        year_only_rows = self.training_data[self.training_data['id'].str.startswith('_')]
        assert len(year_only_rows) == 0, f"Training data must not have rows where the id is missing the name, got {len(year_only_rows)}"

        duplicates = self.training_data.duplicated()
        assert not duplicates.any(), f"Training data must not have duplicates, got {duplicates.sum()}"

        null_value_rows = self.training_data.isnull().any(axis=1)
        assert not null_value_rows.any(), f"Training data must not have rows with null values, got {null_value_rows.sum()}"

        all_zero_rows = self.training_data[self.feature_cols].eq(0).all(axis=1)
        assert not all_zero_rows.any(), f"Training data must not have rows that are 0 for all features, got {all_zero_rows.sum()}"

        mostly_zero_rows = self.training_data[self.feature_cols].eq(0).sum(axis=1) / len(self.feature_cols) > 0.95
        assert not mostly_zero_rows.any(), f"Training data must not have rows that are 0 for 95% of features, got {mostly_zero_rows.sum()}"

        # Spot check some known rookies to see that they are not there
        rookies = ['malik_nabers_2024', 'jamarr_chase_2021', 'saquon_barkley_2018', 'dak_prescott_2016', 'aaron_rodgers_2005']
        for rookie in rookies:
            rookie_rows = self.training_data[self.training_data['id'] == rookie]
            assert len(rookie_rows) == 0, f"Training data must not have rookies, got {rookie}"

        # Spot check some known joins to see that they are correct
        aaron_rodgers_2012_pass_touchdowns = self.training_data[self.training_data['id'] == 'aaron_rodgers_2012']['pass_touchdowns'].iloc[0]
        assert aaron_rodgers_2012_pass_touchdowns == 45, f"Aaron Rodgers' 2012 row should have 2011's passing touchdowns (45), got {aaron_rodgers_2012_pass_touchdowns}"

        christian_mccaffrey_2020_rec_receptions = self.training_data[self.training_data['id'] == 'christian_mccaffrey_2020']['rec_receptions'].iloc[0]
        assert christian_mccaffrey_2020_rec_receptions == 116, f"Christian McCaffrey's 2020 row should have 2019's receptions (116), got {christian_mccaffrey_2020_rec_receptions}"

        saquon_barkley_2024_rushing_yards = self.training_data[self.training_data['id'] == 'saquon_barkley_2024']['rush_yards'].iloc[0]
        assert saquon_barkley_2024_rushing_yards == 962, f"Saquon Barkley's 2024 row should have 2023's rushing yards (962), got {saquon_barkley_2024_rushing_yards}"

        priest_holmes_2002_rush_touchdowns = self.training_data[self.training_data['id'] == 'priest_holmes_2002']['rush_touchdowns'].iloc[0]
        assert priest_holmes_2002_rush_touchdowns == 8, f"Priest Holmes' 2002 row should have 2001's rushing touchdowns (8), got {priest_holmes_2002_rush_touchdowns}"

        terrell_owens_2005_rec_yards = self.training_data[self.training_data['id'] == 'terrell_owens_2005']['rec_yards'].iloc[0]
        assert terrell_owens_2005_rec_yards == 1200, f"Terrell Owens' 2005 row should have 2004's rec yards (1200), got {terrell_owens_2005_rec_yards}"

        terrell_owens_2005_games = self.training_data[self.training_data['id'] == 'terrell_owens_2005']['games'].iloc[0]
        assert terrell_owens_2005_games == 14, f"Terrell Owens' 2005 row should have 2004's games (14), got {terrell_owens_2005_games}"

        self.training_data['year'] = self.training_data['id'].str.split('_').str[-1].astype(int)

        # Should have at least 350 rows for each year
        year_counts = self.training_data['year'].value_counts()
        year_counts_below_350 = year_counts[year_counts < 350]
        assert len(year_counts_below_350) == 0, f"Training data must have at least 350 rows for each year, got {year_counts_below_350['year'].tolist()}"

        # Check that the last year of data is dropped
        last_year_data = self.training_data[self.training_data['year'] == 2000]
        assert len(last_year_data) == 0, f"Training data must not have the last year of data, got {len(last_year_data)}"

        self.training_data.drop(columns=['year'], inplace=True)

    def run_live_data_quality_checks(self) -> None:
        live_shape = self.live_data.shape
        assert live_shape[0] > 250, f"Live data must have at least 200 rows, got {live_shape[0]}"
        assert live_shape[1] >= 150, f"Live data must have at least 150 columns, got {live_shape[1]}"

        non_float_columns = self.live_data.select_dtypes(exclude=['float64']).columns
        assert len(non_float_columns) == 1, f"Id should be the only non float column, got {non_float_columns}"

        duplicates = self.live_data.duplicated()
        assert not duplicates.any(), f"Live data must not have duplicates, got {duplicates.sum()}"

        null_value_rows = self.live_data.isnull().any(axis=1)
        assert not null_value_rows.any(), f"Live data must not have rows with null values, got {null_value_rows.sum()}"

        all_zero_rows = self.live_data[self.feature_cols].eq(0).all(axis=1)
        assert not all_zero_rows.any(), f"Live data must not have rows that are 0 for all features, got {all_zero_rows.sum()}"

        mostly_zero_rows = self.live_data[self.feature_cols].eq(0).sum(axis=1) / len(self.feature_cols) > 0.90
        assert not mostly_zero_rows.any(), f"Live data must not have rows that are 0 for 95% of features, got {mostly_zero_rows.sum()}"

        # Spot check some known rows
        saquon_barkley_rushing_yards = self.live_data[self.live_data['id'] == 'saquon_barkley_rb']['rush_yards'].iloc[0]
        assert saquon_barkley_rushing_yards == 2005, f"Saquon Barkley's rushing yards should be 2005, got {saquon_barkley_rushing_yards}"

        jamarr_chase_rec_yards = self.live_data[self.live_data['id'] == 'jamarr_chase_wr']['rec_yards'].iloc[0]
        assert jamarr_chase_rec_yards == 1708, f"Jamarr Chase's rec yards should be 1708, got {jamarr_chase_rec_yards}"

        josh_allen_pass_touchdowns = self.live_data[self.live_data['id'] == 'josh_allen_qb']['pass_touchdowns'].iloc[0]
        assert josh_allen_pass_touchdowns == 28, f"Josh Allen's pass touchdowns should be 28, got {josh_allen_pass_touchdowns}"

        josh_allen_rush_touchdowns = self.live_data[self.live_data['id'] == 'josh_allen_qb']['rush_touchdowns'].iloc[0]
        assert josh_allen_rush_touchdowns == 12, f"Josh Allen's rush touchdowns should be 12, got {josh_allen_rush_touchdowns}"

        christian_mccaffrey_games = self.live_data[self.live_data['id'] == 'christian_mccaffrey_rb']['games'].iloc[0]
        assert christian_mccaffrey_games == 4, f"Christian McCaffrey's games should be 4, got {christian_mccaffrey_games}"

    def pearsons_correlation_between_features(self, output_file_name: str = "feature_corr_matrix.csv") -> pd.DataFrame:
        """
        Runs a Pearson correlation analysis between all features.
        """
        corr_matrix = self.training_data[self.feature_cols].corr(method='pearson')
        corr_matrix = corr_matrix.round(2)

        corr_matrix.to_csv(os.path.join(self.discovery_data_dir, output_file_name))

        return corr_matrix

    def pearsons_correlation_with_target(self, target: str, output_file_name: str = "target_corr.csv") -> pd.DataFrame:
        """
        Runs a Pearson correlation analysis between all features and a single target.
        """

        training_data_single_target = self.training_data[self.feature_cols].copy()
        training_data_single_target[target] = self.training_data[target]

        corr_matrix = training_data_single_target.corr(method='pearson')

        # Double brackets returns a dataframe as opposed to a series
        # The .drop() call drops the row with index label exam_score (axis is set to 0 by default)
        corr_matrix = corr_matrix[[target]].drop(axis=0, labels=[target])
        corr_matrix = corr_matrix.reset_index().rename(columns={'index': 'feature', target: 'p_corr'})
        corr_matrix['p_corr'] = corr_matrix['p_corr'].round(2)
        corr_matrix = corr_matrix.sort_values(by='p_corr', ascending=False)

        corr_matrix.to_csv(os.path.join(self.discovery_data_dir, output_file_name), index=False)

        return corr_matrix.set_index('feature')

    def mutual_information_with_target(self, target: str, output_file_name: str = "target_mi.csv") -> pd.DataFrame:
        """
        Runs a mutual information analysis between all features and a single target.
        """
        mutual_info_values = mutual_info_regression(self.training_data[self.feature_cols], self.training_data[target], random_state=68)
        mutual_info = pd.DataFrame({
            'feature': self.feature_cols,
            'mi': [round(mi, 2) for mi in mutual_info_values]
        })
        mutual_info = mutual_info.sort_values(by="mi", ascending=False)

        mutual_info.to_csv(os.path.join(self.discovery_data_dir, output_file_name), index=False)

        return mutual_info.set_index('feature')

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str,
        output_file_name: str,
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
        self.plot_correlation_matrix(feature_corr_matrix, title="Correlation Matrix", output_file_name="feature_corr_matrix.png")

        for target in self.target_cols:
            target_corr = self.pearsons_correlation_with_target(target, f"{target}_corr.csv")
            target_mi = self.mutual_information_with_target(target, f"{target}_mi.csv")

            self.plot_correlation_matrix(target_corr, title=f"Pearson Correlation with {target}", output_file_name=f"{target}_corr.png")
            self.plot_correlation_matrix(target_mi, title=f"Mutual Information with {target}", output_file_name=f"{target}_mi.png")


def main():
    data_analysis = DataAnalysis()

    data_analysis.generate_feature_analysis()


if __name__ == "__main__":
    main()
