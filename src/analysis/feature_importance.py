import os
import logging
import argparse
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline

from src.modeling.utils import load_mlflow_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_importance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def pearsons_correlation_between_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix between every pair of columns in feature_cols.

    Args:
        df: DataFrame containing feature_cols
        feature_cols: Columns to correlate against each other

    Returns:
        Square DataFrame of pairwise Pearson correlations, rounded to 2 decimal places.
    """
    return df[feature_cols].corr(method="pearson").round(2)


def pearsons_correlation_with_target(df: pd.DataFrame, feature_cols: List[str], target: str) -> pd.DataFrame:
    """
    Computes each feature's Pearson correlation with a single target column.

    Args:
        df: DataFrame containing feature_cols and target
        feature_cols: Columns to correlate against target
        target: Column to correlate features against

    Returns:
        DataFrame indexed by "feature" with a single "p_corr" column, sorted descending by
        correlation strength (most positively correlated first).
    """
    corr_matrix = df[feature_cols + [target]].corr(method="pearson")

    corr_with_target = (
        corr_matrix[[target]]
        .drop(index=target)
        .reset_index()
        .rename(columns={"index": "feature", target: "p_corr"})
    )
    corr_with_target["p_corr"] = corr_with_target["p_corr"].round(2)
    corr_with_target = corr_with_target.sort_values(by="p_corr", ascending=False)

    return corr_with_target.set_index("feature")


def mutual_information_with_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    random_state: int = 68,
) -> pd.DataFrame:
    """
    Computes each feature's mutual information with a single target column.

    Args:
        df: DataFrame containing feature_cols and target
        feature_cols: Columns to score against target
        target: Column to score features against
        random_state: Passed through to sklearn's mutual_info_regression for
            reproducibility (default: 68)

    Returns:
        DataFrame indexed by "feature" with a single "mi" column, sorted descending by
        mutual information.
    """
    mutual_info_values = mutual_info_regression(df[feature_cols], df[target], random_state=random_state)
    mutual_info = pd.DataFrame({
        "feature": feature_cols,
        "mi": [round(mi, 2) for mi in mutual_info_values],
    })

    return mutual_info.sort_values(by="mi", ascending=False).set_index("feature")


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str,
    font_size: int = 12,
    annot_size: int = 8,
) -> plt.Axes:
    """
    Plots a correlation-style matrix (Pearson correlation or mutual information) as a
    heatmap, with figure size scaled to the number of features so it stays readable
    whether it's a handful of features or the full set.

    Args:
        corr_matrix: Correlation (or MI) matrix DataFrame to plot
        title: Plot title
        font_size: Base font size for axis labels (default: 12)
        annot_size: Font size for the values annotated on each cell (default: 8)

    Returns:
        The Axes the heatmap was drawn on. Caller is responsible for saving/showing/closing
        the figure.
    """
    n_features = len(corr_matrix)

    base_size = 0.6
    min_size = 12
    max_size = 60
    fig_size = min(max(n_features * base_size, min_size), max_size)

    _, ax = plt.subplots(figsize=(fig_size, fig_size))

    label_font_size = max(8, font_size - (n_features // 30))
    value_font_size = max(6, annot_size - (n_features // 40))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": value_font_size},
        ax=ax,
    )

    ax.set_title(title, fontsize=label_font_size + 4)
    ax.tick_params(axis="x", labelrotation=45, labelsize=label_font_size)
    ax.tick_params(axis="y", labelrotation=0, labelsize=label_font_size)
    plt.setp(ax.get_xticklabels(), ha="right")
    plt.tight_layout()

    return ax


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Reads a fitted sklearn Pipeline's tree-based model feature_importances_ back out
    against their (post-preprocessing) feature names.

    Args:
        pipeline: A fitted sklearn Pipeline whose final step is a tree-based model exposing
            feature_importances_ (e.g. RandomForestRegressor, GradientBoostingRegressor),
            with every preceding step implementing get_feature_names_out (e.g.
            TabularModel.create_pipeline's imputer/scaler steps).

    Returns:
        DataFrame with "feature"/"importance" columns, sorted descending by importance.
    """
    feature_names = pipeline[:-1].get_feature_names_out()
    return pd.DataFrame({
        "feature": feature_names,
        "importance": pipeline["model"].feature_importances_,
    }).sort_values("importance", ascending=False)


def plot_feature_importance(feature_importance_df: pd.DataFrame, num_features: int = 30) -> plt.Axes:
    """
    Plots the top num_features most important features as a horizontal bar chart.

    Args:
        feature_importance_df: Output of get_feature_importance
        num_features: Number of top features to plot (default: 30)

    Returns:
        The Axes the bar chart was drawn on.
    """
    return feature_importance_df[:num_features].plot(
        x="feature", y="importance", kind="barh", figsize=(12, 7), legend=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Loads a registered tabular model from mlflow, computes its feature "
                     "importance, and writes it to data_dir/analysis/"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Parent directory to write the analysis/ output to, relative to the repo root (default: data)"
    )
    parser.add_argument(
        "--tracking-dir",
        type=str,
        default="mlruns",
        help="Top-level mlruns tracking/registry store directory, relative to the repo root (default: mlruns)"
    )
    parser.add_argument(
        "--registered-model",
        type=str,
        default=None,
        help="The registered model to load."
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Specific registered model version to load. Defaults to the latest version."
    )

    args = parser.parse_args()

    pipeline, mv = load_mlflow_model(args.registered_model, args.model_version, args.tracking_dir)
    feature_importance_df = get_feature_importance(pipeline)

    output_dir = os.path.join(args.data_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir, f"{args.registered_model}_v{mv.version}_feature_importance.csv"
    )
    feature_importance_df.to_csv(output_path, index=False)
    logger.info(f"Saved feature importance for {args.registered_model} v{mv.version} to {output_path}")


if __name__ == "__main__":
    main()
