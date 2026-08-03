import os
from typing import List

import yaml

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "column_registry.yaml")


def _load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def get_included_columns(table: str) -> List[str]:
    """
    Returns the columns from `table` that are candidates for career-feature engineering
    (stat_columns in TrainingSetBuilder._add_career_features / _positional_baseline), per
    column_registry.yaml.

    Args:
        table: Silver table name, e.g. "player_stats"

    Returns:
        List of column names
    """
    return _load_registry()[table]["included"]


def get_excluded_columns(table: str) -> List[str]:
    """
    Returns the columns from `table` that are explicitly not stat_columns candidates --
    either identity/context columns, or stats judged not useful for fantasy modeling.

    Args:
        table: Silver table name, e.g. "player_stats"

    Returns:
        List of column names
    """
    return _load_registry()[table]["excluded"]


def get_targets(table: str) -> List[str]:
    """
    Returns the columns from `table` that are candidate prediction targets (for
    _join_with_target's target_col). Always a subset of get_included_columns(table).

    Args:
        table: Silver table name, e.g. "player_stats"

    Returns:
        List of column names
    """
    return _load_registry()[table]["targets"]
