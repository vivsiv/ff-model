import os
from typing import List

import yaml

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "column_registry.yaml")


def _load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def get_identity_columns(source: str, table: str) -> List[str]:
    """
    Returns the columns from `source`/`table` that are kept for joins/grouping/output but are
    not career-averaged as a stat.

    Args:
        source: Data source name, e.g. "nflverse"
        table: Silver table name, e.g. "player_stats"

    Returns:
        List of column names
    """
    return _load_registry()[source][table]["identity"]


def get_stat_columns(source: str, table: str) -> List[str]:
    """
    Returns the columns from `source`/`table` that are candidates for feature engineering.

    `stats` is either a flat list, or (for tables that classify stats into "counting"/"rate",
    e.g. player_stats/snap_counts, see get_counting_stat_columns) a dict of sub-lists -- this
    flattens either shape into a single list, so callers don't need to care which one a given
    table uses.

    Args:
        source: Data source name, e.g. "nflverse"
        table: Silver table name, e.g. "player_stats"

    Returns:
        List of column names
    """
    stats = _load_registry()[source][table]["stats"]
    if isinstance(stats, dict):
        return [col for columns in stats.values() for col in columns]

    return stats


def get_counting_stat_columns(source: str, table: str) -> List[str]:
    """
    Returns the "counting" (season-total) stat columns from `source`/`table` -- these are
    eligible for gold.py's per-game transform, unlike "rate" stats (e.g. target_share,
    passing_cpoe), which are already normalized. Only defined for tables whose `stats` is
    split into "counting"/"rate" sub-lists (currently player_stats, snap_counts).

    Args:
        source: Data source name, e.g. "nflverse"
        table: Silver table name, e.g. "player_stats"

    Returns:
        List of column names
    """
    return _load_registry()[source][table]["stats"]["counting"]


def get_targets(source: str, table: str) -> List[str]:
    """
    Returns the columns from `source`/`table` that are candidate prediction targets (for
    _join_with_target's target_col). Always a subset of get_stat_columns(source, table).

    Args:
        source: Data source name, e.g. "nflverse"
        table: Silver table name, e.g. "player_stats"

    Returns:
        List of column names
    """
    return _load_registry()[source][table]["targets"]
