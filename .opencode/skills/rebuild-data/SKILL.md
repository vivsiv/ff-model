---
name: rebuild-data
description: Rebuild this project's bronze, silver, or gold datasets and their required upstream tables. Use when asked to regenerate or refresh a data layer, selected source tables, or gold training and prediction datasets.
---

# Rebuild datasets

Rebuild the requested layer and all of its required upstream dependencies, even when upstream files already exist. Work in dependency order and rebuild each shared dependency only once. Do not rebuild downstream layers or unrelated tables.

## Interpret the request

- Require a layer: `bronze`, `silver`, or `gold`. Ask if it is missing or ambiguous.
- For bronze and silver, accept a data source and table list. Default to `nflv` (NFLverse) and all tables implemented for that source and layer. Accept `nflverse` as an alias for `nflv`, and `pfr` or `pro_football_reference` for the legacy PFR source. Normalize table names with or without `.csv`. Ask about unknown sources or tables; do not silently ignore them or replace a selective request with an all-tables build.
- For gold, accept the current `src.processing.gold` CLI parameters except that the prediction season is fixed at **2026** for now. Check the code or `uv run --no-sync python -m src.processing.gold --help` and confirm its default is still 2026 before changing data. Omit `--prediction-season` when invoking the CLI; if the user requests a different season or the code's default changes, explain the restriction and stop rather than silently using another year. An explicit request for 2026 is compatible with this restriction. For other parameters, omit unspecified options so the script's defaults apply. The gold CLI builds both training and prediction sets and currently reads NFLverse silver data. Pass its `--data-dir` through to all upstream builders. Do not invent a separate gold source or table selector.
- Support a supplied data directory for bronze and silver too; otherwise use the builders' `data` default. Resolve relative paths from the repository root.

## Plan dependencies

Read the current source code in `src/data/` and `src/processing/` to identify implemented outputs, their inputs, defaults, and methods. The graph below describes the current implementation, not a substitute for checking changes to those files. Expand dependencies transitively, deduplicate them, and identify the exact methods and output paths before running. Include source years, the network requests, and any existing files that will be overwritten in the plan; communicate the plan to the user. A request to rebuild authorizes replacing the selected generated tables and their necessary upstream generated tables, not deleting unrelated files, resetting the repository, or overwriting unrelated user data.

### Current NFLverse graph

Bronze tables `player_stats`, `team_stats`, `draft_picks`, `snap_counts`, `players`, `ff_rankings`, and `ff_playerids` are fetched independently by the corresponding `NflverseScraper.fetch_*` methods in `src/data/nflverse.py`. `fetch_all()` fetches all seven; the module CLI always calls it, so use individual methods for a subset.

| Silver table | Required upstream tables |
| --- | --- |
| `player_stats` | bronze `player_stats` |
| `team_stats` | bronze `team_stats` |
| `draft_picks` | bronze `draft_picks` |
| `player_ids` | bronze `players` |
| `snap_counts` | bronze `snap_counts` and silver `player_ids` (which requires bronze `players`) |
| `ff_rankings` | bronze `ff_rankings` and `ff_playerids` |

Use `NflverseProcessor.build_*` methods in `src/processing/nflverse.py` for selective builds. In particular, pass the dataframe returned by `build_player_ids()` into `build_snap_counts()`; it cannot be called without that argument. Silver `ff_playerids` and `players` are not tables in this implementation.

The current gold CLI needs silver `player_stats`, `snap_counts`, `team_stats`, `draft_picks`, and `ff_rankings`; expand their dependencies using the graph above. It writes `{target_col}__training_set.csv` and `{target_col}__prediction_set.csv` under `data-dir/gold`. A gold rebuild does not need PFR data.

### Current PFR graph

`src/data/pro_football_reference.py` scrapes per-season bronze files: `player_fantasy_stats`, `player_passing_stats`, `player_rushing_stats`, `player_rushing_advanced_stats`, `player_receiving_stats`, `player_receiving_advanced_stats`, and `team_offense`. Its CLI scrapes all of them for 2023–2024 by default. For selective builds use `scrape_player_fantasy_stats`, `scrape_player_offensive_stats` with the appropriate category, or `scrape_team_offensive_stats`, once per season.

`src/processing/pro_football_reference.py` builds silver `player_fantasy_stats`, `player_passing_stats`, `player_rushing_stats`, `player_receiving_stats`, and `team_offense` from the same-named per-season bronze files. The team output is `team_offense`, although its builder is `build_team_stats`. The advanced bronze tables have no corresponding silver builder in the current implementation.

PFR silver builders combine **all** matching bronze years, not just a passed year range. Before rebuilding their inputs, determine the complete range of matching existing bronze years and ensure every year included in the eventual silver result is rebuilt; do not silently combine newly rebuilt years with stale older files. If the intended historical coverage is unclear, ask before proceeding. For direct bronze requests use an explicitly requested year range or the scraper CLI's current defaults, and report the range. PFR caches downloaded HTML and defaults to reusing it: preserve that behavior unless the user explicitly asks to refresh the cached HTML. The project notes that PFR blocks automated scraping; report failures rather than attempting to bypass restrictions.

## Execute and verify

1. Run from the repository root using the existing `uv` environment; prefer `uv run --no-sync` to avoid surprise dependency changes. If dependencies are unavailable, explain the blocker instead of installing them without permission. Inspect any source-specific CLI defaults and requirements before use.
2. Run the plan sequentially in topological order. Call only selected methods for partial builds. For NFLverse, check each fetch's returned dataframe: an empty fetch logs a warning and leaves the old CSV untouched, so treat it as failure rather than silently processing stale data. For PFR, likewise check each requested table and season for a nonempty successful result. Stop dependent builds on a failed or incomplete upstream step; report what succeeded and any files already overwritten. Never claim an old file was refreshed just because it exists.
3. Invoke gold using the actual CLI, for example `uv run --no-sync python -m src.processing.gold` plus only the provided options other than `--prediction-season`. Use the verified 2026 code default; do not reconstruct other defaults or alter source code.
4. Verify the expected outputs for each executed stage and perform lightweight sanity checks appropriate to the table, such as readable CSV headers, nonempty rows, and relevant season coverage. Do not treat a successful process exit alone as proof that every requested fetch or output succeeded. Do not delete or replace unrelated data, caches, or generated outputs.
5. Summarize the requested layer and parameters, the upstream and target tables rebuilt, source and year coverage where applicable, output paths, validation results, and any failures or partial results. Explicitly flag unverified coverage or stale cached inputs.

Example requests: `@rebuild-data bronze nflv player_stats team_stats`, `@rebuild-data silver nflv snap_counts`, or `@rebuild-data gold --target-col ppr_points_per_game`.
