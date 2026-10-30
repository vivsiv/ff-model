---
name: create-config
description: Create one or more model data-preparation YAML configs from a description, including targets, positions, chronological splits, sample weights, and feature selection. Use when asked to generate or customize model training configs for this project.
---

# Create model data-preparation configs

Translate the user's description into one or more valid YAML files for the project's current data-preparation code. Create configs only; do not train models or rebuild data unless separately requested.

## Determine the configs

1. Inspect the shared config schema in `src/modeling/data_builder.py`, `src/modeling/tabular/data_builder.py`, relevant config examples in `configs/`, and, for sequence configs, `src/modeling/sequence/data_builder.py`. Verify the current schema and defaults rather than relying on a static copy. Tabular and sequence data builders inherit the shared `ModelDataBuilder` schema; a sequence config does not imply that a sequence training implementation exists.
2. Determine the requested target, number of configs, their differences, output paths, and any explicit base config. Ask for missing or ambiguous choices instead of inventing targets, position groups, split sizes, feature lists, or weights. Omit unspecified optional settings so the code supplies its defaults. Do not copy a customized project config unless the user chooses it as a base or clearly asks for its settings: the example configs have non-default feature exclusions, weights, and training windows.
3. If a base config is supplied, preserve its other settings and apply only the requested changes. Support requests for multiple distinct configs, such as one per position or target. Put new reusable files under `configs/` by default with descriptive, unique `.yaml` filenames; respect explicit paths. Show the proposed paths and effective differences. If an intended path exists, ask before overwriting unless the user explicitly requested that exact replacement; otherwise choose a non-conflicting path and tell the user.
4. Validate target names against the current gold/column registry, and check user-provided positions, split values, weighting buckets, feature modes and patterns against the implementation. Ask if a requested setting is unsupported or unclear; never add estimator hyperparameters or CLI arguments to the YAML as though the data-prep code consumed them. Flag potentially empty splits, invalid or unmatched feature patterns, unusable weights, or future-information leakage when evidence is available. Historical performance available before prediction is not automatically leakage.

## Current schema details

- `target` is required. `positions` is an optional list; omitting it or providing an empty list keeps all positions.
- `split` can contain `eval_data_years`, `test_data_years`, and `num_training_seasons`. The current defaults are 1, 1, and no limit, respectively. The split uses `target_season` chronologically.
- `sample_weights` can contain `global_weight` or `buckets` of `{min, weight}`. Currently omitted/empty buckets use global weight 1.0 unless specified; nonempty buckets determine the weight and rows below the smallest `min` get zero weight. Clarify ambiguous bucket boundaries rather than guessing.
- `features` takes `mode: include` or `mode: exclude` and a `columns` list. The implementation supports `*` at the beginning, end, or both ends for matching; it is not a general glob. Quote YAML values that begin with `*`. If omitted, the code uses all non-identity, non-target candidate columns.

## Write and verify

1. Render each config as YAML that follows the current schema. Preserve unrelated files and user changes. Do not modify an existing config in place without the appropriate authorization above. Use a temporary file if a training caller requests an ephemeral config; otherwise keep the newly created configs at the agreed paths.
2. Round-trip parse each written YAML and validate its required keys, types, and requested differences. When the gold training file exists, use read-only checks to confirm requested positions, selected feature patterns, and chronological coverage against it where practical; do not train or mutate the data. If the file is absent, still validate the config structurally and clearly state which data-dependent checks could not run.
3. Return every created path, its target and meaningful settings or base/overrides, applicable defaults, and validation results or limitations. If only some configs were created, report partial results accurately.

Examples: `@create--config create a ppr_points_per_game config with one eval season, one test season, and the last 10 training seasons`; or `@create-config copy configs/ppr_points_per_game.yaml into separate RB and WR configs`.
