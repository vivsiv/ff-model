---
name: inference-tabular
description: Run inference for a registered tabular fantasy-football model on its saved eval/test split or prediction season and verify MLflow scores and prediction CSVs. Use when asked to evaluate a saved sklearn/tabular model, export its predictions, or verify inference logging without retraining.
---

# Run tabular inference

Use the saved sklearn pipeline and builder config. Do not retrain, refit preprocessing, change model/config/code, rebuild/download data, install dependencies, or alter existing models/runs/UI servers. A request to prepare a skill or explain inference is not authorization to run it.

## Resolve and preflight

1. Inspect `src/modeling/tabular/inference.py`, shared `src/modeling/inference.py`, and current CLI help. Run `uv run --no-sync python -m src.modeling.tabular.inference` from the repository root. Current modes are `eval`, `test`, and `predict`.
2. Determine the registered name, mode, and data/tracking paths from the request/context. Ask if the model or operation is ambiguous; an evaluation request without a test request may use `eval`, but never silently score the held-out test split. Do not select an unrelated model because it is newest or perform inference when the user only asked for instructions/configuration.
3. Resolve an explicit version or the requested/latest numeric version in the chosen registry and pin it for execution. Verify its source run and saved `data_prep_config.json`. Inspect top-level MLflow flavors: sklearn models commonly also expose python_function; the native sklearn flavor is preferred by `load_mlflow_model` so estimator attributes remain available. Do not infer family solely from the model's name.
4. Reconstruct `TabularModelInference.from_registry(...)` without fitting. Verify `<data-dir>/gold/<target>__<prediction_season>.csv` exists for the saved season, and that the requested split/feature selection can be built. The saved season is not automatically the current calendar or upcoming season. Explain any legacy-season warning or missing config/artifact; do not silently replace the snapshot, edit saved config, or repair/rebuild data.
5. Accept current flags `--data-dir`, `--tracking-dir`, `--registered-model`, `--model-version`, `--mode`, and `--top-n-rows`. Pass explicit options and check code defaults. Top-N currently defaults to `[50, 100, 200]` for eval/test; explicitly disable with `--top-n-rows` and no values. Prediction ignores top-N. There is no accelerator/device flag for this sklearn inference CLI.

## Execute and verify

1. State the exact model/version/source run, mode, saved target/season, data snapshot, tracking store, and effective top-N choices. Execute the approved command once using `uv run --no-sync`. It creates a new source-linked inference run and writes/logs a CSV under `<data-dir>/predictions`, not the original training run.
2. Capture output and exit status. Identify the invocation's run by its source/model-version/phase and output artifact, not an arbitrary latest run. Verify it uses the source experiment (including legacy experiment names), model-family/type tags, and saved builder config. Preserve logs and partial results on failure; do not auto-fix or relaunch.
3. Verify the CSV exists and is logged to the exact inference run. Check selected player identities and target seasons, finite predictions, row counts, and actuals for eval/test only. The pipeline uses its fitted feature names/order; extra gold features may be ignored, missing required features must fail clearly. Prediction produces the saved season's player pool without scoring unknown outcomes.
4. Report actual logged RMSE/R² and requested top-N metrics when present. Current scoring can skip metrics for fewer than two rows, and R² for zero-variance actuals; distinguish those skips from success with complete metrics. Do not fabricate missing metrics or claim evaluation is test scoring. Check prediction runs have no outcome-scoring metrics. CSV values are rounded for presentation; scores use unrounded predictions.
5. Do not start/open an MLflow UI unless requested, run test evaluation solely for a logging check, or claim fresh-environment/platform portability based on loading in the installed environment.

## Report

Return the command, resolved model/version/source run, mode/season, new inference run/experiment, local CSV and artifact location/row count, available metrics and skipped ones, and verification outcome/warnings. Avoid unsupported accuracy/ranking-quality claims.

Example (only after an evaluation request):

```sh
uv run --no-sync python -m src.modeling.tabular.inference \
  --registered-model ppr_points_per_game_rb_ridge --model-version 1 --mode eval
```
