---
name: inference-sequence
description: Run inference for a registered sequence fantasy-football model, currently LSTM, on its saved eval/test split or prediction season and verify MLflow metrics and prediction CSVs. Use when asked to evaluate a saved LSTM/sequential model, export its predictions, or verify inference logging without retraining.
---

# Run sequence inference

Use the saved pipeline and builder config. Do not retrain, refit preprocessing, change model/config/code, rebuild/download data, install dependencies, or alter existing models/runs/UI servers. A request to prepare a skill or explain inference is not authorization to run it.

## Resolve and preflight

1. Inspect `src/modeling/sequence/inference.py`, shared `src/modeling/inference.py`, and the current CLI help. The entry point is `uv run --no-sync python -m src.modeling.sequence.inference`, run from the repository root. Current supported modes are `eval`, `test`, and `predict`; sequence scoring reports MSE/RMSE, not tabular R²/top-N metrics.
2. Determine the registered model and tracking store from the user's request/context. Do not choose an unrelated model simply because it is newest. If mode is unclear, ask; an evaluation request without a test request may use `eval`, but never silently score `test`. A logging check can use eval only when evaluation itself has been requested. Preserve explicit data/tracking paths and versions.
3. Inspect the selected version through the existing MLflow environment. Use an explicit requested version, or resolve the requested/latest version numerically and pin it in the executed command so concurrent registration cannot change the model mid-invocation. Verify its source run, saved `data_prep_config.json`, sequence-compatible raw X/lengths signature, and feature metadata. The saved preprocessor being sklearn does not make this a tabular model; inspect top-level flavors and model-family metadata rather than component types or names alone.
4. Reconstruct `SequenceModelInference.from_registry(...)` without fitting. Verify the saved prediction season and `<data-dir>/gold/<target>__<season>.csv`; do not substitute another year's table or automatically advance the prediction season. Check the requested split is usable. Missing data, config/artifacts, mismatched features, unavailable packages, or non-finite data/output are blockers, not permission to repair or regenerate them.
5. The current loaded pipeline runs on CPU even if training used MPS/CUDA. There is no inference `--device` flag or top-N option; explain unsupported requests rather than inventing flags. Do not silently modify the saved builder config to override target, positions, features, or chronological splits.

## Execute and verify

1. State the exact registered name/version, source run, mode, saved season, data snapshot, and tracking store. The CLI creates a new inference run and writes/logs a CSV under `<data-dir>/predictions`; it does not add predictions to the original training run. Execute one invocation for each explicitly requested combination, using `uv run --no-sync`.
2. Capture output/exit status and identify the run created by this invocation. Verify its source experiment, phase/model-family tags, source_run_id/model_version parameters, and saved config; do not assume an unrelated latest run is yours. For long commands use background completion notifications, not polling or duplicate launches.
3. Verify the CSV exists locally and is logged to that exact run. Eval/test CSV rows should correspond only to loss-mask steps and include actuals. Prediction CSV rows should correspond to each player's last real step and omit actuals. Check identity/target-season alignment, finite predictions, and row counts against the selected dataset. Technical tensor coordinates are not output columns. CSV rounding is presentation only; metrics must use unrounded predictions.
4. For eval/test, report the recorded unweighted MSE/RMSE and explicitly label the split; inspect that they are finite. For predict, confirm no outcome-scoring metrics were logged. Earlier real steps remain input context even when unscored, and training sample weights do not weight held-out metrics.
5. Report failures and any partially created run/CSV without deleting them or automatically rerunning. Loading in the installed environment does not verify dependency recreation or other platforms. Do not open/start a UI or perform test scoring merely to validate eval logging.

## Report

Return the actual command, resolved model/version/source run, mode, saved target/prediction season, inference run/experiment, row count and local CSV/MLflow artifact location, metrics when applicable, and warnings/limitations. State whether verification succeeded; avoid unsupported accuracy or ranking-quality claims.

Example (only after an evaluation request):

```sh
uv run --no-sync python -m src.modeling.sequence.inference \
  --registered-model ppr_points_per_game_lstm --model-version 1 --mode eval
```
