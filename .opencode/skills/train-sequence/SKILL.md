---
name: train-sequence
description: Train and evaluate this project's implemented sequence models, currently LSTM, on CPU, MPS, or CUDA and verify MLflow logging. Use when asked to fit or retrain an LSTM/sequential fantasy-football model with a config and default or explicit training parameters, including accelerator training or model-logging verification.
---

# Train a sequence model

Run the existing sequence trainer once for each explicitly requested config/model combination. Preserve configs, source code, data, previous runs, branches, and stashes. Do not install dependencies, rebuild/download data, tune parameters, score the held-out test set, start a UI, or change code unless separately requested. Report actual outcomes, including partial failures.

## Resolve the request

1. Inspect the current sequence model implementation and CLI before choosing a workflow. Currently LSTM is implemented in `src/modeling/sequence/lstm.py` and `src/modeling/sequence/lstm_trainer.py`; run from the repository root with `uv run --no-sync python -m src.modeling.sequence.lstm_trainer`. A transformer config or sequence data builder alone is not a trainable transformer. Stop with a clear limitation for an unimplemented architecture rather than substituting another model.
2. Use a supplied YAML config as-is. If the user requests a target/config or overrides without an existing suitable file, follow `create-config` and create a separate temporary/new config with only authorized settings. The CLI requires `--config`; never silently select a customized project config or replace the original file.
3. Read the shared builder schema in `src/modeling/data_builder.py` and sequence assembly in `src/modeling/sequence/data_builder.py`. Config controls target, positions, chronological split years, sample weights, and feature selection. The resolved `split.prediction_season` selects `<data-dir>/gold/<target>__<season>.csv`; omitted seasons use the current roster season, not necessarily the calendar year. Preprocessing is fitted on real training inputs only; historical unscored steps remain context and padding is excluded.
4. Pass only explicit CLI options so omitted values come from current code. Supported options currently include `--data-dir`, `--tracking-dir`, `--hidden-size`, `--num-layers`, `--dropout`, `--epochs`, `--batch-size`, `--learning-rate`, and `--device`. Current defaults are data/mlruns, hidden size 64, one layer, dropout 0, 20 epochs, batch size 32, learning rate 0.001, and CPU. The CLI fixes Torch seed to 42. Verify these rather than treating them as permanent. Do not claim unsupported search/early-stopping/model options were honored.

## Preflight and execute

1. Confirm the config, resolved gold CSV, and required packages exist in the current uv environment without syncing. Check requested device availability: `torch.backends.mps.is_available()` for MPS, `torch.cuda.is_available()` for CUDA. If the requested accelerator is unavailable, report that and ask before falling back to CPU. Do not silently enable CPU fallbacks for unsupported accelerator operations.
2. Build/inspect the requested splits to verify a nonempty positively weighted training objective, usable eval split, feature count, player counts, sequence widths, and scored-step counts. This is data preparation, not authorization to score test outcomes. Missing raw features are allowed for fitted imputation; other builder/preprocessor errors are blockers, not permission to regenerate or repair data.
3. State the effective config, model parameters, device, input snapshot, and tracking store. Training creates an MLflow run and registered pipeline; epoch evaluation uses only the eval split. Test scoring is a separate operation: only if explicitly requested, follow `inference-sequence` after successful registration and use `uv run --no-sync python -m src.modeling.sequence.inference --registered-model <name> --model-version <version> --mode test`, pinning the version tied to this training run and preserving the requested data/tracking paths. This creates a separate source-linked inference run with test metrics and a predictions CSV; the trainer does not score the test split.
4. Execute the requested fit using `uv run --no-sync` and capture command output and exit status in managed shell output or a uniquely named temporary log. For long runs, use the harness's background execution and completion notification; do not poll or relaunch an active run. Do not run baseline/search fits just to validate this skill. A failed run may leave artifacts: report them without deleting them or automatically retrying/fixing code.

## Verify logging and reload

1. Extract this invocation's run ID from its output (`MLflow training run: ...`) and verify it in the selected tracking store. Do not choose an unrelated latest run if concurrent jobs exist. Report an unsuccessful/partial run if training, logging, or registration fails.
2. Inspect saved `data_prep_config.json` for the resolved season/config; actual model/training parameters, including device/seed; model-family/type tags; and train/eval MSE/RMSE histories. Verify metric steps and count against completed epochs. Distinguish weighted training metrics from unweighted eval metrics; never label eval metrics as test metrics or invent absent R²/top-N values.
3. Find the registered model version tied to this run, not just the latest version globally. Current names are `<target>[_positions]_lstm` in the shared base-model experiment. The artifact must restore both the fitted preprocessor and trained network, architecture, and feature order. Avoid relying on private serialization filenames when the public loaded pipeline can verify that state.
4. For logging-verification requests, reload the exact registered version with `mlflow.pyfunc.load_model`. Use a small raw train/eval input batch with matching float32 X and int64 lengths; compare finite, correctly shaped predictions against the corresponding in-memory model or saved epoch evaluation when available. Loaded inference runs on CPU even if training used an accelerator. Use reasonable numerical tolerances; MPS/CPU need not be bit-identical. Do not refit preprocessing, score test targets, or run the inference CLI merely to verify reload.
5. Loading in the existing environment does not establish that dependency recreation, CUDA, other platforms, or predictive accuracy were validated. Explicitly state those limits where relevant. Do not change an existing UI/server or tracking store to make validation pass.

## Report

Lead with success, failure, or the blocker. Include the exact command/config, actual effective parameters/device, verified input season/dimensions and split scope, run ID/experiment, registered model/version, final train/eval MSE and RMSE, and logging/reload checks. Include output/log locations and warnings relevant to the result. Do not claim success merely because a run was created or a process exited; distinguish training completion from registration/reload validation.

Example: train the LSTM with `configs/sequence_ppr_points_per_game.yaml`, defaults, and MPS:

```sh
uv run --no-sync python -m src.modeling.sequence.lstm_trainer \
  --config configs/sequence_ppr_points_per_game.yaml --device mps
```
