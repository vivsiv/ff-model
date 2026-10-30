---
name: run-tests
description: Run the project's pytest suite, a specified test file, or tests corresponding to a source file. Use when asked to run or check tests, including a full test run or targeted tests for changed code.
---

# Run tests

Run the requested tests in the current project checkout and report the actual result. Do not change code, regenerate data, install dependencies, or fix failures unless separately requested. Preserve existing user changes.

## Select tests

- With no file or selector, run the default suite using `uv run --no-sync pytest`. The current `pyproject.toml` sets `addopts = "-m 'not integration'"`; preserve this default so integration tests are excluded unless explicitly requested. Check the current configuration first and report the scope.
- For a test file or pytest node ID, run that exact selection, preserving any `::class::test` selector. A directly named integration test file or node is an explicit request to run it: override the default marker as needed, for example with `-m ""`. If the user supplies a marker or asks for integration tests explicitly, honor the requested scope; `-m integration` runs only integration tests, while `-m ""` includes both integration and non-integration tests.
- For a source file, identify corresponding tests by matching its relative directory and module name under `test/`, then inspect imports and references to find additional direct tests. For example, `src/processing/nflverse.py` maps to `test/processing/test_nflverse.py`, not the similarly named data-source tests. Run the relevant non-integration tests, deduplicated; include corresponding integration tests only if explicitly requested. Do not run an unrelated file just because it has the same basename.
- If no corresponding tests are found, say so and do not report a passing test run. If the mapping is genuinely ambiguous, explain the candidates and ask or state the justified selection. Respect explicitly requested files, test selectors, markers, or scope.

## Execute

1. From the repository root, use the existing `uv` environment and `uv run --no-sync pytest` with the selected paths and marker options. Check that the relevant dependencies and fixtures exist; do not install dependencies or rebuild data without permission. When explicitly requested, integration tests currently read real, gitignored gold CSVs and may skip when they are absent; snapshot-specific checks may fail if the data differs. Do not silently replace those skips with a data rebuild.
2. Capture the exact command, exit status, pytest summary, and relevant failure details. Report passed, failed, errors, skipped, xfailed, xpassed, and deselected counts when available. Zero collected tests, an interruption, timeout, missing prerequisites, or collection errors are not a passing suite. If an integration test is skipped or cannot run, identify it and why.
3. Report the executed scope and outcome. For failures give the failing test IDs and concise error information; distinguish test failures from setup or environment errors. Do not claim broader validation than was actually executed.

Examples: `@run-tests`, `@run-tests src/processing/gold.py`, or `@run-tests test/modeling/test_split_years.py::TestSplitYears`.
