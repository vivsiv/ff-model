---
name: mlflow-ui
description: Find or start this project's local MLflow UI and return its URL. Use when asked to open, launch, start, find, or access the MLflow UI or dashboard, including to view experiment runs in the UI.
---

# MLflow UI

Find a running UI for this project's `mlruns` tracking store, or start one if none is running. Return the URL; do not open a browser unless requested. Preserve existing servers and tracking data.

## Find an existing server

1. Determine the project root and the absolute path to its `mlruns` directory. A relative backend URI belongs to the process's working directory, not necessarily the current shell's directory.
2. Look for local MLflow UI or tracking-server processes and their listening ports, including servers started on ports other than 5001. On macOS, `ps` and `lsof -nP -iTCP -sTCP:LISTEN` can help; correlate listener PIDs with MLflow processes and their children, and inspect the command line and working directory as needed. Do not equate an occupied port or a generic HTTP response with MLflow.
3. Verify that a candidate is responding and, where possible, that its backend is this project's `mlruns`. For example, send a read-only `POST` to `/api/2.0/mlflow/experiments/search` with `{"max_results": 1}` and check for a successful MLflow-shaped JSON response. Combine the response with process and backend evidence; do not assume the API reveals the backend path. If verified, return its URL and say it was already running. If several matching servers exist, return their URLs rather than starting another.
4. If a running MLflow server uses a different store, do not present it as this project's server. If its backend or identity cannot be determined and it could be this project's server, report the uncertainty and ask before starting a possible duplicate. Do not stop, restart, or reconfigure any existing process.

## Start a server when needed

1. Prefer port 5001. If it is in use, select another available local TCP port, and do not terminate the process using it. Recheck at startup: another process can claim a port after the availability check.
2. From the project root, launch the server in the background, bound to localhost, with the project store:

   ```sh
   uv run --no-sync mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port PORT
   ```

   Replace `PORT` with the selected port. `--no-sync` avoids unexpectedly installing or changing dependencies; if the project's environment or MLflow is unavailable, report the error and ask before installing or syncing dependencies. If `mlruns` does not yet exist, explain that starting MLflow may initialize a new, empty store.
3. Retain the process identifier and capture startup output in an appropriate temporary log without overwriting existing files. Use the execution environment's background-process facility or a detached process so the server remains available after the command returns. Do not modify tracked files or delete existing logs.
4. Wait for readiness with a bounded timeout, checking both process status and the MLflow API. If startup fails because the port was taken, choose another free port and retry a bounded number of times. For other failures, or if readiness times out, report the relevant log output and do not claim the server is running. Only stop a process started by this invocation if cleanup is necessary; never kill an unrelated process.
5. Once verified, return `http://127.0.0.1:PORT`, indicate that the server was started, and include the port and log location if useful. The server should remain running for the user.
