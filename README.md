# Product Quality Check

A small full-stack demo for screening SECOM manufacturing records from the original 590 sensor readings. The app accepts a CSV containing one or more records, applies the model's saved median imputation and scaling, and returns a **Pass**, **Warning**, or **Fail** status, the Pass and Fail scores for every row, and the top ten globally influential features.

This POC uses public SECOM sample data and has not been validated on a production line. Use its results as review signals and keep established inspections in place.

All application code, copied SECOM inputs, training outputs, and the model bundle live in this folder. The app does not read or write files outside it at runtime.

## Start the app

### Docker with NGINX (recommended)

Install and start Docker Desktop, open a terminal in this folder, and run:

```bash
docker compose up --build
```

Open <http://localhost:8010>. NGINX is the only published service; it forwards browser and API requests to FastAPI over Compose's private network. Stop the application with `Ctrl+C`, then run `docker compose down` if it was started in the background.

To start it in the background, use `docker compose up --build -d`. View logs with `docker compose logs -f`; stop it with `docker compose down`.

The existing `secom_quality_model.pkl` is copied into the app image, so the Docker startup does not retrain the model. Rebuild the image with `docker compose up --build` after changing the model or app code.

### Run directly with Python

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at <http://127.0.0.1:8000/docs>. Run `python train_model.py` only when you intend to retrain the model; it updates the local model bundle and metrics.

The training script reports a stratified 80/20 holdout and repeated stratified 5-fold cross-validation (5 repeats, 25 folds) within the 80% training split. In every fold, median imputation, constant-feature filtering, scaling, and SMOTE are fitted using only that fold's training rows. For an honest demo, `secom_quality_model.pkl` is fitted only on the 80% training split, while the untouched 20% holdout is saved under `demo_test_data/`. Both evaluation summaries are written to `training_metrics.json`.

The latest leakage-aware holdout run detected only 1 of 21 failures (4.8% failure recall) using the model's default 50% decision threshold. Repeated cross-validation gives a more stable estimate; review its mean and standard deviation in `training_metrics.json`. The application applies an additional screening rule described below. Treat this as a prototype, not a validated production quality-control system.

## Input format

Upload a CSV with one or more data rows and 590 sensor columns, in the same order as the original SECOM dataset. A header row is optional. Download the CSV template from the app. Blank cells are imputed with the corresponding training-set median. The `demo_test_data/examples/` folder contains held-out examples for a detected failure, a missed failure, and a correctly predicted pass. The `pass_and_fail_batch.csv` file combines the correctly predicted pass and detected failure for demonstrating batch upload. Actual labels and individual example predictions are listed in `example_labels.csv`.

The browser sends records to `POST /api/predict-batch`; `POST /api/predict` remains available for a single record. Batch summaries count Pass, Warning, and Fail outcomes. These are model predictions, not verified defect counts, because actual labels are not provided with live uploads.

The full held-out readings are in `demo_test_data/secom_test_features.data` (space-separated, same format as the source file). Corresponding actual labels and original source row numbers are in `demo_test_data/secom_test_labels.csv`; the row ordering matches the feature file.

The SECOM labels are mapped as `1 → Fail` and `-1 → Pass`. The app assigns status using the model's Fail score: **50% or higher → Fail**, **25% to below 50% → Warning**, and **below 25% → Pass**. Warning and Fail are signals to review or inspect records under the site's established process. The displayed model confidence is the higher of the underlying Pass and Fail scores; a Warning is a status band, not a third model probability. This threshold policy was explored on the held-out sample, so its observed results are not an independent validation. Probabilities and feature importances reflect the limitations of this small, imbalanced reference dataset.

## Project files

- `app.py` — FastAPI backend and prediction endpoints.
- `static/` — browser interface and CSV upload behavior.
- `train_model.py` — leakage-aware training and model packaging.
- `secom.data`, `secom_labels.data`, `secom.names` — local copies of the SECOM source data and description.
- `secom_quality_model.pkl` — trained model bundle.
- `training_metrics.json` — holdout evaluation results from the last training run.
