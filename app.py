"""FastAPI backend and static frontend for SECOM product quality checks."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MODEL_PATH = ROOT / "secom_quality_model.pkl"

app = FastAPI(
    title="Product Quality Check",
    description="Predict SECOM manufacturing pass/fail outcomes from sensor readings.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionRequest(BaseModel):
    features: list[Optional[float]] = Field(
        ..., description="One production record containing all 590 sensor readings."
    )


class BatchPredictionRequest(BaseModel):
    records: list[list[Optional[float]]] = Field(
        ..., description="Production records, each containing all 590 sensor readings."
    )


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file is missing at {MODEL_PATH}. Run `python train_model.py` first."
        )
    bundle = joblib.load(MODEL_PATH)
    required = {"model", "scaler", "median", "feature_indices", "input_feature_count"}
    missing = required.difference(bundle)
    if missing:
        raise RuntimeError(f"Model bundle is missing fields: {', '.join(sorted(missing))}")
    return bundle


BUNDLE = load_bundle()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True}


@app.get("/api/model-info")
def model_info() -> dict:
    metrics_path = ROOT / "training_metrics.json"
    failure_recall = None
    cv_failure_recall = None
    cv_failure_recall_std = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        failure_recall = metrics.get("classification_report", {}).get("fail", {}).get("recall")
        cv_summary = metrics.get("repeated_stratified_cv", {})
        cv_failure_recall = cv_summary.get("mean", {}).get("fail_recall")
        cv_failure_recall_std = cv_summary.get("standard_deviation", {}).get("fail_recall")
    return {
        "input_feature_count": BUNDLE["input_feature_count"],
        "selected_feature_count": len(BUNDLE["feature_indices"]),
        "training_rows": BUNDLE.get("training_rows"),
        "holdout_failure_recall": failure_recall,
        "cv_failure_recall_mean": cv_failure_recall,
        "cv_failure_recall_std": cv_failure_recall_std,
        "model": "Random Forest classifier",
    }


@app.get("/api/template.csv")
def csv_template() -> StreamingResponse:
    output = io.StringIO()
    writer = csv.writer(output)
    count = int(BUNDLE["input_feature_count"])
    writer.writerow([f"Sensor_{i}" for i in range(count)])
    writer.writerow([""] * count)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="sensor_readings_template.csv"'},
    )


def predict_records(records: list[list[Optional[float]]]) -> list[dict]:
    expected = int(BUNDLE["input_feature_count"])
    if not records:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one production record.",
        )
    for index, record in enumerate(records, start=1):
        if len(record) != expected:
            raise HTTPException(
                status_code=422,
                detail=f"Record {index} must contain {expected} sensor readings; received {len(record)}.",
            )
    try:
        values = np.asarray(
            [
                [np.nan if value is None else float(value) for value in record]
                for record in records
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Sensor readings must be numbers or blank.")
    if values.ndim != 2 or values.shape[1] != expected:
        received = values.shape[1] if values.ndim == 2 else 0
        raise HTTPException(
            status_code=422,
            detail=f"Each record must contain {expected} sensor readings; received {received}.",
        )
    if np.isinf(values).any():
        raise HTTPException(status_code=422, detail="Sensor readings cannot be infinite.")

    medians = np.asarray(BUNDLE["median"], dtype=float)
    missing = np.isnan(values)
    values[missing] = np.broadcast_to(medians, values.shape)[missing]
    selected = values[:, np.asarray(BUNDLE["feature_indices"], dtype=int)]
    scaled = BUNDLE["scaler"].transform(selected)
    model = BUNDLE["model"]
    predicted_classes = model.predict(scaled).astype(int)
    class_to_column = {int(label): index for index, label in enumerate(model.classes_)}
    probabilities = model.predict_proba(scaled)

    importances = getattr(model, "feature_importances_", None)
    top_features = []
    if importances is not None:
        indices = np.argsort(importances)[-10:][::-1]
        original_indices = BUNDLE["feature_indices"]
        top_features = [
            {
                "feature": f"Feature_{int(original_indices[index])}",
                "importance_percent": round(float(importances[index]) * 100, 2),
            }
            for index in indices
        ]

    results = []
    for predicted_class, row_probabilities in zip(predicted_classes, probabilities):
        fail_probability = float(row_probabilities[class_to_column[0]])
        pass_probability = float(row_probabilities[class_to_column[1]])
        confidence = float(row_probabilities[class_to_column[int(predicted_class)]])
        results.append({
            "prediction": "Fail" if predicted_class == 0 else "Pass",
            "model_confidence": round(confidence, 6),
            "defect_probability": round(fail_probability, 6),
            "pass_probability": round(pass_probability, 6),
            "top_influential_features": top_features,
            "note": "Class probabilities and global feature importance are model estimates based on the SECOM training data.",
        })
    return results


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    return predict_records([request.features])[0]


@app.post("/api/predict-batch")
def predict_batch(request: BatchPredictionRequest) -> dict:
    results = predict_records(request.records)
    failed = sum(result["prediction"] == "Fail" for result in results)
    top_features = results[0]["top_influential_features"]
    for result in results:
        result.pop("top_influential_features", None)
    return {
        "record_count": len(results),
        "predicted_fail_count": failed,
        "predicted_pass_count": len(results) - failed,
        "top_influential_features": top_features,
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
