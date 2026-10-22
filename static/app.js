const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#file-input");
const fileChip = document.querySelector("#file-chip");
const fileName = document.querySelector("#file-name");
const removeFileButton = document.querySelector("#remove-file");
const predictButton = document.querySelector("#predict-button");
const errorMessage = document.querySelector("#error-message");
let selectedFile = null;

function setFile(file) {
  selectedFile = file;
  fileName.textContent = file ? file.name : "";
  fileChip.hidden = !file;
  predictButton.disabled = !file;
  errorMessage.hidden = true;
  dropzone.classList.toggle("has-file", Boolean(file));
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

dropzone.addEventListener("click", (event) => {
  if (event.target !== removeFileButton) fileInput.click();
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0] || null));
removeFileButton.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.value = "";
  setFile(null);
});
for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file && file.name.toLowerCase().endsWith(".csv")) setFile(file);
  else showError("Please choose a CSV file.");
});

function parseCsvLine(line) {
  return line.split(",").map((cell) => cell.trim().replace(/^"(.*)"$/, "$1"));
}

function parseSensorCsv(source, expectedCount) {
  const lines = source.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length) throw new Error("The CSV is empty.");
  const firstRow = parseCsvLine(lines[0]);
  const firstRowIsData = firstRow.every((cell) => cell === "" || Number.isFinite(Number(cell)));
  const dataLines = firstRowIsData ? lines : lines.slice(1);
  if (!dataLines.length) throw new Error("The CSV has a header but no sensor data rows.");
  return dataLines.map((line, rowIndex) => {
    const row = parseCsvLine(line);
    if (row.length !== expectedCount) {
      throw new Error(`Record ${rowIndex + 1} has ${row.length} columns. The model expects ${expectedCount} sensor readings in the original order.`);
    }
    return row.map((cell, columnIndex) => {
      if (cell === "") return null;
      const value = Number(cell);
      if (!Number.isFinite(value)) throw new Error(`Record ${rowIndex + 1}, sensor ${columnIndex} is not a valid number.`);
      return value;
    });
  });
}

function renderResult(batch) {
  document.querySelector("#empty-result").hidden = true;
  document.querySelector("#result-content").hidden = false;
  const results = batch.results;
  const failCount = batch.predicted_fail_count;
  const warningCount = batch.predicted_warning_count;
  const passCount = batch.predicted_pass_count;
  document.querySelector("#result-status").textContent = `${batch.record_count} RECORD${batch.record_count === 1 ? "" : "S"} CHECKED`;
  const overallStatus = failCount ? "fail" : warningCount ? "warning" : "success";
  document.querySelector("#result-status").className = `status-pill ${overallStatus}`;
  document.querySelector("#batch-summary").textContent = `${batch.record_count} record${batch.record_count === 1 ? "" : "s"} processed · ${passCount} Pass · ${warningCount} Warning · ${failCount} Fail`;
  const tableBody = document.querySelector("#batch-results");
  tableBody.replaceChildren();
  results.forEach((result, index) => {
    const row = document.createElement("tr");
    const values = [
      `Record ${index + 1}`,
      result.prediction,
      `${(result.model_confidence * 100).toFixed(2)}% (${result.most_likely_class} score)`,
      `${(result.pass_probability * 100).toFixed(2)}%`,
      `${(result.defect_probability * 100).toFixed(2)}%`,
    ];
    values.forEach((value, columnIndex) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (columnIndex === 1) cell.className = `prediction-cell ${result.prediction.toLowerCase()}`;
      row.append(cell);
    });
    tableBody.append(row);
  });
  const sensorList = document.querySelector("#sensor-list");
  sensorList.replaceChildren();
  const topFeatures = batch.top_influential_features || [];
  const maximum = Math.max(...topFeatures.map((item) => item.importance_percent), 1);
  for (const feature of topFeatures) {
    const row = document.createElement("div");
    row.className = "sensor-row";
    const name = document.createElement("span");
    name.className = "sensor-name";
    name.textContent = feature.feature;
    const bar = document.createElement("div");
    bar.className = "sensor-bar";
    const fill = document.createElement("span");
    fill.style.width = `${(feature.importance_percent / maximum) * 100}%`;
    bar.append(fill);
    const percent = document.createElement("span");
    percent.className = "sensor-percent";
    percent.textContent = `${feature.importance_percent.toFixed(2)}%`;
    row.append(name, bar, percent);
    sensorList.append(row);
  }
  document.querySelector("#result-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  predictButton.disabled = true;
  predictButton.classList.add("loading");
  predictButton.querySelector("span:first-child").textContent = "Checking uploaded records…";
  errorMessage.hidden = true;
  try {
    const [source, infoResponse] = await Promise.all([
      selectedFile.text(),
      fetch("/api/model-info"),
    ]);
    if (!infoResponse.ok) throw new Error("Could not read model information.");
    const info = await infoResponse.json();
    const records = parseSensorCsv(source, info.input_feature_count);
    const response = await fetch("/api/predict-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ records }),
    });
    const batch = await response.json();
    if (!response.ok) throw new Error(batch.detail || "Prediction request failed.");
    renderResult(batch);
  } catch (error) {
    showError(error.message || "Something went wrong while checking this file.");
  } finally {
    predictButton.disabled = !selectedFile;
    predictButton.classList.remove("loading");
    predictButton.querySelector("span:first-child").textContent = "Run quality checks";
  }
});

fetch("/api/model-info")
  .then((response) => response.json())
  .then((info) => {
    document.querySelector("#model-detail").textContent = `${info.selected_feature_count} active sensors · ${info.training_rows.toLocaleString()} training records`;
  })
  .catch(() => {
    document.querySelector("#model-detail").textContent = "Model information unavailable";
  });
