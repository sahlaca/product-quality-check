# Product Quality Check — Technical Overview

## Project summary

This proof of concept (POC) screens manufacturing records from the public SECOM sample dataset and assigns **Pass**, **Warning**, or **Fail** statuses from 590 in-process sensor readings. It demonstrates a CSV workflow for applying a trained classifier to one record or a batch. It has not been validated on a production line and does not establish production readiness or guarantee defect prevention.

## Application architecture

```text
User's browser
    │  CSV upload / results
    ▼
NGINX gateway (localhost:8010)
    │  forwards requests inside Docker network
    ▼
FastAPI application
    │  validates 590 numeric sensor values per record
    │  imputes missing values, selects trained features, scales values
    ▼
Saved Random Forest model bundle
    │
    └── Pass/Warning/Fail screening status, class scores, global feature ranking
```

The frontend is served by FastAPI. NGINX is the only service exposed on the host; Docker Compose connects it to the application container. The current integration accepts CSV uploads. A future manufacturing integration would need a source adapter to read the equipment or data-platform format, map sensor names and order, handle units and missing readings, then submit the resulting record to the prediction API.

## Data and model

- **Reference data:** SECOM, with 1,567 records and 590 sensor features; the outcome classes are imbalanced.
- **Label mapping:** source label `1` is **Fail**; source label `-1` is **Pass**.
- **Preprocessing:** missing values are filled using training-set medians; constant features are removed; remaining values are standardized. SMOTE is applied to training rows to address class imbalance.
- **Classifier:** Random Forest. The packaged model is trained on the 80% training split. A separate 20% holdout is kept for evaluation and demo examples.
- **Output:** Pass, Warning, or Fail screening status; the underlying Pass and Fail scores; model confidence (the higher of those two scores); and the model’s globally top-ranked features. The importances describe the fitted model overall; they do not explain why a particular record received its prediction.

## Application screening rule

The app maps the model's Fail score to a review status: **50% or higher → Fail**, **25% to below 50% → Warning**, and **below 25% → Pass**. Warning and Fail can help direct attention to records for review under the site's established inspection process. The model itself still produces only Pass and Fail scores; Warning is a threshold band, not a third model class. The rule changes the app's status display, not the trained model. The displayed confidence is the higher of the Pass and Fail scores, with its class shown explicitly.

## Evaluation and current limitation

Evaluation uses a stratified 80/20 holdout and repeated stratified 5-fold cross-validation (5 repeats) on the training portion. Preprocessing and SMOTE are fitted within each training fold to reduce evaluation leakage.

On the held-out 314 records, there were 21 actual failures. The model detected **1 of 21** (failure recall **4.8%**) and missed 20. Repeated cross-validation estimated mean failure recall of **2.2%** (standard deviation **3.8 percentage points**). The holdout ROC-AUC for Fail was **0.714**, which measures ranking across thresholds; it does not mean that the default Pass/Fail decisions catch failures reliably. High overall accuracy is largely driven by the much more common Pass class.

For context, applying the app's 25% Warning / 50% Fail rule to those same holdout records resulted in 271 Pass, 41 Warning, and 2 Fail statuses. Of 21 actual failures, 9 fell into Warning or Fail and 12 fell into Pass. Of 293 actual passes, 34 fell into Warning or Fail. This is a retrospective result on the same holdout used while considering the threshold, not an independent validation of the policy; the warning band can increase the number of records routed for review and can also create more false alarms.

SMOTE is already included in model training and cross-validation folds. It has not produced adequate failure detection with this dataset and model. Model scores are estimates, not guarantees or calibrated measurements of real-world risk. Treat Fail as a prompt for review and retain the approved inspection process.

## Challenges and next steps

1. **Few failures:** the dataset contains relatively few Fail examples, limiting what the model can learn and how precisely recall can be measured.
2. **Missed failures:** compare imbalance strategies, class weights, algorithms, feature handling, and decision thresholds using recall, precision, PR-AUC, and the cost of false alarms. Choose a threshold on validation data and report the resulting trade-off.
3. **Transfer to a real production line:** confirm sensor meaning, units, process context, label quality, and whether SECOM resembles the target equipment. Re-evaluate on representative, time-separated production data before operational use.
4. **Live integration:** agree on the source-system schema, record identifiers, data quality rules, API authentication, monitoring, and human review workflow.

## Suggested demo framing

Suggested framing: “This demo uses public SECOM sample data to show how manufacturing sensor records can flow through a quality-screening application. It has not been validated on a production line.” If discussing model performance, explain that the default model decision detected 1 of 21 holdout failures, while the exploratory Warning band surfaced 9 of 21 for review on that same holdout. The provided examples are selected records from the held-out set. Keep established inspections in place and validate any future threshold policy with representative production data before relying on it for quality-control decisions.
