"""Validate the production bundle, metadata, and held-out evaluation."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "customer_churn_production_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
METRIC_TOLERANCE = 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    bundle = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    require(isinstance(bundle, dict), "Model artifact must be a bundle dictionary.")
    for key in ("model", "features", "categorical_features", "numeric_features", "threshold"):
        require(key in bundle, f"Model bundle is missing '{key}'.")

    model = bundle["model"]
    features = list(bundle["features"])
    threshold = float(bundle["threshold"])
    require(type(model).__name__ == metadata["model_type"], "Model type does not match metadata.")
    require(features == metadata["selected_features"], "Feature order does not match metadata.")
    require(bundle["categorical_features"] == metadata["categorical_features"], "Categorical features do not match metadata.")
    require(bundle["numeric_features"] == metadata["numeric_features"], "Numeric features do not match metadata.")
    require(abs(threshold - float(metadata["threshold"])) <= 1e-9, "Decision threshold does not match metadata.")

    actual_parameters = model.get_params()
    for name, expected in metadata["parameters"].items():
        require(actual_parameters.get(name) == expected, f"Model parameter '{name}' does not match metadata.")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    require(set(features).issubset(train.columns), "Training data is missing model features.")
    require(set(features + ["Churn"]).issubset(test.columns), "Test data is missing model features or target.")
    require(len(test) == metadata["test_set"]["samples"], "Test row count does not match metadata.")

    X_test = test.loc[:, features]
    y_test = test["Churn"].astype(int)
    probabilities = model.predict_proba(X_test)[:, 1]
    require(len(probabilities) == len(test), "Prediction count does not match test rows.")
    require(bool(((probabilities >= 0) & (probabilities <= 1)).all()), "Model returned an invalid probability.")

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    actual_metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "brier": brier_score_loss(y_test, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    expected_metrics = metadata["final_test_metrics"]
    for name, actual in actual_metrics.items():
        expected = expected_metrics[name]
        require(abs(float(actual) - float(expected)) <= METRIC_TOLERANCE,
                f"Held-out metric '{name}' differs from metadata: {actual} != {expected}.")

    overlap_columns = features + ["Churn"]
    overlap_count = len(
        test[overlap_columns].astype(str).drop_duplicates().merge(
            train[overlap_columns].astype(str).drop_duplicates(),
            on=overlap_columns,
            how="inner",
        )
    )
    require(overlap_count == metadata["test_set"]["exact_profile_overlap_count"],
            "Exact profile overlap count differs from metadata.")

    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    print("PRODUCTION MODEL VERIFICATION PASSED")
    print(f"Artifact: {MODEL_PATH.name} (SHA-256 {digest})")
    print(f"Model: {type(model).__name__}; {len(features)} ordered features; threshold {threshold:.4f}")
    print(f"Held-out rows: {len(test)}; training artifact rows: {len(train)}")
    print("Metrics:")
    for name, value in actual_metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {name}: {formatted}")
    if overlap_count:
        print(f"Note: {overlap_count} feature-and-target profiles appear in both processed split files.")
    print("A sample prediction:")
    print(f"  probability={probabilities[0]:.6f}; prediction={int(predictions[0])}; actual={int(y_test.iloc[0])}")


if __name__ == "__main__":
    main()
