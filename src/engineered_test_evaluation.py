import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

from catboost import CatBoostClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "data/processed/cleaned_churn.csv"

BASE_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

ENGINEERED_FEATURES = [
    "TenureGroup",
    "MonthlyChargeGroup"
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
    "MonthlyChargeGroup"
]

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# Threshold selected from training-side validation,
# not optimized on the untouched test set.
THRESHOLD = 0.50


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_engineered_features(df):
    df = df.copy()

    # --------------------------------------------------------
    # Tenure groups
    # --------------------------------------------------------

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 36, 48, 60, 72],
        labels=[
            "0-6",
            "7-12",
            "13-24",
            "25-36",
            "37-48",
            "49-60",
            "61-72"
        ]
    ).astype(str)

    # --------------------------------------------------------
    # Monthly charge groups
    # --------------------------------------------------------

    df["MonthlyChargeGroup"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-np.inf, 30, 50, 70, 90, 120, np.inf],
        labels=[
            "<=30",
            "31-50",
            "51-70",
            "71-90",
            "91-120",
            ">120"
        ]
    ).astype(str)

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("ENGINEERED FEATURE TEST EVALUATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

df = create_engineered_features(df)

X = df[FEATURES].copy()
y = df["Churn"].astype(int).copy()


# ============================================================
# ORIGINAL TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nOriginal split:")
print(f"  Training rows: {len(X_train)}")
print(f"  Test rows:     {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# BALANCE TRAINING DATA ONLY
# ============================================================

ros = RandomOverSampler(
    sampling_strategy="auto",
    random_state=RANDOM_STATE
)

X_train_balanced, y_train_balanced = ros.fit_resample(
    X_train,
    y_train
)

print("\nBalanced training distribution:")
print(y_train_balanced.value_counts().sort_index())


# ============================================================
# CATBOOST MODEL
# ============================================================

catboost_indices = [
    FEATURES.index(feature)
    for feature in CATEGORICAL_FEATURES
]

model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.03,
    depth=5,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining CatBoost...")

model.fit(
    X_train_balanced,
    y_train_balanced,
    cat_features=catboost_indices
)

print("Training complete.")


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

brier = brier_score_loss(
    y_test,
    probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(f"\nThreshold: {THRESHOLD:.2f}")

print(f"\nAccuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score : {f1:.4f}")

print(f"\nROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
print(f"Brier    : {brier:.4f}")

print("\nConfusion Matrix:")
print(f"  TN: {tn}")
print(f"  FP: {fp}")
print(f"  FN: {fn}")
print(f"  TP: {tp}")


# ============================================================
# PROBABILITY SUMMARY
# ============================================================

print("\nProbability summary:")

print(
    f"  Mean   : {probabilities.mean():.4f}"
)

print(
    f"  Median : {np.median(probabilities):.4f}"
)

print(
    f"  Min    : {probabilities.min():.4f}"
)

print(
    f"  Max    : {probabilities.max():.4f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.get_feature_importance()
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 80)
print("TOP FEATURE IMPORTANCES")
print("=" * 80)

print(
    importance.head(15).to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "Feature_Set": "Baseline_19 + TenureGroup + MonthlyChargeGroup",
    "Feature_Count": len(FEATURES),
    "Threshold": THRESHOLD,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "Brier": brier,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}])

results.to_csv(
    "reports/engineered_test_evaluation.csv",
    index=False
)

print("\nResults saved:")
print("  reports/engineered_test_evaluation.csv")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)