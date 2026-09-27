import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

from feature_config import SELECTED_FEATURES


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

TRAIN_PATH = "data/processed/train.csv"

df = pd.read_csv(TRAIN_PATH)

X = df[SELECTED_FEATURES]
y = df["Churn"]


# ============================================================
# 2. FEATURE TYPES
# ============================================================

categorical_features = [
    "gender",
    "Dependents",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


# ============================================================
# 3. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# 4. XGBOOST WITH BEST PARAMETERS
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)


# ============================================================
# 5. OUT-OF-FOLD PREDICTIONS
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("=" * 75)
print("GENERATING XGBOOST OOF PREDICTIONS")
print("=" * 75)

probabilities = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]


# ============================================================
# 6. TEST THRESHOLDS
# ============================================================

thresholds = [
    0.25,
    0.26,
    0.27,
    0.28,
    0.29,
    0.30,
    0.31,
    0.32,
    0.33,
    0.34,
    0.35,
    0.36,
    0.37,
    0.38,
    0.39,
    0.40,
    0.41,
    0.42,
    0.43,
    0.44,
    0.45,
]

results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
    ).ravel()

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn,
    })


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 95)
print("XGBOOST FALSE-POSITIVE OPTIMIZATION")
print("=" * 95)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("=" * 95)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "reports/xgb_fp_optimization.csv",
    index=False,
)

print("\nResults saved to:")
print("reports/xgb_fp_optimization.csv")