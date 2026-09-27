import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

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
# 4. FINAL XGBOOST MODEL
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
# 5. TRAIN MODEL
# ============================================================

print("=" * 75)
print("TRAINING FINAL XGBOOST MODEL FOR INTERPRETATION")
print("=" * 75)

model.fit(X, y)


# ============================================================
# 6. GET FEATURE NAMES AFTER PREPROCESSING
# ============================================================

feature_names = (
    model
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

xgb_model = model.named_steps["model"]


# ============================================================
# 7. BUILT-IN XGBOOST FEATURE IMPORTANCE
# ============================================================

importance_values = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values,
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False,
)

print("\n")
print("=" * 75)
print("XGBOOST BUILT-IN FEATURE IMPORTANCE")
print("=" * 75)

print(
    importance_df.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# 8. PERMUTATION IMPORTANCE
# ============================================================

print("\n")
print("=" * 75)
print("CALCULATING PERMUTATION IMPORTANCE")
print("=" * 75)

permutation = permutation_importance(
    model,
    X,
    y,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=1,
)


permutation_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance_Mean": permutation.importances_mean,
    "Importance_Std": permutation.importances_std,
})

permutation_df = permutation_df.sort_values(
    by="Importance_Mean",
    ascending=False,
)


print("\n")
print("=" * 75)
print("PERMUTATION IMPORTANCE")
print("=" * 75)

print(
    permutation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

importance_df.to_csv(
    "reports/xgb_feature_importance.csv",
    index=False,
)

permutation_df.to_csv(
    "reports/xgb_permutation_importance.csv",
    index=False,
)

print("\n")
print("Saved:")
print("reports/xgb_feature_importance.csv")
print("reports/xgb_permutation_importance.csv")