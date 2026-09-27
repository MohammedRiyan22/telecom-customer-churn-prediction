import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# 1. PATH
# ============================================================

TRAIN_PATH = "data/processed/train.csv"


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(TRAIN_PATH)

print("\n" + "=" * 60)
print("1. TRAINING DATA")
print("=" * 60)

print(f"Shape: {df.shape}")


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]


# ============================================================
# 4. DEFINE FEATURE TYPES
# ============================================================

categorical_features = [
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
    "PaymentMethod"
]

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_encoded = X.copy()

X_encoded[categorical_features] = encoder.fit_transform(
    X_encoded[categorical_features]
)


# ============================================================
# 6. DEFINE WHICH FEATURES ARE DISCRETE
# ============================================================

discrete_features = [
    column in categorical_features
    for column in X_encoded.columns
]


# ============================================================
# 7. CALCULATE MUTUAL INFORMATION
# ============================================================

mi_scores = mutual_info_classif(
    X_encoded,
    y,
    discrete_features=discrete_features,
    random_state=42
)


# ============================================================
# 8. CREATE FEATURE RANKING
# ============================================================

mi_results = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Mutual Information": mi_scores
})

mi_results = mi_results.sort_values(
    by="Mutual Information",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("2. MUTUAL INFORMATION FEATURE RANKING")
print("=" * 60)

print(
    mi_results.to_string(
        index=False,
        formatters={
            "Mutual Information": "{:.4f}".format
        }
    )
)


# ============================================================
# 10. LOW-INFORMATION FEATURES
# ============================================================

print("\n" + "=" * 60)
print("3. LOWEST MUTUAL INFORMATION FEATURES")
print("=" * 60)

print(
    mi_results.tail(5).to_string(
        index=False,
        formatters={
            "Mutual Information": "{:.4f}".format
        }
    )
)

# ============================================================
# 11. CATBOOST MODEL-BASED FEATURE IMPORTANCE
# ============================================================

from catboost import CatBoostClassifier


print("\n" + "=" * 60)
print("4. CATBOOST FEATURE IMPORTANCE")
print("=" * 60)


# Create a fresh copy of the original training data
X_catboost = X.copy()

# CatBoost requires categorical column positions
categorical_indices = [
    X_catboost.columns.get_loc(column)
    for column in categorical_features
]


# Create model
catboost_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False
)


# Train using training data only
catboost_model.fit(
    X_catboost,
    y,
    cat_features=categorical_indices
)


# Get feature importance
catboost_importance = catboost_model.get_feature_importance()


# Create ranking
catboost_results = pd.DataFrame({
    "Feature": X_catboost.columns,
    "Importance": catboost_importance
})

catboost_results = catboost_results.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# Display ranking
print(
    catboost_results.to_string(
        index=False,
        formatters={
            "Importance": "{:.4f}".format
        }
    )
)

# ============================================================
# 12. PERMUTATION IMPORTANCE
# ============================================================

from sklearn.inspection import permutation_importance


print("\n" + "=" * 60)
print("5. PERMUTATION IMPORTANCE")
print("=" * 60)


# Calculate permutation importance on the training data
permutation = permutation_importance(
    catboost_model,
    X_catboost,
    y,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


# Create ranking
permutation_results = pd.DataFrame({
    "Feature": X_catboost.columns,
    "Importance Mean": permutation.importances_mean,
    "Importance Std": permutation.importances_std
})

permutation_results = permutation_results.sort_values(
    by="Importance Mean",
    ascending=False
).reset_index(drop=True)


# Display ranking
print(
    permutation_results.to_string(
        index=False,
        formatters={
            "Importance Mean": "{:.4f}".format,
            "Importance Std": "{:.4f}".format
        }
    )
)

# ============================================================
# 13. VALIDATION-BASED FEATURE SET COMPARISON
# ============================================================

from sklearn.model_selection import StratifiedKFold, cross_validate


print("\n" + "=" * 60)
print("6. VALIDATION-BASED FEATURE SET COMPARISON")
print("=" * 60)


# ------------------------------------------------------------
# Feature sets
# ------------------------------------------------------------

all_features = [
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

strong_consensus_features = [
    "Contract",
    "tenure",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    "OnlineSecurity",
    "PaymentMethod",
    "TechSupport"
]

reduced_features = [
    "Contract",
    "tenure",
    "InternetService",
    "MonthlyCharges",
    "OnlineSecurity",
    "PaymentMethod",
    "TechSupport"
]


feature_sets = {
    "All 19 features": all_features,
    "Strong consensus": strong_consensus_features,
    "Reduced core": reduced_features
}


# ------------------------------------------------------------
# Cross-validation setup
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# Evaluate each feature set
# ------------------------------------------------------------

validation_results = []


for set_name, features in feature_sets.items():

    X_subset = X[features].copy()

    categorical_subset = [
        feature for feature in features
        if feature in categorical_features
    ]

    categorical_indices_subset = [
        X_subset.columns.get_loc(feature)
        for feature in categorical_subset
    ]

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    )

    scores = cross_validate(
        model,
        X_subset,
        y,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1"
        },
        return_train_score=False,
        n_jobs=1,
        params={
            "cat_features": categorical_indices_subset
        }
    )

    validation_results.append({
        "Feature Set": set_name,
        "Features": len(features),
        "Mean ROC-AUC": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std(),
        "Mean F1": scores["test_f1"].mean(),
        "F1 Std": scores["test_f1"].std()
    })


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

results_df = pd.DataFrame(validation_results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Mean ROC-AUC": "{:.4f}".format,
            "ROC-AUC Std": "{:.4f}".format,
            "Mean F1": "{:.4f}".format,
            "F1 Std": "{:.4f}".format
        }
    )
)

# ============================================================
# 14. INCREMENTAL FEATURE SELECTION
# ============================================================

print("\n" + "=" * 60)
print("7. INCREMENTAL FEATURE SELECTION")
print("=" * 60)


# ------------------------------------------------------------
# Use permutation-importance ranking
# ------------------------------------------------------------

ranked_features = permutation_results["Feature"].tolist()


# Feature counts to test
feature_counts = [5, 7, 9, 11, 13, 15, 17, 19]


incremental_results = []


# ------------------------------------------------------------
# Evaluate progressively larger feature sets
# ------------------------------------------------------------

for count in feature_counts:

    selected_features = ranked_features[:count]

    X_subset = X[selected_features].copy()

    categorical_subset = [
        feature
        for feature in selected_features
        if feature in categorical_features
    ]

    categorical_indices_subset = [
        X_subset.columns.get_loc(feature)
        for feature in categorical_subset
    ]

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    )

    scores = cross_validate(
        model,
        X_subset,
        y,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1"
        },
        return_train_score=False,
        n_jobs=1,
        params={
            "cat_features": categorical_indices_subset
        }
    )

    incremental_results.append({
        "Features": count,
        "Mean ROC-AUC": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std(),
        "Mean F1": scores["test_f1"].mean(),
        "F1 Std": scores["test_f1"].std(),
        "Feature List": ", ".join(selected_features)
    })


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

incremental_df = pd.DataFrame(incremental_results)

print(
    incremental_df[
        [
            "Features",
            "Mean ROC-AUC",
            "ROC-AUC Std",
            "Mean F1",
            "F1 Std"
        ]
    ].to_string(
        index=False,
        formatters={
            "Mean ROC-AUC": "{:.4f}".format,
            "ROC-AUC Std": "{:.4f}".format,
            "Mean F1": "{:.4f}".format,
            "F1 Std": "{:.4f}".format
        }
    )
)


# ------------------------------------------------------------
# Display feature sets
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("8. FEATURES USED AT EACH LEVEL")
print("=" * 60)

for _, row in incremental_df.iterrows():

    print(f"\nTop {int(row['Features'])} features:")
    print(row["Feature List"])