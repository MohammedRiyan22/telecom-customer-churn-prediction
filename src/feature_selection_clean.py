from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_churn.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "feature_selection_clean.csv"
)


# ============================================================
# FEATURES
# ============================================================

ALL_FEATURES = [
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
    "TotalCharges",
]

TARGET = "Churn"


# ============================================================
# CURRENT FEATURE SET
# ============================================================

CURRENT_15 = [
    "Contract",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
    "OnlineBackup",
    "MultipleLines",
    "gender",
    "Dependents",
    "StreamingMovies",
    "DeviceProtection",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("LEAKAGE-SAFE FEATURE SELECTION")
print("=" * 100)

df = pd.read_csv(DATA_PATH)

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

print("\nDataset:")
print(f"Rows    : {len(df)}")
print(f"Features: {len(ALL_FEATURES)}")

print("\nOriginal target distribution:")
print(y.value_counts().sort_index())

print("\nOriginal target percentage:")
print(
    (
        y.value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


# ============================================================
# 80/20 TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print("\n")
print("=" * 100)
print("TRAIN / TEST SPLIT")
print("=" * 100)

print(
    f"\nTraining rows: {len(X_train)}"
)

print(
    f"Test rows    : {len(X_test)}"
)

print("\nTraining target:")
print(y_train.value_counts().sort_index())

print("\nTest target:")
print(y_test.value_counts().sort_index())

print(
    "\nIMPORTANT:"
    "\nThe test set will NOT be used for feature selection."
)


# ============================================================
# FEATURE TYPES
# ============================================================

NUMERICAL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    feature
    for feature in ALL_FEATURES
    if feature not in NUMERICAL_FEATURES
]


# ============================================================
# MUTUAL INFORMATION
# ============================================================

print("\n")
print("=" * 100)
print("MUTUAL INFORMATION RANKING")
print("=" * 100)

mi_data = X_train.copy()

# Encode categorical features only for MI calculation.
mi_encoded = pd.DataFrame(index=mi_data.index)

for feature in ALL_FEATURES:

    if feature in NUMERICAL_FEATURES:

        mi_encoded[feature] = pd.to_numeric(
            mi_data[feature]
        )

    else:

        mi_encoded[feature] = (
            mi_data[feature]
            .astype("category")
            .cat.codes
        )

discrete_mask = [
    feature not in NUMERICAL_FEATURES
    for feature in ALL_FEATURES
]

mi_scores = mutual_info_classif(
    mi_encoded,
    y_train,
    discrete_features=discrete_mask,
    random_state=42,
)

mi_df = pd.DataFrame(
    {
        "Feature": ALL_FEATURES,
        "MI": mi_scores,
    }
)

mi_df = mi_df.sort_values(
    "MI",
    ascending=False,
).reset_index(drop=True)

mi_df["MI"] = mi_df["MI"].round(6)

print(
    mi_df.to_string(index=False)
)


# ============================================================
# FEATURE RANKING
# ============================================================

mi_ranking = mi_df["Feature"].tolist()

print("\n")
print("=" * 100)
print("FEATURE RANKING")
print("=" * 100)

for position, feature in enumerate(
    mi_ranking,
    start=1,
):

    score = mi_df.loc[
        mi_df["Feature"] == feature,
        "MI",
    ].iloc[0]

    print(
        f"{position:2d}. "
        f"{feature:<20} "
        f"MI = {score:.6f}"
    )


# ============================================================
# OVERSAMPLING FUNCTION
# ============================================================

def oversample_training_fold(
    X_fold,
    y_fold,
):

    fold_df = X_fold.copy()

    fold_df["_target_"] = (
        y_fold.to_numpy()
    )

    majority = fold_df[
        fold_df["_target_"] == 0
    ]

    minority = fold_df[
        fold_df["_target_"] == 1
    ]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )

    balanced = pd.concat(
        [
            majority,
            minority_upsampled,
        ]
    )

    balanced = balanced.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    y_balanced = balanced.pop(
        "_target_"
    )

    X_balanced = balanced

    return X_balanced, y_balanced


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    features
):

    numerical = [
        feature
        for feature in features
        if feature in NUMERICAL_FEATURES
    ]

    categorical = [
        feature
        for feature in features
        if feature in CATEGORICAL_FEATURES
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# FEATURE SUBSETS
# ============================================================

subset_sizes = [
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
]

feature_sets = {}

for size in subset_sizes:

    feature_sets[
        f"Top_{size}"
    ] = mi_ranking[:size]

# Add current 15-feature configuration
feature_sets[
    "Current_15"
] = CURRENT_15


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


results = []


# ============================================================
# FEATURE SET EVALUATION
# ============================================================

for set_name, features in feature_sets.items():

    print("\n")
    print("=" * 100)

    print(
        f"Evaluating feature set: "
        f"{set_name}"
    )

    print(
        f"Number of features: "
        f"{len(features)}"
    )

    print(
        "Features:"
    )

    print(
        ", ".join(features)
    )

    print("=" * 100)

    fold_results = []

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        cv.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"  Fold {fold}/5",
            end="",
            flush=True,
        )

        X_fold_train = (
            X_train.iloc[train_idx][
                features
            ].copy()
        )

        X_fold_valid = (
            X_train.iloc[valid_idx][
                features
            ].copy()
        )

        y_fold_train = (
            y_train.iloc[train_idx]
        )

        y_fold_valid = (
            y_train.iloc[valid_idx]
        )

        # ----------------------------------------------------
        # BALANCE TRAINING FOLD ONLY
        # ----------------------------------------------------

        (
            X_fold_balanced,
            y_fold_balanced,
        ) = oversample_training_fold(
            X_fold_train,
            y_fold_train,
        )

        # ----------------------------------------------------
        # VERIFY BALANCE
        # ----------------------------------------------------

        if fold == 1:

            print(
                f" | Balanced train: "
                f"No={sum(y_fold_balanced == 0)}, "
                f"Yes={sum(y_fold_balanced == 1)}",
                end="",
            )

        # ----------------------------------------------------
        # PIPELINE
        # ----------------------------------------------------

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    create_preprocessor(
                        features
                    ),
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
            ]
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        pipeline.fit(
            X_fold_balanced,
            y_fold_balanced,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        probabilities = (
            pipeline.predict_proba(
                X_fold_valid
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        fold_result = {

            "ROC-AUC":
                roc_auc_score(
                    y_fold_valid,
                    probabilities,
                ),

            "PR-AUC":
                average_precision_score(
                    y_fold_valid,
                    probabilities,
                ),

            "Precision":
                precision_score(
                    y_fold_valid,
                    predictions,
                    zero_division=0,
                ),

            "Recall":
                recall_score(
                    y_fold_valid,
                    predictions,
                    zero_division=0,
                ),

            "F1":
                f1_score(
                    y_fold_valid,
                    predictions,
                    zero_division=0,
                ),
        }

        fold_results.append(
            fold_result
        )

        print(" ✓")

    # ========================================================
    # MEAN CV RESULTS
    # ========================================================

    mean_results = {

        metric:
            np.mean(
                [
                    fold[metric]
                    for fold in fold_results
                ]
            )

        for metric in fold_results[0]
    }

    results.append(
        {
            "Feature_Set": set_name,
            "Feature_Count": len(features),
            "Features": ", ".join(features),
            "ROC-AUC":
                mean_results["ROC-AUC"],
            "PR-AUC":
                mean_results["PR-AUC"],
            "Precision":
                mean_results["Precision"],
            "Recall":
                mean_results["Recall"],
            "F1":
                mean_results["F1"],
        }
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    [
        "PR-AUC",
        "ROC-AUC",
        "F1",
    ],
    ascending=False,
)


print("\n")
print("=" * 100)
print("FEATURE SET COMPARISON")
print("=" * 100)

display_columns = [
    "Feature_Set",
    "Feature_Count",
    "ROC-AUC",
    "PR-AUC",
    "Precision",
    "Recall",
    "F1",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "ROC-AUC":
                "{:.4f}".format,
            "PR-AUC":
                "{:.4f}".format,
            "Precision":
                "{:.4f}".format,
            "Recall":
                "{:.4f}".format,
            "F1":
                "{:.4f}".format,
        },
    )
)


# ============================================================
# BEST FEATURE SET
# ============================================================

best = results_df.iloc[0]

print("\n")
print("=" * 100)
print("BEST FEATURE SET BY PR-AUC")
print("=" * 100)

print(
    f"\nFeature set : "
    f"{best['Feature_Set']}"
)

print(
    f"Feature count: "
    f"{best['Feature_Count']}"
)

print(
    f"ROC-AUC     : "
    f"{best['ROC-AUC']:.4f}"
)

print(
    f"PR-AUC      : "
    f"{best['PR-AUC']:.4f}"
)

print(
    f"Precision   : "
    f"{best['Precision']:.4f}"
)

print(
    f"Recall      : "
    f"{best['Recall']:.4f}"
)

print(
    f"F1          : "
    f"{best['F1']:.4f}"
)

print(
    "\nSelected features:"
)

print(
    best["Features"]
)


# ============================================================
# CURRENT 15-FEATURE COMPARISON
# ============================================================

current = results_df[
    results_df["Feature_Set"]
    == "Current_15"
]

if not current.empty:

    current = current.iloc[0]

    print("\n")
    print("=" * 100)
    print("CURRENT 15-FEATURE SET")
    print("=" * 100)

    print(
        f"\nROC-AUC : "
        f"{current['ROC-AUC']:.4f}"
    )

    print(
        f"PR-AUC  : "
        f"{current['PR-AUC']:.4f}"
    )

    print(
        f"Precision: "
        f"{current['Precision']:.4f}"
    )

    print(
        f"Recall  : "
        f"{current['Recall']:.4f}"
    )

    print(
        f"F1      : "
        f"{current['F1']:.4f}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    REPORT_PATH,
    index=False,
)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 100)
print("LEAKAGE-SAFE FEATURE SELECTION COMPLETED")
print("=" * 100)

print(
    f"\nReport saved to:"
    f"\n{REPORT_PATH}"
)

print(
    "\nIMPORTANT:"
    "\n- Test set was NOT used for feature selection."
    "\n- Oversampling occurred only inside CV training folds."
    "\n- Validation folds remained untouched."
)