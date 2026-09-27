from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


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


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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

TARGET = "Churn"


# ============================================================
# PREPROCESSING
# ============================================================

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in NUMERICAL_FEATURES
]


def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# MODELS
# ============================================================

def create_models():

    return {

        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),

        "CatBoost": CatBoostClassifier(
            iterations=250,
            depth=4,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=42,
            thread_count=1,
        ),
    }


# ============================================================
# OVERSAMPLING
# ============================================================

def oversample_training_data(X_train, y_train):

    train_df = X_train.copy()

    train_df["_target_"] = y_train.to_numpy()

    majority = train_df[
        train_df["_target_"] == 0
    ]

    minority = train_df[
        train_df["_target_"] == 1
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
        ],
        axis=0,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    y_balanced = balanced.pop("_target_")

    X_balanced = balanced

    return X_balanced, y_balanced


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

def create_sample_weights(y_train):

    negative_count = np.sum(
        y_train == 0
    )

    positive_count = np.sum(
        y_train == 1
    )

    positive_weight = (
        negative_count
        / positive_count
    )

    return np.where(
        y_train == 1,
        positive_weight,
        1.0,
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    strategy,
):

    sample_weights = None

    if strategy == "Original Distribution":

        X_fit = X_train
        y_fit = y_train

    elif strategy == "50/50 Oversampling":

        X_fit, y_fit = oversample_training_data(
            X_train,
            y_train,
        )

    elif strategy == "Class Weighted":

        X_fit = X_train
        y_fit = y_train

        sample_weights = create_sample_weights(
            y_train.to_numpy()
        )

    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    if sample_weights is not None:

        pipeline.fit(
            X_fit,
            y_fit,
            model__sample_weight=sample_weights,
        )

    else:

        pipeline.fit(
            X_fit,
            y_fit,
        )

    return pipeline


# ============================================================
# DATA
# ============================================================

print("=" * 100)
print("STRATEGY + THRESHOLD ANALYSIS")
print("=" * 100)

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(
    f"\nDataset rows: {len(df)}"
)

print(
    "\nTarget distribution:"
)

print(
    y.value_counts()
)


# ============================================================
# OOF PREDICTIONS
# ============================================================

strategies = [
    "Original Distribution",
    "50/50 Oversampling",
    "Class Weighted",
]

models = create_models()

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


all_results = []


for strategy in strategies:

    for model_name in models:

        print("\n")
        print(
            "=" * 100
        )

        print(
            f"Generating OOF predictions:"
            f" {strategy} + {model_name}"
        )

        print(
            "=" * 100
        )

        oof_probabilities = np.zeros(
            len(y)
        )

        for fold, (
            train_idx,
            valid_idx,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            print(
                f"  Fold {fold}/5",
                end="",
                flush=True,
            )

            X_train = X.iloc[
                train_idx
            ].copy()

            X_valid = X.iloc[
                valid_idx
            ].copy()

            y_train = y.iloc[
                train_idx
            ].copy()

            model = models[
                model_name
            ]

            pipeline = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                strategy=strategy,
            )

            probabilities = (
                pipeline.predict_proba(
                    X_valid
                )[:, 1]
            )

            oof_probabilities[
                valid_idx
            ] = probabilities

            print(" ✓")

        # ====================================================
        # THRESHOLD ANALYSIS
        # ====================================================

        thresholds = np.arange(
            0.10,
            0.91,
            0.01,
        )

        for threshold in thresholds:

            predictions = (
                oof_probabilities
                >= threshold
            ).astype(int)

            tn, fp, fn, tp = (
                confusion_matrix(
                    y,
                    predictions,
                    labels=[0, 1],
                ).ravel()
            )

            precision = (
                precision_score(
                    y,
                    predictions,
                    zero_division=0,
                )
            )

            recall = (
                recall_score(
                    y,
                    predictions,
                    zero_division=0,
                )
            )

            f1 = (
                f1_score(
                    y,
                    predictions,
                    zero_division=0,
                )
            )

            accuracy = (
                accuracy_score(
                    y,
                    predictions,
                )
            )

            predicted_churn_rate = (
                predictions.mean()
            )

            all_results.append(
                {
                    "Strategy": strategy,
                    "Model": model_name,
                    "Threshold": round(
                        float(threshold),
                        2,
                    ),
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                    "Predicted_Churn_Rate":
                        predicted_churn_rate,
                    "ROC-AUC":
                        roc_auc_score(
                            y,
                            oof_probabilities,
                        ),
                    "PR-AUC":
                        average_precision_score(
                            y,
                            oof_probabilities,
                        ),
                }
            )


results_df = pd.DataFrame(
    all_results
)


# ============================================================
# BEST F1
# ============================================================

print("\n")
print("=" * 100)
print("BEST F1 CONFIGURATION")
print("=" * 100)

best_f1 = (
    results_df
    .sort_values(
        "F1",
        ascending=False,
    )
    .iloc[0]
)

print(
    best_f1.to_string()
)


# ============================================================
# BEST PRECISION WITH REASONABLE RECALL
# ============================================================

print("\n")
print("=" * 100)
print(
    "BEST PRECISION WITH RECALL >= 60%"
)
print("=" * 100)

eligible = results_df[
    results_df["Recall"] >= 0.60
]

best_precision_60 = (
    eligible
    .sort_values(
        [
            "Precision",
            "F1",
        ],
        ascending=False,
    )
    .iloc[0]
)

print(
    best_precision_60.to_string()
)


# ============================================================
# FP TARGET ANALYSIS
# ============================================================

for fp_limit in [
    50,
    75,
    100,
    125,
    150,
]:

    print("\n")
    print("=" * 100)

    print(
        f"BEST RECALL WITH FP <= {fp_limit}"
    )

    print("=" * 100)

    eligible = results_df[
        results_df["FP"] <= fp_limit
    ]

    if eligible.empty:

        print(
            "No configuration found."
        )

    else:

        best = (
            eligible
            .sort_values(
                [
                    "Recall",
                    "F1",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        print(
            best[
                [
                    "Strategy",
                    "Model",
                    "Threshold",
                    "Precision",
                    "Recall",
                    "F1",
                    "TN",
                    "FP",
                    "FN",
                    "TP",
                    "Accuracy",
                    "ROC-AUC",
                    "PR-AUC",
                ]
            ].to_string()
        )


# ============================================================
# BEST CONFIGURATIONS BY STRATEGY
# ============================================================

for strategy in strategies:

    strategy_df = results_df[
        results_df["Strategy"]
        == strategy
    ]

    print("\n")
    print("=" * 100)

    print(
        f"BEST CONFIGURATIONS: {strategy}"
    )

    print("=" * 100)

    # Best F1
    best = (
        strategy_df
        .sort_values(
            "F1",
            ascending=False,
        )
        .iloc[0]
    )

    print("\nBest F1:")

    print(
        best[
            [
                "Model",
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "TN",
                "FP",
                "FN",
                "TP",
                "Accuracy",
            ]
        ].to_string()
    )

    # Best precision with recall >= 60%
    eligible = strategy_df[
        strategy_df["Recall"] >= 0.60
    ]

    if not eligible.empty:

        best = (
            eligible
            .sort_values(
                [
                    "Precision",
                    "F1",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        print(
            "\nBest Precision "
            "with Recall >= 60%:"
        )

        print(
            best[
                [
                    "Model",
                    "Threshold",
                    "Precision",
                    "Recall",
                    "F1",
                    "TN",
                    "FP",
                    "FN",
                    "TP",
                    "Accuracy",
                ]
            ].to_string()
        )


# ============================================================
# SAVE
# ============================================================

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "strategy_threshold_analysis.csv"
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
print("STRATEGY THRESHOLD ANALYSIS COMPLETED")
print("=" * 100)

print(
    f"\nResults saved to:"
    f"\n{REPORT_PATH}"
)

print(
    "\nThe untouched test set was NOT used."
)