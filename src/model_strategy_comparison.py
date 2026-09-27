from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_churn.csv"


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
# DATA LOADING
# ============================================================

print("=" * 100)
print("MODEL TRAINING STRATEGY COMPARISON")
print("=" * 100)

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("\nDataset:")
print(f"Rows: {len(df)}")
print(f"Features: {len(FEATURES)}")

print("\nOriginal target distribution:")
print(y.value_counts().sort_index())

print("\nOriginal target percentage:")
print(
    (y.value_counts(normalize=True).sort_index() * 100)
    .round(2)
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
# STRATEGIES
# ============================================================

STRATEGIES = [
    "Original Distribution",
    "50/50 Oversampling",
    "Class Weighted",
]


# ============================================================
# OVERSAMPLING
# ============================================================

def oversample_training_data(X_train, y_train):

    train_df = X_train.copy()
    train_df["_target_"] = y_train.to_numpy()

    majority = train_df[train_df["_target_"] == 0]
    minority = train_df[train_df["_target_"] == 1]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )

    balanced = pd.concat(
        [majority, minority_upsampled],
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
# CLASS WEIGHT / SAMPLE WEIGHT
# ============================================================

def create_sample_weights(y_train):

    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)

    positive_weight = negative_count / positive_count

    weights = np.where(
        y_train == 1,
        positive_weight,
        1.0,
    )

    return weights


# ============================================================
# MODEL FITTING
# ============================================================

def fit_model(
    model_name,
    model,
    X_train,
    y_train,
    strategy,
):

    sample_weights = None

    # --------------------------------------------------------
    # Strategy 1: Original distribution
    # --------------------------------------------------------

    if strategy == "Original Distribution":

        X_fit = X_train
        y_fit = y_train

    # --------------------------------------------------------
    # Strategy 2: 50/50 oversampling
    # --------------------------------------------------------

    elif strategy == "50/50 Oversampling":

        X_fit, y_fit = oversample_training_data(
            X_train,
            y_train,
        )

    # --------------------------------------------------------
    # Strategy 3: Class weighted
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    preprocessor = create_preprocessor()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

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
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


results = []


# ============================================================
# MAIN EXPERIMENT
# ============================================================

for strategy in STRATEGIES:

    print("\n")
    print("=" * 100)
    print(f"STRATEGY: {strategy}")
    print("=" * 100)

    for model_name in create_models().keys():

        print(f"\n{model_name}")

        fold_metrics = []

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            print(
                f"  Fold {fold}/5",
                end="",
                flush=True,
            )

            X_train = X.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()

            y_train = y.iloc[train_idx].copy()
            y_valid = y.iloc[valid_idx].copy()

            model = create_models()[model_name]

            pipeline = fit_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                strategy=strategy,
            )

            probabilities = pipeline.predict_proba(
                X_valid
            )[:, 1]

            predictions = (
                probabilities >= 0.50
            ).astype(int)

            metrics = {

                "accuracy": accuracy_score(
                    y_valid,
                    predictions,
                ),

                "precision": precision_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),

                "recall": recall_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),

                "f1": f1_score(
                    y_valid,
                    predictions,
                    zero_division=0,
                ),

                "roc_auc": roc_auc_score(
                    y_valid,
                    probabilities,
                ),

                "pr_auc": average_precision_score(
                    y_valid,
                    probabilities,
                ),
            }

            fold_metrics.append(metrics)

            print(" ✓")

        # ----------------------------------------------------
        # Average folds
        # ----------------------------------------------------

        mean_metrics = {
            metric: np.mean(
                [
                    fold[metric]
                    for fold in fold_metrics
                ]
            )
            for metric in fold_metrics[0]
        }

        results.append(
            {
                "Strategy": strategy,
                "Model": model_name,
                "Accuracy": mean_metrics["accuracy"],
                "Precision": mean_metrics["precision"],
                "Recall": mean_metrics["recall"],
                "F1": mean_metrics["f1"],
                "ROC-AUC": mean_metrics["roc_auc"],
                "PR-AUC": mean_metrics["pr_auc"],
            }
        )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "PR-AUC",
        "F1",
        "ROC-AUC",
    ],
    ascending=False,
)


print("\n")
print("=" * 100)
print("FINAL STRATEGY COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
            "PR-AUC": "{:.4f}".format,
        },
    )
)


# ============================================================
# BEST BY EACH METRIC
# ============================================================

print("\n")
print("=" * 100)
print("BEST RESULT BY METRIC")
print("=" * 100)

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
]:

    best_row = results_df.loc[
        results_df[metric].idxmax()
    ]

    print(
        f"\n{metric}:"
    )

    print(
        f"  Strategy : {best_row['Strategy']}"
    )

    print(
        f"  Model    : {best_row['Model']}"
    )

    print(
        f"  Score    : {best_row[metric]:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_strategy_comparison.csv"
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    REPORT_PATH,
    index=False,
)

print("\n")
print("=" * 100)
print("EXPERIMENT COMPLETED")
print("=" * 100)

print(
    f"\nResults saved to:"
    f"\n{REPORT_PATH}"
)

print(
    "\nIMPORTANT:"
    "\nThe untouched test set was NOT used."
)