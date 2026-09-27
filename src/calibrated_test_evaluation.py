from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split, StratifiedKFold
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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET = "Churn"

TEST_SIZE = 0.20

INNER_SPLITS = 4


# Thresholds selected ONLY from previous OOF experiment
THRESHOLDS = {
    "Uncalibrated": 0.68,
    "Sigmoid": 0.44,
    "Isotonic": 0.42,
}


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
]


# ============================================================
# CATBOOST
# ============================================================

def build_catboost():

    cat_indices = [
        ALL_FEATURES.index(feature)
        for feature in CATEGORICAL_FEATURES
    ]

    return CatBoostClassifier(
        iterations=250,
        depth=4,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=1,
        cat_features=cat_indices,
    )


# ============================================================
# 50/50 OVERSAMPLING
# ============================================================

def oversample_training_data(
    X_train,
    y_train,
):

    train_df = X_train.copy()

    train_df[TARGET] = y_train.values

    majority = train_df[
        train_df[TARGET] == 0
    ]

    minority = train_df[
        train_df[TARGET] == 1
    ]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=RANDOM_STATE,
    )

    balanced = pd.concat(
        [
            majority,
            minority_upsampled,
        ],
        ignore_index=True,
    )

    balanced = (
        balanced
        .sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    return (
        balanced.drop(
            columns=[TARGET]
        ),
        balanced[TARGET],
    )


# ============================================================
# FIT MODEL
# ============================================================

def fit_model(
    X_train,
    y_train,
):

    model = build_catboost()

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# SIGMOID CALIBRATION
# ============================================================

def fit_sigmoid_calibrator(
    probabilities,
    y_true,
):

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    logits = np.log(
        probabilities
        / (1 - probabilities)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    calibrator.fit(
        logits,
        y_true,
    )

    return calibrator


def apply_sigmoid_calibrator(
    calibrator,
    probabilities,
):

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    logits = np.log(
        probabilities
        / (1 - probabilities)
    ).reshape(-1, 1)

    return calibrator.predict_proba(
        logits
    )[:, 1]


# ============================================================
# ISOTONIC CALIBRATION
# ============================================================

def fit_isotonic_calibrator(
    probabilities,
    y_true,
):

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        probabilities,
        y_true,
    )

    return calibrator


def apply_isotonic_calibrator(
    calibrator,
    probabilities,
):

    return calibrator.predict(
        probabilities
    )


# ============================================================
# CREATE CALIBRATION DATA
# ============================================================

def generate_calibration_predictions(
    X_train,
    y_train,
):

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    calibration_probabilities = np.zeros(
        len(X_train)
    )

    for fold, (
        inner_train_idx,
        calibration_idx,
    ) in enumerate(
        inner_cv.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"Calibration fold "
            f"{fold}/{INNER_SPLITS}"
        )

        X_inner_train = (
            X_train
            .iloc[inner_train_idx]
            .copy()
            .reset_index(drop=True)
        )

        y_inner_train = (
            y_train
            .iloc[inner_train_idx]
            .copy()
            .reset_index(drop=True)
        )

        X_calibration = (
            X_train
            .iloc[calibration_idx]
            .copy()
            .reset_index(drop=True)
        )

        (
            X_inner_train,
            y_inner_train,
        ) = oversample_training_data(
            X_inner_train,
            y_inner_train,
        )

        model = fit_model(
            X_inner_train,
            y_inner_train,
        )

        calibration_probabilities[
            calibration_idx
        ] = model.predict_proba(
            X_calibration
        )[:, 1]

    return calibration_probabilities


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    return {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
        "PR-AUC": average_precision_score(
            y_true,
            probabilities,
        ),
        "Brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CALIBRATED CATBOOST - FINAL TEST EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    X = df[
        ALL_FEATURES
    ].copy()

    y = (
        df[TARGET]
        .astype(int)
        .copy()
    )

    # --------------------------------------------------------
    # RECREATE ORIGINAL 80/20 SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train = X_train.reset_index(
        drop=True
    )

    X_test = X_test.reset_index(
        drop=True
    )

    y_train = y_train.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )

    print(
        f"\nTraining rows: {len(X_train)}"
    )

    print(
        f"Test rows:     {len(X_test)}"
    )

    print(
        "\nTraining target:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )

    print(
        "\nUntouched test target:"
    )

    print(
        y_test.value_counts()
        .sort_index()
    )

    # ========================================================
    # CALIBRATION DATA
    # ========================================================

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Generating calibration predictions "
        "from training data only..."
    )

    print(
        "-" * 80
    )

    calibration_probabilities = (
        generate_calibration_predictions(
            X_train,
            y_train,
        )
    )

    # ========================================================
    # FIT FINAL BALANCED MODEL
    # ========================================================

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Training final 50/50 CatBoost "
        "on complete training data..."
    )

    print(
        "-" * 80
    )

    (
        X_train_balanced,
        y_train_balanced,
    ) = oversample_training_data(
        X_train,
        y_train,
    )

    print(
        "\nBalanced training distribution:"
    )

    print(
        y_train_balanced
        .value_counts()
        .sort_index()
    )

    final_model = fit_model(
        X_train_balanced,
        y_train_balanced,
    )

    # ========================================================
    # RAW TEST PROBABILITIES
    # ========================================================

    raw_test_probabilities = (
        final_model.predict_proba(
            X_test
        )[:, 1]
    )

    # ========================================================
    # FIT CALIBRATORS
    # ========================================================

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Fitting sigmoid calibrator..."
    )

    print(
        "-" * 80
    )

    sigmoid_calibrator = (
        fit_sigmoid_calibrator(
            calibration_probabilities,
            y_train.values,
        )
    )

    print(
        "Sigmoid calibrator fitted."
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Fitting isotonic calibrator..."
    )

    print(
        "-" * 80
    )

    isotonic_calibrator = (
        fit_isotonic_calibrator(
            calibration_probabilities,
            y_train.values,
        )
    )

    print(
        "Isotonic calibrator fitted."
    )

    # ========================================================
    # CALIBRATED TEST PROBABILITIES
    # ========================================================

    sigmoid_test_probabilities = (
        apply_sigmoid_calibrator(
            sigmoid_calibrator,
            raw_test_probabilities,
        )
    )

    isotonic_test_probabilities = (
        apply_isotonic_calibrator(
            isotonic_calibrator,
            raw_test_probabilities,
        )
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    results = []

    results.append(
        evaluate_model(
            "Uncalibrated",
            y_test,
            raw_test_probabilities,
            THRESHOLDS["Uncalibrated"],
        )
    )

    results.append(
        evaluate_model(
            "Sigmoid",
            y_test,
            sigmoid_test_probabilities,
            THRESHOLDS["Sigmoid"],
        )
    )

    results.append(
        evaluate_model(
            "Isotonic",
            y_test,
            isotonic_test_probabilities,
            THRESHOLDS["Isotonic"],
        )
    )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "UNTOUCHED TEST SET RESULTS"
    )

    print(
        "=" * 80
    )

    print(

        results_df[

            [
                "Model",
                "Threshold",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC",
                "Brier",
                "TN",
                "FP",
                "FN",
                "TP",
            ]

        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.4f}",

        )

    )

    # ========================================================
    # PROBABILITY SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TEST PROBABILITY SUMMARY"
    )

    print(
        "=" * 80
    )

    probability_summary = pd.DataFrame(
        {
            "Model": [
                "Uncalibrated",
                "Sigmoid",
                "Isotonic",
            ],
            "Mean_Probability": [
                raw_test_probabilities.mean(),
                sigmoid_test_probabilities.mean(),
                isotonic_test_probabilities.mean(),
            ],
            "Median_Probability": [
                np.median(
                    raw_test_probabilities
                ),
                np.median(
                    sigmoid_test_probabilities
                ),
                np.median(
                    isotonic_test_probabilities
                ),
            ],
            "Min": [
                raw_test_probabilities.min(),
                sigmoid_test_probabilities.min(),
                isotonic_test_probabilities.min(),
            ],
            "Max": [
                raw_test_probabilities.max(),
                sigmoid_test_probabilities.max(),
                isotonic_test_probabilities.max(),
            ],
        }
    )

    print(
        probability_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = (
        REPORTS_DIR
        / "calibrated_catboost_test_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    probability_path = (
        REPORTS_DIR
        / "calibrated_catboost_test_probabilities.csv"
    )

    test_probability_df = pd.DataFrame(
        {
            "Actual_Churn":
                y_test.values,

            "Uncalibrated":
                raw_test_probabilities,

            "Sigmoid":
                sigmoid_test_probabilities,

            "Isotonic":
                isotonic_test_probabilities,
        }
    )

    test_probability_df.to_csv(
        probability_path,
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 80
    )

    print(
        f"Results: "
        f"{results_path}"
    )

    print(
        f"Probabilities: "
        f"{probability_path}"
    )

    print(
        "\nUntouched test evaluation completed."
    )


if __name__ == "__main__":
    main()