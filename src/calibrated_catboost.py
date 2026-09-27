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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
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

OUTER_SPLITS = 5
INNER_SPLITS = 4

TARGET = "Churn"


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

    X_balanced = balanced.drop(
        columns=[TARGET]
    )

    y_balanced = balanced[TARGET]

    return (
        X_balanced,
        y_balanced,
    )


# ============================================================
# FIT CATBOOST
# ============================================================

def fit_catboost(
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
# NESTED CALIBRATION
# ============================================================

def generate_calibrated_oof(
    X,
    y,
    calibration_method,
):

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.zeros(
        len(X)
    )

    for outer_fold, (
        outer_train_idx,
        outer_valid_idx,
    ) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):

        print(
            f"{calibration_method} | "
            f"Outer Fold "
            f"{outer_fold}/{OUTER_SPLITS}"
        )

        X_outer_train = (
            X.iloc[outer_train_idx]
            .copy()
            .reset_index(drop=True)
        )

        y_outer_train = (
            y.iloc[outer_train_idx]
            .copy()
            .reset_index(drop=True)
        )

        X_outer_valid = (
            X.iloc[outer_valid_idx]
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # INNER CV FOR CALIBRATION
        # ----------------------------------------------------

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        calibration_probabilities = []

        calibration_targets = []

        for inner_fold, (
            inner_train_idx,
            inner_cal_idx,
        ) in enumerate(
            inner_cv.split(
                X_outer_train,
                y_outer_train,
            ),
            start=1,
        ):

            X_inner_train = (
                X_outer_train
                .iloc[inner_train_idx]
                .copy()
                .reset_index(drop=True)
            )

            y_inner_train = (
                y_outer_train
                .iloc[inner_train_idx]
                .copy()
                .reset_index(drop=True)
            )

            X_inner_cal = (
                X_outer_train
                .iloc[inner_cal_idx]
                .copy()
                .reset_index(drop=True)
            )

            y_inner_cal = (
                y_outer_train
                .iloc[inner_cal_idx]
                .copy()
                .reset_index(drop=True)
            )

            # -----------------------------------------------
            # BALANCE ONLY INNER TRAINING DATA
            # -----------------------------------------------

            (
                X_inner_train,
                y_inner_train,
            ) = oversample_training_data(
                X_inner_train,
                y_inner_train,
            )

            # -----------------------------------------------
            # TRAIN BASE MODEL
            # -----------------------------------------------

            model = fit_catboost(
                X_inner_train,
                y_inner_train,
            )

            inner_probabilities = (
                model.predict_proba(
                    X_inner_cal
                )[:, 1]
            )

            calibration_probabilities.extend(
                inner_probabilities
            )

            calibration_targets.extend(
                y_inner_cal.tolist()
            )

            print(
                f"  Inner Fold "
                f"{inner_fold}/{INNER_SPLITS} "
                f"complete"
            )

        calibration_probabilities = np.asarray(
            calibration_probabilities
        )

        calibration_targets = np.asarray(
            calibration_targets
        )

        # ----------------------------------------------------
        # FIT CALIBRATOR
        # ----------------------------------------------------

        if calibration_method == "Sigmoid":

            calibrator = (
                fit_sigmoid_calibrator(
                    calibration_probabilities,
                    calibration_targets,
                )
            )

        elif calibration_method == "Isotonic":

            calibrator = (
                fit_isotonic_calibrator(
                    calibration_probabilities,
                    calibration_targets,
                )
            )

        else:

            raise ValueError(
                f"Unknown calibration method: "
                f"{calibration_method}"
            )

        # ----------------------------------------------------
        # TRAIN FINAL OUTER MODEL
        # ----------------------------------------------------

        (
            X_outer_train_balanced,
            y_outer_train_balanced,
        ) = oversample_training_data(
            X_outer_train,
            y_outer_train,
        )

        outer_model = fit_catboost(
            X_outer_train_balanced,
            y_outer_train_balanced,
        )

        # ----------------------------------------------------
        # OUTER VALIDATION
        # ----------------------------------------------------

        outer_raw_probabilities = (
            outer_model.predict_proba(
                X_outer_valid
            )[:, 1]
        )

        if calibration_method == "Sigmoid":

            calibrated_probabilities = (
                apply_sigmoid_calibrator(
                    calibrator,
                    outer_raw_probabilities,
                )
            )

        else:

            calibrated_probabilities = (
                apply_isotonic_calibrator(
                    calibrator,
                    outer_raw_probabilities,
                )
            )

        oof_probabilities[
            outer_valid_idx
        ] = calibrated_probabilities

        print(
            f"  Outer Fold "
            f"{outer_fold} complete"
        )

    return oof_probabilities


# ============================================================
# UNCALIBRATED BASELINE
# ============================================================

def generate_uncalibrated_oof(
    X,
    y,
):

    cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.zeros(
        len(X)
    )

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"Uncalibrated | "
            f"Fold {fold}/{OUTER_SPLITS}"
        )

        X_train = (
            X.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        y_train = (
            y.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        X_valid = (
            X.iloc[valid_idx]
            .copy()
            .reset_index(drop=True)
        )

        (
            X_train,
            y_train,
        ) = oversample_training_data(
            X_train,
            y_train,
        )

        model = fit_catboost(
            X_train,
            y_train,
        )

        oof_probabilities[
            valid_idx
        ] = model.predict_proba(
            X_valid
        )[:, 1]

        print(
            f"  Fold {fold} complete"
        )

    return oof_probabilities


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

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

    }


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    y_true,
    probabilities,
):

    rows = []

    thresholds = np.round(
        np.arange(
            0.10,
            0.901,
            0.02,
        ),
        2,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn = np.sum(
            (y_true == 0)
            & (predictions == 0)
        )

        fp = np.sum(
            (y_true == 0)
            & (predictions == 1)
        )

        fn = np.sum(
            (y_true == 1)
            & (predictions == 0)
        )

        tp = np.sum(
            (y_true == 1)
            & (predictions == 1)
        )

        rows.append(

            {

                "Threshold": threshold,

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

                "Accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),

                "TN": tn,

                "FP": fp,

                "FN": fn,

                "TP": tp,

                "Predicted_Churn_%":
                    predictions.mean() * 100,

            }

        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "CALIBRATED CATBOOST EXPERIMENT"
    )
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

    print(
        f"\nDataset shape: {df.shape}"
    )

    print(
        f"Features used: "
        f"{len(ALL_FEATURES)}"
    )

    print(
        "\nOriginal target distribution:"
    )

    print(
        y.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # GENERATE PROBABILITIES
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Generating uncalibrated OOF predictions..."
    )

    print(
        "-" * 80
    )

    uncalibrated = (
        generate_uncalibrated_oof(
            X,
            y,
        )
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Generating sigmoid-calibrated "
        "OOF predictions..."
    )

    print(
        "-" * 80
    )

    sigmoid = (
        generate_calibrated_oof(
            X,
            y,
            "Sigmoid",
        )
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Generating isotonic-calibrated "
        "OOF predictions..."
    )

    print(
        "-" * 80
    )

    isotonic = (
        generate_calibrated_oof(
            X,
            y,
            "Isotonic",
        )
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    probability_sets = {

        "Uncalibrated":
            uncalibrated,

        "Sigmoid":
            sigmoid,

        "Isotonic":
            isotonic,

    }

    results = []

    threshold_results = []

    for method, probabilities in (
        probability_sets.items()
    ):

        metrics = calculate_metrics(
            y,
            probabilities,
            threshold=0.50,
        )

        results.append(
            {
                "Method": method,
                **metrics,
            }
        )

        threshold_df = (
            threshold_analysis(
                y,
                probabilities,
            )
        )

        threshold_df.insert(
            0,
            "Method",
            method,
        )

        threshold_results.append(
            threshold_df
        )

    results_df = pd.DataFrame(
        results
    )

    threshold_results_df = (
        pd.concat(
            threshold_results,
            ignore_index=True,
        )
    )

    # --------------------------------------------------------
    # SAVE OOF PROBABILITIES
    # --------------------------------------------------------

    probability_output = pd.DataFrame(
        {
            "Actual_Churn": y.values,

            "Uncalibrated":
                uncalibrated,

            "Sigmoid":
                sigmoid,

            "Isotonic":
                isotonic,

        }
    )

    probability_path = (
        REPORTS_DIR
        / "calibrated_catboost_oof_probabilities.csv"
    )

    probability_output.to_csv(
        probability_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_path = (
        REPORTS_DIR
        / "calibrated_catboost_metrics.csv"
    )

    results_df.to_csv(
        metrics_path,
        index=False,
    )

    threshold_path = (
        REPORTS_DIR
        / "calibrated_catboost_thresholds.csv"
    )

    threshold_results_df.to_csv(
        threshold_path,
        index=False,
    )

    # ========================================================
    # PRINT MAIN RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CALIBRATION COMPARISON"
    )

    print(
        "=" * 80
    )

    print(

        results_df[
            [
                "Method",
                "Brier",
                "ROC-AUC",
                "PR-AUC",
                "Precision",
                "Recall",
                "F1",
                "Accuracy",
            ]
        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.4f}",

        )

    )

    # ========================================================
    # BEST F1
    # ========================================================

    best_f1 = (

        threshold_results_df

        .sort_values(
            "F1",
            ascending=False,
        )

        .groupby(
            "Method",
            as_index=False,
        )

        .first()

    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST F1 THRESHOLD"
    )

    print(
        "=" * 80
    )

    print(

        best_f1[

            [
                "Method",
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "Accuracy",
                "FP",
                "FN",
                "TP",
                "TN",
            ]

        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.4f}",

        )

    )

    # ========================================================
    # BEST PRECISION WITH RECALL >= 60%
    # ========================================================

    recall_target = (

        threshold_results_df[
            threshold_results_df[
                "Recall"
            ] >= 0.60
        ]

        .sort_values(
            "Precision",
            ascending=False,
        )

        .groupby(
            "Method",
            as_index=False,
        )

        .first()

    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST PRECISION WITH RECALL >= 60%"
    )

    print(
        "=" * 80
    )

    print(

        recall_target[

            [
                "Method",
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "Accuracy",
                "FP",
                "FN",
                "TP",
                "TN",
            ]

        ].to_string(

            index=False,

            float_format=lambda x:
                f"{x:.4f}",

        )

    )

    # ========================================================
    # CALIBRATION PLOT
    # ========================================================

    from sklearn.calibration import calibration_curve

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(9, 7)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    for method, probabilities in (
        probability_sets.items()
    ):

        prob_true, prob_pred = (
            calibration_curve(
                y,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
        )

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Observed Churn Rate"
    )

    plt.title(
        "CatBoost Probability Calibration"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plot_path = (
        REPORTS_DIR
        / "calibrated_catboost.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FILES
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
        f"Probabilities: "
        f"{probability_path}"
    )

    print(
        f"Metrics:       "
        f"{metrics_path}"
    )

    print(
        f"Thresholds:    "
        f"{threshold_path}"
    )

    print(
        f"Plot:          "
        f"{plot_path}"
    )

    print(
        "\nCalibration experiment completed successfully."
    )


if __name__ == "__main__":
    main()