from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

from catboost import CatBoostClassifier

from feature_config import ALL_FEATURES


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
N_SPLITS = 5

TARGET = "Churn"


# ============================================================
# FEATURE TYPES
# ============================================================

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    feature
    for feature in ALL_FEATURES
    if feature not in NUMERIC_FEATURES
]


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# GRADIENT BOOSTING
# ============================================================

def build_gradient_boosting():

    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )


# ============================================================
# CATBOOST
# ============================================================

def build_catboost(
    strategy,
):

    if strategy == "Class Weighted":

        return CatBoostClassifier(
            iterations=250,
            depth=4,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=1,
            auto_class_weights="Balanced",
        )

    return CatBoostClassifier(
        iterations=250,
        depth=4,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=1,
    )


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

def prepare_training_data(
    X_train,
    y_train,
    strategy,
):

    X_train = (
        X_train
        .reset_index(drop=True)
        .copy()
    )

    y_train = (
        y_train
        .reset_index(drop=True)
        .copy()
    )

    if strategy == "Original Distribution":

        return X_train, y_train

    if strategy == "50/50 Oversampling":

        train_df = X_train.copy()

        train_df[TARGET] = y_train

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

    if strategy == "Class Weighted":

        return X_train, y_train

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# BUILD GRADIENT BOOSTING PIPELINE
# ============================================================

def build_gradient_pipeline():

    preprocessor = build_preprocessor()

    model = build_gradient_boosting()

    return Pipeline(
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


# ============================================================
# SAMPLE WEIGHTS FOR GRADIENT BOOSTING
# ============================================================

def get_sample_weights(y_train):

    y_array = np.asarray(y_train)

    count_0 = np.sum(
        y_array == 0
    )

    count_1 = np.sum(
        y_array == 1
    )

    total = len(y_array)

    weight_0 = (
        total
        / (2 * count_0)
    )

    weight_1 = (
        total
        / (2 * count_1)
    )

    return np.where(
        y_array == 0,
        weight_0,
        weight_1,
    )


# ============================================================
# OOF PREDICTIONS
# ============================================================

def generate_oof_predictions(
    X,
    y,
    model_name,
    strategy,
):

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
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
        skf.split(X, y),
        start=1,
    ):

        print(
            f"{model_name} | "
            f"{strategy} | "
            f"Fold {fold}/{N_SPLITS}"
        )

        # ----------------------------------------------------
        # ORIGINAL FOLD
        # ----------------------------------------------------

        X_train = (
            X.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        X_valid = (
            X.iloc[valid_idx]
            .copy()
            .reset_index(drop=True)
        )

        y_train = (
            y.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # BALANCE TRAINING DATA ONLY
        # ----------------------------------------------------

        (
            X_train,
            y_train,
        ) = prepare_training_data(
            X_train,
            y_train,
            strategy,
        )

        # ====================================================
        # GRADIENT BOOSTING
        # ====================================================

        if model_name == "Gradient Boosting":

            pipeline = (
                build_gradient_pipeline()
            )

            fit_kwargs = {}

            if strategy == "Class Weighted":

                sample_weights = (
                    get_sample_weights(
                        y_train
                    )
                )

                fit_kwargs[
                    "model__sample_weight"
                ] = sample_weights

            pipeline.fit(
                X_train,
                y_train,
                **fit_kwargs,
            )

            probabilities = (
                pipeline
                .predict_proba(
                    X_valid
                )[:, 1]
            )

        # ====================================================
        # CATBOOST
        # ====================================================

        elif model_name == "CatBoost":

            catboost = build_catboost(
                strategy
            )

            cat_indices = [
                X_train.columns.get_loc(
                    feature
                )
                for feature
                in CATEGORICAL_FEATURES
            ]

            catboost.set_params(
                cat_features=cat_indices
            )

            catboost.fit(
                X_train,
                y_train,
            )

            probabilities = (
                catboost
                .predict_proba(
                    X_valid
                )[:, 1]
            )

        else:

            raise ValueError(
                f"Unknown model: "
                f"{model_name}"
            )

        oof_probabilities[
            valid_idx
        ] = probabilities

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

                "Predicted_Churn_%": (
                    predictions.mean()
                    * 100
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PROBABILITY CALIBRATION EXPERIMENT"
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

    print(
        "\nFeature list:"
    )

    for feature in ALL_FEATURES:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # --------------------------------------------------------

    strategies = [

        "Original Distribution",

        "50/50 Oversampling",

        "Class Weighted",

    ]

    models = [

        "Gradient Boosting",

        "CatBoost",

    ]

    all_results = []

    all_threshold_results = []

    calibration_curves = []

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    for model_name in models:

        for strategy in strategies:

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"Running: "
                f"{model_name} | "
                f"{strategy}"
            )

            print(
                "-" * 80
            )

            probabilities = (
                generate_oof_predictions(
                    X,
                    y,
                    model_name,
                    strategy,
                )
            )

            # ------------------------------------------------
            # METRICS AT 0.50
            # ------------------------------------------------

            metrics = (
                calculate_metrics(
                    y,
                    probabilities,
                    threshold=0.50,
                )
            )

            result = {

                "Model": model_name,

                "Strategy": strategy,

                **metrics,

            }

            all_results.append(
                result
            )

            print(
                "\nMetrics at threshold 0.50:"
            )

            for key, value in metrics.items():

                print(
                    f"{key:<12}: "
                    f"{value:.4f}"
                )

            # ------------------------------------------------
            # THRESHOLD ANALYSIS
            # ------------------------------------------------

            threshold_df = (
                threshold_analysis(
                    y,
                    probabilities,
                )
            )

            threshold_df.insert(
                0,
                "Strategy",
                strategy,
            )

            threshold_df.insert(
                0,
                "Model",
                model_name,
            )

            all_threshold_results.append(
                threshold_df
            )

            # ------------------------------------------------
            # CALIBRATION CURVE
            # ------------------------------------------------

            from sklearn.calibration import (
                calibration_curve
            )

            prob_true, prob_pred = (
                calibration_curve(
                    y,
                    probabilities,
                    n_bins=10,
                    strategy="quantile",
                )
            )

            calibration_curves.append(

                {

                    "Model": model_name,

                    "Strategy": strategy,

                    "Mean_Predicted":
                        prob_pred,

                    "Fraction_Positive":
                        prob_true,

                }
            )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        REPORTS_DIR
        / "probability_calibration_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ========================================================
    # SAVE THRESHOLD RESULTS
    # ========================================================

    threshold_results_df = (
        pd.concat(
            all_threshold_results,
            ignore_index=True,
        )
    )

    threshold_path = (
        REPORTS_DIR
        / "probability_calibration_thresholds.csv"
    )

    threshold_results_df.to_csv(
        threshold_path,
        index=False,
    )

    # ========================================================
    # PRINT CALIBRATION RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CALIBRATION RESULTS"
    )

    print(
        "=" * 80
    )

    display_columns = [

        "Model",

        "Strategy",

        "Brier",

        "ROC-AUC",

        "PR-AUC",

        "Precision",

        "Recall",

        "F1",

        "Accuracy",

    ]

    print(

        results_df[
            display_columns
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
            [
                "Model",
                "Strategy",
            ],
            as_index=False,
        )

        .first()

    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST F1 THRESHOLD "
        "FOR EACH MODEL + STRATEGY"
    )

    print(
        "=" * 80
    )

    print(

        best_f1[

            [

                "Model",

                "Strategy",

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
            [
                "Model",
                "Strategy",
            ],
            as_index=False,
        )

        .first()

    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST PRECISION "
        "WITH RECALL >= 60%"
    )

    print(
        "=" * 80
    )

    print(

        recall_target[

            [

                "Model",

                "Strategy",

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

    for curve in calibration_curves:

        label = (

            f"{curve['Model']} - "

            f"{curve['Strategy']}"

        )

        plt.plot(

            curve[
                "Mean_Predicted"
            ],

            curve[
                "Fraction_Positive"
            ],

            marker="o",

            label=label,

        )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Observed Churn Rate"
    )

    plt.title(
        "Probability Calibration Comparison"
    )

    plt.legend(
        fontsize=8,
        loc="best",
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plot_path = (
        REPORTS_DIR
        / "probability_calibration.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FINAL OUTPUT
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
        f"Metrics:    "
        f"{results_path}"
    )

    print(
        f"Thresholds: "
        f"{threshold_path}"
    )

    print(
        f"Plot:       "
        f"{plot_path}"
    )

    print(
        "\nExperiment completed successfully."
    )


if __name__ == "__main__":

    main()