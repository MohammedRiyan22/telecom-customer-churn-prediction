from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

N_SPLITS = 5

TARGET = "Churn"


# ============================================================
# BASE FEATURES
# ============================================================

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
    "TotalCharges",
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

BASE_CATEGORICAL_FEATURES = [
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
# FEATURE ENGINEERING
# ============================================================

def create_engineered_features(df):
    """
    Create candidate interaction and count features.

    No target information is used.
    """

    df = df.copy()

    # --------------------------------------------------------
    # SERVICE / SUPPORT COUNT
    # --------------------------------------------------------

    support_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]

    df["SecuritySupportCount"] = 0

    for column in support_columns:

        df["SecuritySupportCount"] += (
            df[column]
            .eq("Yes")
            .astype(int)
        )

    # --------------------------------------------------------
    # TOTAL ADDITIONAL SERVICE COUNT
    # --------------------------------------------------------

    service_columns = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["AdditionalServiceCount"] = 0

    for column in service_columns:

        df["AdditionalServiceCount"] += (
            df[column]
            .eq("Yes")
            .astype(int)
        )

    # --------------------------------------------------------
    # TENURE GROUP
    # --------------------------------------------------------

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[
            -1,
            6,
            12,
            24,
            36,
            48,
            60,
            72,
        ],
        labels=[
            "0-6",
            "7-12",
            "13-24",
            "25-36",
            "37-48",
            "49-60",
            "61-72",
        ],
    ).astype(str)

    # --------------------------------------------------------
    # MONTHLY CHARGE GROUP
    # --------------------------------------------------------

    df["MonthlyChargeGroup"] = pd.cut(
        df["MonthlyCharges"],
        bins=[
            -1,
            30,
            50,
            70,
            90,
            120,
            np.inf,
        ],
        labels=[
            "<=30",
            "31-50",
            "51-70",
            "71-90",
            "91-120",
            ">120",
        ],
    ).astype(str)

    # --------------------------------------------------------
    # CONTRACT + TENURE
    # --------------------------------------------------------

    df["Contract_Tenure"] = (
        df["Contract"].astype(str)
        + "_"
        + df["TenureGroup"].astype(str)
    )

    # --------------------------------------------------------
    # CONTRACT + PAYMENT
    # --------------------------------------------------------

    df["Contract_Payment"] = (
        df["Contract"].astype(str)
        + "_"
        + df["PaymentMethod"].astype(str)
    )

    # --------------------------------------------------------
    # INTERNET + SECURITY
    # --------------------------------------------------------

    df["Internet_Security"] = (
        df["InternetService"].astype(str)
        + "_"
        + df["OnlineSecurity"].astype(str)
    )

    # --------------------------------------------------------
    # INTERNET + SUPPORT
    # --------------------------------------------------------

    df["Internet_Support"] = (
        df["InternetService"].astype(str)
        + "_"
        + df["TechSupport"].astype(str)
    )

    # --------------------------------------------------------
    # INTERNET + CONTRACT
    # --------------------------------------------------------

    df["Internet_Contract"] = (
        df["InternetService"].astype(str)
        + "_"
        + df["Contract"].astype(str)
    )

    # --------------------------------------------------------
    # PAYMENT + SUPPORT
    # --------------------------------------------------------

    df["Payment_Support"] = (
        df["PaymentMethod"].astype(str)
        + "_"
        + df["TechSupport"].astype(str)
    )

    # --------------------------------------------------------
    # HIGH CHARGE + LOW TENURE
    # --------------------------------------------------------

    df["HighCharge_LowTenure"] = (
        (
            df["MonthlyCharges"] >= 70
        )
        &
        (
            df["tenure"] <= 12
        )
    ).astype(int)

    # --------------------------------------------------------
    # LOW SUPPORT + MONTH-TO-MONTH
    # --------------------------------------------------------

    df["MonthToMonth_LowSupport"] = (
        (
            df["Contract"]
            == "Month-to-month"
        )
        &
        (
            df[
                "SecuritySupportCount"
            ] <= 1
        )
    ).astype(int)

    return df


# ============================================================
# FEATURE GROUPS
# ============================================================

def get_feature_groups():

    return {

        "Baseline_19": [],

        "Support_Counts": [
            "SecuritySupportCount",
            "AdditionalServiceCount",
        ],

        "Tenure_And_Charge_Groups": [
            "TenureGroup",
            "MonthlyChargeGroup",
        ],

        "Major_Interactions": [
            "Contract_Tenure",
            "Contract_Payment",
            "Internet_Security",
            "Internet_Support",
            "Internet_Contract",
            "Payment_Support",
        ],

        "Risk_Indicators": [
            "HighCharge_LowTenure",
            "MonthToMonth_LowSupport",
        ],

        "All_Engineered": [
            "SecuritySupportCount",
            "AdditionalServiceCount",
            "TenureGroup",
            "MonthlyChargeGroup",
            "Contract_Tenure",
            "Contract_Payment",
            "Internet_Security",
            "Internet_Support",
            "Internet_Contract",
            "Payment_Support",
            "HighCharge_LowTenure",
            "MonthToMonth_LowSupport",
        ],

    }


# ============================================================
# OVERSAMPLING
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
# CATBOOST
# ============================================================

def build_model(
    feature_columns,
):

    categorical_features = [
        feature
        for feature in feature_columns
        if (
            feature in BASE_CATEGORICAL_FEATURES
            or feature
            in [
                "TenureGroup",
                "MonthlyChargeGroup",
                "Contract_Tenure",
                "Contract_Payment",
                "Internet_Security",
                "Internet_Support",
                "Internet_Contract",
                "Payment_Support",
            ]
        )
    ]

    cat_indices = [
        feature_columns.index(feature)
        for feature in categorical_features
    ]

    model = CatBoostClassifier(
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

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_feature_group(
    df,
    feature_columns,
    group_name,
):

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET
    ].astype(int).copy()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.zeros(
        len(df)
    )

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

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
        # BALANCE TRAINING FOLD ONLY
        # ----------------------------------------------------

        (
            X_train,
            y_train,
        ) = oversample_training_data(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = build_model(
            feature_columns
        )

        model.fit(
            X_train,
            y_train,
        )

        oof_probabilities[
            valid_idx
        ] = model.predict_proba(
            X_valid
        )[:, 1]

        print(
            f"{group_name} | "
            f"Fold {fold}/{N_SPLITS} complete"
        )

    # --------------------------------------------------------
    # THRESHOLD = 0.50
    # --------------------------------------------------------

    threshold = 0.50

    predictions = (
        oof_probabilities
        >= threshold
    ).astype(int)

    return {

        "Feature_Group":
            group_name,

        "Feature_Count":
            len(feature_columns),

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

        "Accuracy":
            accuracy_score(
                y,
                predictions,
            ),

        "Precision":
            precision_score(
                y,
                predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y,
                predictions,
                zero_division=0,
            ),

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "FEATURE ENGINEERING EXPERIMENT"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"\nDataset shape: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # CREATE FEATURES
    # --------------------------------------------------------

    engineered_df = (
        create_engineered_features(
            df
        )
    )

    engineered_features = [
        column
        for column in engineered_df.columns
        if column not in df.columns
        and column != TARGET
    ]

    print(
        "\nEngineered features:"
    )

    for feature in engineered_features:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # FEATURE GROUPS
    # --------------------------------------------------------

    groups = (
        get_feature_groups()
    )

    results = []

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    for group_name, additions in (
        groups.items()
    ):

        feature_columns = (
            BASE_FEATURES
            + additions
        )

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Running: "
            f"{group_name}"
        )

        print(
            f"Features: "
            f"{len(feature_columns)}"
        )

        print(
            "-" * 80
        )

        result = (
            evaluate_feature_group(
                engineered_df,
                feature_columns,
                group_name,
            )
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "PR-AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FEATURE ENGINEERING RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        REPORTS_DIR
        / "feature_engineering_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RESULTS SAVED"
    )

    print(
        "=" * 80
    )

    print(
        output_path
    )

    print(
        "\nFeature engineering experiment completed."
    )


if __name__ == "__main__":

    main()