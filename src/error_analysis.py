from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


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

PROBABILITY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "calibrated_catboost_test_probabilities.csv"
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

TARGET = "Churn"

THRESHOLD = 0.44

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CUSTOMER CHURN ERROR ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD ORIGINAL DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"\nOriginal dataset shape: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # RECREATE EXACT ORIGINAL TEST SPLIT
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET].astype(int)

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

    # --------------------------------------------------------
    # LOAD SAVED TEST PROBABILITIES
    # --------------------------------------------------------

    probability_df = pd.read_csv(
        PROBABILITY_PATH
    )

    if len(probability_df) != len(X_test):

        raise ValueError(
            "Probability file and test set "
            "have different numbers of rows."
        )

    print(
        f"Test rows: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # ALIGN TEST DATA
    # --------------------------------------------------------

    analysis_df = X_test.copy()

    analysis_df = analysis_df.reset_index(
        drop=False
    )

    analysis_df.rename(
        columns={
            "index": "Original_Row_Index"
        },
        inplace=True,
    )

    analysis_df[
        "Actual_Churn"
    ] = y_test.values

    analysis_df[
        "Uncalibrated_Probability"
    ] = probability_df[
        "Uncalibrated"
    ].values

    analysis_df[
        "Sigmoid_Probability"
    ] = probability_df[
        "Sigmoid"
    ].values

    analysis_df[
        "Isotonic_Probability"
    ] = probability_df[
        "Isotonic"
    ].values

    # --------------------------------------------------------
    # FINAL PREDICTION
    # --------------------------------------------------------

    analysis_df[
        "Predicted_Churn"
    ] = (
        analysis_df[
            "Sigmoid_Probability"
        ] >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # ERROR TYPE
    # --------------------------------------------------------

    analysis_df[
        "Error_Type"
    ] = "Correct"

    analysis_df.loc[
        (
            (analysis_df["Actual_Churn"] == 0)
            &
            (analysis_df["Predicted_Churn"] == 1)
        ),
        "Error_Type",
    ] = "False Positive"

    analysis_df.loc[
        (
            (analysis_df["Actual_Churn"] == 1)
            &
            (analysis_df["Predicted_Churn"] == 0)
        ),
        "Error_Type",
    ] = "False Negative"

    analysis_df.loc[
        (
            (analysis_df["Actual_Churn"] == 1)
            &
            (analysis_df["Predicted_Churn"] == 1)
        ),
        "Error_Type",
    ] = "True Positive"

    analysis_df.loc[
        (
            (analysis_df["Actual_Churn"] == 0)
            &
            (analysis_df["Predicted_Churn"] == 0)
        ),
        "Error_Type",
    ] = "True Negative"

    # ========================================================
    # OVERALL CONFUSION SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OVERALL ERROR SUMMARY"
    )

    print(
        "=" * 80
    )

    error_counts = (
        analysis_df[
            "Error_Type"
        ]
        .value_counts()
    )

    total = len(
        analysis_df
    )

    for error_type in [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive",
    ]:

        count = error_counts.get(
            error_type,
            0,
        )

        percentage = (
            count / total * 100
        )

        print(
            f"{error_type:<18}"
            f"{count:>6}"
            f" ({percentage:>6.2f}%)"
        )

    # ========================================================
    # FALSE POSITIVE ANALYSIS
    # ========================================================

    fp = analysis_df[
        analysis_df["Error_Type"]
        == "False Positive"
    ].copy()

    fn = analysis_df[
        analysis_df["Error_Type"]
        == "False Negative"
    ].copy()

    tp = analysis_df[
        analysis_df["Error_Type"]
        == "True Positive"
    ].copy()

    tn = analysis_df[
        analysis_df["Error_Type"]
        == "True Negative"
    ].copy()

    # --------------------------------------------------------
    # PROBABILITY SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PROBABILITY BY ERROR TYPE"
    )

    print(
        "=" * 80
    )

    probability_summary = (
        analysis_df
        .groupby("Error_Type")[
            "Sigmoid_Probability"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reindex(
            [
                "True Negative",
                "False Positive",
                "False Negative",
                "True Positive",
            ]
        )
    )

    print(
        probability_summary.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # GROUP ERROR ANALYSIS FUNCTION
    # ========================================================

    def analyze_categorical(
        column,
    ):

        grouped = (
            analysis_df
            .groupby(column)
            .agg(
                Customers=(
                    "Actual_Churn",
                    "size",
                ),
                Actual_Churn_Rate=(
                    "Actual_Churn",
                    "mean",
                ),
                Predicted_Churn_Rate=(
                    "Predicted_Churn",
                    "mean",
                ),
                Avg_Probability=(
                    "Sigmoid_Probability",
                    "mean",
                ),
            )
        )

        grouped[
            "False_Positive_Rate"
        ] = (
            analysis_df
            .groupby(column)
            .apply(
                lambda group:
                (
                    (
                        (group["Actual_Churn"] == 0)
                        &
                        (group["Predicted_Churn"] == 1)
                    ).sum()
                    /
                    max(
                        (group["Actual_Churn"] == 0).sum(),
                        1,
                    )
                ),
                include_groups=False,
            )
        )

        grouped[
            "False_Negative_Rate"
        ] = (
            analysis_df
            .groupby(column)
            .apply(
                lambda group:
                (
                    (
                        (group["Actual_Churn"] == 1)
                        &
                        (group["Predicted_Churn"] == 0)
                    ).sum()
                    /
                    max(
                        (group["Actual_Churn"] == 1).sum(),
                        1,
                    )
                ),
                include_groups=False,
            )
        )

        return grouped.reset_index()

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    categorical_features = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines",
        "PaperlessBilling",
        "Dependents",
        "Partner",
        "PhoneService",
        "gender",
        "SeniorCitizen",
    ]

    categorical_results = []

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CATEGORICAL ERROR ANALYSIS"
    )

    print(
        "=" * 80
    )

    for feature in categorical_features:

        print(
            f"\n--- {feature} ---"
        )

        result = analyze_categorical(
            feature
        )

        result.insert(
            0,
            "Feature",
            feature,
        )

        categorical_results.append(
            result
        )

        print(
            result.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    categorical_df = pd.concat(
        categorical_results,
        ignore_index=True,
    )

    # ========================================================
    # NUMERIC ERROR ANALYSIS
    # ========================================================

    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    numeric_results = []

    print(
        "\n"
        + "=" * 80
    )

    print(
        "NUMERIC ERROR ANALYSIS"
    )

    print(
        "=" * 80
    )

    for feature in numeric_features:

        result = (
            analysis_df
            .groupby("Error_Type")[feature]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max",
                ]
            )
            .reindex(
                [
                    "True Negative",
                    "False Positive",
                    "False Negative",
                    "True Positive",
                ]
            )
        )

        print(
            f"\n--- {feature} ---"
        )

        print(
            result.to_string(
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

        for error_type in result.index:

            row = result.loc[
                error_type
            ]

            numeric_results.append(
                {
                    "Feature": feature,
                    "Error_Type": error_type,
                    "Count": row["count"],
                    "Mean": row["mean"],
                    "Median": row["median"],
                    "Min": row["min"],
                    "Max": row["max"],
                }
            )

    numeric_df = pd.DataFrame(
        numeric_results
    )

    # ========================================================
    # TENURE BINS
    # ========================================================

    analysis_df[
        "Tenure_Group"
    ] = pd.cut(
        analysis_df["tenure"],
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
    )

    tenure_analysis = (
        analyze_categorical(
            "Tenure_Group"
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TENURE ERROR ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        tenure_analysis.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # MONTHLY CHARGE BINS
    # ========================================================

    analysis_df[
        "MonthlyCharges_Group"
    ] = pd.cut(
        analysis_df[
            "MonthlyCharges"
        ],
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
    )

    monthly_analysis = (
        analyze_categorical(
            "MonthlyCharges_Group"
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MONTHLY CHARGES ERROR ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        monthly_analysis.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # TOTAL CHARGES / TENURE RELATIONSHIP
    # ========================================================

    analysis_df[
        "Average_Monthly_Spend"
    ] = np.where(
        analysis_df["tenure"] > 0,
        analysis_df["TotalCharges"]
        / analysis_df["tenure"],
        analysis_df["MonthlyCharges"],
    )

    analysis_df[
        "Charge_Ratio"
    ] = np.where(
        analysis_df["MonthlyCharges"] > 0,
        analysis_df["TotalCharges"]
        / (
            analysis_df["MonthlyCharges"]
            * np.maximum(
                analysis_df["tenure"],
                1,
            )
        ),
        0,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DERIVED NUMERIC FEATURE INVESTIGATION"
    )

    print(
        "=" * 80
    )

    derived_features = [
        "Average_Monthly_Spend",
        "Charge_Ratio",
    ]

    for feature in derived_features:

        result = (
            analysis_df
            .groupby("Error_Type")[feature]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max",
                ]
            )
            .reindex(
                [
                    "True Negative",
                    "False Positive",
                    "False Negative",
                    "True Positive",
                ]
            )
        )

        print(
            f"\n--- {feature} ---"
        )

        print(
            result.to_string(
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

    # ========================================================
    # MOST UNCERTAIN CASES
    # ========================================================

    analysis_df[
        "Distance_From_Threshold"
    ] = np.abs(
        analysis_df[
            "Sigmoid_Probability"
        ]
        - THRESHOLD
    )

    uncertain_cases = (
        analysis_df
        .sort_values(
            "Distance_From_Threshold"
        )
        .head(30)
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "30 MOST UNCERTAIN TEST CASES"
    )

    print(
        "=" * 80
    )

    print(
        uncertain_cases[
            [
                "Original_Row_Index",
                "Actual_Churn",
                "Predicted_Churn",
                "Error_Type",
                "Sigmoid_Probability",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "InternetService",
                "PaymentMethod",
                "OnlineSecurity",
                "TechSupport",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # FALSE POSITIVES CLOSE TO THRESHOLD
    # ========================================================

    fp_uncertain = (
        fp.copy()
        .assign(
            Distance_From_Threshold=lambda x:
            np.abs(
                x[
                    "Sigmoid_Probability"
                ]
                - THRESHOLD
            )
        )
        .sort_values(
            "Distance_From_Threshold"
        )
        .head(20)
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "20 CLOSEST FALSE POSITIVES "
        "TO THRESHOLD"
    )

    print(
        "=" * 80
    )

    print(
        fp_uncertain[
            [
                "Original_Row_Index",
                "Sigmoid_Probability",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "InternetService",
                "PaymentMethod",
                "OnlineSecurity",
                "TechSupport",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # FALSE NEGATIVES CLOSE TO THRESHOLD
    # ========================================================

    fn_uncertain = (
        fn.copy()
        .assign(
            Distance_From_Threshold=lambda x:
            np.abs(
                x[
                    "Sigmoid_Probability"
                ]
                - THRESHOLD
            )
        )
        .sort_values(
            "Distance_From_Threshold"
        )
        .head(20)
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "20 CLOSEST FALSE NEGATIVES "
        "TO THRESHOLD"
    )

    print(
        "=" * 80
    )

    print(
        fn_uncertain[
            [
                "Original_Row_Index",
                "Sigmoid_Probability",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "InternetService",
                "PaymentMethod",
                "OnlineSecurity",
                "TechSupport",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE FULL ERROR DATASET
    # ========================================================

    output_path = (
        REPORTS_DIR
        / "churn_error_analysis.csv"
    )

    analysis_df.to_csv(
        output_path,
        index=False,
    )

    categorical_path = (
        REPORTS_DIR
        / "churn_error_analysis_categorical.csv"
    )

    categorical_df.to_csv(
        categorical_path,
        index=False,
    )

    numeric_path = (
        REPORTS_DIR
        / "churn_error_analysis_numeric.csv"
    )

    numeric_df.to_csv(
        numeric_path,
        index=False,
    )

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
        f"Full errors: "
        f"{output_path}"
    )

    print(
        f"Categorical: "
        f"{categorical_path}"
    )

    print(
        f"Numeric: "
        f"{numeric_path}"
    )

    print(
        "\nError analysis completed successfully."
    )


if __name__ == "__main__":

    main()