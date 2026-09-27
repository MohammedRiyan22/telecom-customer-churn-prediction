import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.base import clone
from sklearn.utils import resample

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from feature_config import SELECTED_FEATURES


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/cleaned_churn.csv"

df = pd.read_csv(DATA_PATH)

X = df[SELECTED_FEATURES].copy()
y = df["Churn"].copy()


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
    "DeviceProtection",
    "TechSupport",
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
# 3. PREPROCESSOR
# ============================================================

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ],
        remainder="drop"
    )


# ============================================================
# 4. FINAL CANDIDATE MODELS
# ============================================================

models = {

    "Gradient Boosting 150": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ),

    "CatBoost Shallow": CatBoostClassifier(
        iterations=250,
        depth=4,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    ),

    "XGBoost Lower LR": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ),

    "XGBoost Base": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ),
}


# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. CATBOOST CATEGORICAL INDICES
# ============================================================

categorical_indices = [
    X.columns.get_loc(feature)
    for feature in categorical_features
]


# ============================================================
# 7. GENERATE OOF PREDICTIONS
# ============================================================

oof_predictions = {
    model_name: np.zeros(len(X))
    for model_name in models
}


for model_name, model in models.items():

    print(
        f"\nGenerating OOF predictions: {model_name}"
    )

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print(f"  Fold {fold}/5")

        X_train_fold = X.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()

        X_valid_fold = X.iloc[valid_idx].copy()


        # ----------------------------------------------------
        # Oversample ONLY training fold
        # ----------------------------------------------------

        train_fold = X_train_fold.copy()

        train_fold["Churn"] = y_train_fold.values

        majority = train_fold[
            train_fold["Churn"] == 0
        ]

        minority = train_fold[
            train_fold["Churn"] == 1
        ]

        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )

        balanced_train = pd.concat(
            [
                majority,
                minority_upsampled
            ],
            axis=0
        )

        balanced_train = balanced_train.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        X_balanced = balanced_train.drop(
            columns=["Churn"]
        )

        y_balanced = balanced_train["Churn"]


        # ----------------------------------------------------
        # CatBoost
        # ----------------------------------------------------

        if isinstance(
            model,
            CatBoostClassifier
        ):

            fitted_model = clone(model)

            fitted_model.fit(
                X_balanced,
                y_balanced,
                cat_features=categorical_indices
            )

            probabilities = fitted_model.predict_proba(
                X_valid_fold
            )[:, 1]


        # ----------------------------------------------------
        # Other models
        # ----------------------------------------------------

        else:

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessing",
                        create_preprocessor()
                    ),
                    (
                        "model",
                        clone(model)
                    )
                ]
            )

            pipeline.fit(
                X_balanced,
                y_balanced
            )

            probabilities = pipeline.predict_proba(
                X_valid_fold
            )[:, 1]


        oof_predictions[
            model_name
        ][valid_idx] = probabilities


# ============================================================
# 8. FINE-GRAINED THRESHOLDS
# ============================================================

thresholds = np.arange(
    0.30,
    0.801,
    0.02
)


# ============================================================
# 9. ANALYZE EACH MODEL
# ============================================================

for model_name, probabilities in oof_predictions.items():

    print("\n")
    print("=" * 100)
    print(
        f"FINE THRESHOLD ANALYSIS: {model_name}"
    )
    print("=" * 100)

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        accuracy = accuracy_score(
            y,
            predictions
        )

        predicted_churn_rate = (
            predictions.mean()
        )

        threshold_results.append({
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Predicted_Churn_Rate":
                predicted_churn_rate
        })


    threshold_df = pd.DataFrame(
        threshold_results
    )


    print(
        threshold_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
                "Predicted_Churn_Rate":
                    "{:.2%}".format
            }
        )
    )


    # ========================================================
    # BEST F1
    # ========================================================

    best_f1 = threshold_df.loc[
        threshold_df["F1"].idxmax()
    ]

    print("\nBest F1 threshold:")

    print(
        f"  Threshold : "
        f"{best_f1['Threshold']:.2f}"
    )

    print(
        f"  Precision : "
        f"{best_f1['Precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{best_f1['Recall']:.4f}"
    )

    print(
        f"  F1        : "
        f"{best_f1['F1']:.4f}"
    )


    # ========================================================
    # PRECISION >= 60%
    # ========================================================

    precision_60 = threshold_df[
        threshold_df["Precision"] >= 0.60
    ]

    if not precision_60.empty:

        best_precision_60 = precision_60.loc[
            precision_60["Recall"].idxmax()
        ]

        print("\nBest recall while precision >= 60%:")

        print(
            f"  Threshold : "
            f"{best_precision_60['Threshold']:.2f}"
        )

        print(
            f"  Precision : "
            f"{best_precision_60['Precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{best_precision_60['Recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{best_precision_60['F1']:.4f}"
        )

    else:

        print(
            "\nNo threshold reached 60% precision."
        )


    # ========================================================
    # PRECISION >= 65%
    # ========================================================

    precision_65 = threshold_df[
        threshold_df["Precision"] >= 0.65
    ]

    if not precision_65.empty:

        best_precision_65 = precision_65.loc[
            precision_65["Recall"].idxmax()
        ]

        print("\nBest recall while precision >= 65%:")

        print(
            f"  Threshold : "
            f"{best_precision_65['Threshold']:.2f}"
        )

        print(
            f"  Precision : "
            f"{best_precision_65['Precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{best_precision_65['Recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{best_precision_65['F1']:.4f}"
        )

    else:

        print(
            "\nNo threshold reached 65% precision."
        )


    # ========================================================
    # PRECISION >= 70%
    # ========================================================

    precision_70 = threshold_df[
        threshold_df["Precision"] >= 0.70
    ]

    if not precision_70.empty:

        best_precision_70 = precision_70.loc[
            precision_70["Recall"].idxmax()
        ]

        print("\nBest recall while precision >= 70%:")

        print(
            f"  Threshold : "
            f"{best_precision_70['Threshold']:.2f}"
        )

        print(
            f"  Precision : "
            f"{best_precision_70['Precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{best_precision_70['Recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{best_precision_70['F1']:.4f}"
        )

    else:

        print(
            "\nNo threshold reached 70% precision."
        )


print("\n")
print("=" * 100)
print("FINE-GRAINED THRESHOLD ANALYSIS COMPLETED")
print("=" * 100)

print(
    "\nThe test set was NOT used."
)