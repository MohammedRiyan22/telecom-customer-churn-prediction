import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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
# 4. HYPERPARAMETER CONFIGURATIONS
# ============================================================

models = {

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    "Logistic Regression C=0.1": LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42
    ),

    "Logistic Regression C=1.0": LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ),

    "Logistic Regression C=10.0": LogisticRegression(
        C=10.0,
        max_iter=1000,
        random_state=42
    ),


    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    "Gradient Boosting 100": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ),

    "Gradient Boosting 150": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ),

    "Gradient Boosting 200": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ),

    "Gradient Boosting Depth3": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

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

    "XGBoost Shallow": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ),

    "XGBoost More Trees": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
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


    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    "CatBoost Base": CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
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

    "CatBoost More Trees": CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
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
# 7. RESULTS
# ============================================================

results = []


# ============================================================
# 8. MODEL TUNING
# ============================================================

for model_name, model in models.items():

    print(f"\nRunning: {model_name}")

    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": []
    }


    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        # ----------------------------------------------------
        # Split fold
        # ----------------------------------------------------

        X_train_fold = X.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()

        X_valid_fold = X.iloc[valid_idx].copy()
        y_valid_fold = y.iloc[valid_idx].copy()


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

        if isinstance(model, CatBoostClassifier):

            fitted_model = clone(model)

            fitted_model.fit(
                X_balanced,
                y_balanced,
                cat_features=categorical_indices
            )

            predictions = fitted_model.predict(
                X_valid_fold
            ).astype(int).ravel()

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

            predictions = pipeline.predict(
                X_valid_fold
            )

            probabilities = pipeline.predict_proba(
                X_valid_fold
            )[:, 1]


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        fold_metrics["accuracy"].append(
            accuracy_score(
                y_valid_fold,
                predictions
            )
        )

        fold_metrics["precision"].append(
            precision_score(
                y_valid_fold,
                predictions,
                zero_division=0
            )
        )

        fold_metrics["recall"].append(
            recall_score(
                y_valid_fold,
                predictions,
                zero_division=0
            )
        )

        fold_metrics["f1"].append(
            f1_score(
                y_valid_fold,
                predictions,
                zero_division=0
            )
        )

        fold_metrics["roc_auc"].append(
            roc_auc_score(
                y_valid_fold,
                probabilities
            )
        )


    # --------------------------------------------------------
    # Mean CV metrics
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": np.mean(
            fold_metrics["accuracy"]
        ),
        "Precision": np.mean(
            fold_metrics["precision"]
        ),
        "Recall": np.mean(
            fold_metrics["recall"]
        ),
        "F1": np.mean(
            fold_metrics["f1"]
        ),
        "ROC-AUC": np.mean(
            fold_metrics["roc_auc"]
        )
    })


# ============================================================
# 9. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)


print("\n")
print("=" * 90)
print("TUNED MODEL RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)

print("=" * 90)

print("\nValidation methodology:")
print("  - 5-fold stratified cross-validation")
print("  - Original dataset used")
print("  - Oversampling performed ONLY inside training folds")
print("  - Validation folds remained untouched")
print("  - Test set was not used")
print("  - Random state: 42")

print("\nTuning completed successfully.")