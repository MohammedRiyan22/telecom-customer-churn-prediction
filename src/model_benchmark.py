import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from feature_config import SELECTED_FEATURES


# ============================================================
# 1. PATH
# ============================================================

DATA_PATH = "data/processed/cleaned_churn.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 75)
print("MODEL BENCHMARK - FOLD-WISE OVERSAMPLING")
print("=" * 75)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df[SELECTED_FEATURES].copy()
y = df["Churn"].copy()


print(f"Selected features: {len(SELECTED_FEATURES)}")

print("\nSelected features:")
for feature in SELECTED_FEATURES:
    print(f"  - {feature}")


# ============================================================
# 4. FEATURE TYPES
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
# 5. PREPROCESSOR
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
# 6. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=1
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
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

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        verbosity=-1,
        random_state=42,
        n_jobs=1
    ),
}


# ============================================================
# 7. CATBOOST MODEL
# ============================================================

catboost_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False
)


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. METRIC STORAGE
# ============================================================

results = []


# ============================================================
# 10. STANDARD SKLEARN / XGB / LIGHTGBM MODELS
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
        # Oversample ONLY the training fold
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
        # Create fresh preprocessing pipeline
        # ----------------------------------------------------

        preprocessor = create_preprocessor()

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    preprocessor
                ),
                (
                    "model",
                    clone(model)
                )
            ]
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(
            X_balanced,
            y_balanced
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

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
    # Store mean metrics
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
# 11. CATBOOST
# ============================================================

print("\nRunning: CatBoost")


# CatBoost categorical column positions
categorical_indices = [
    X.columns.get_loc(feature)
    for feature in categorical_features
]


catboost_metrics = {
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

    # --------------------------------------------------------
    # Split fold
    # --------------------------------------------------------

    X_train_fold = X.iloc[train_idx].copy()
    y_train_fold = y.iloc[train_idx].copy()

    X_valid_fold = X.iloc[valid_idx].copy()
    y_valid_fold = y.iloc[valid_idx].copy()


    # --------------------------------------------------------
    # Oversample ONLY training fold
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Fresh CatBoost model for each fold
    # --------------------------------------------------------

    model = clone(catboost_model)


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_balanced,
        y_balanced,
        cat_features=categorical_indices
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_valid_fold
    ).astype(int).ravel()

    probabilities = model.predict_proba(
        X_valid_fold
    )[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    catboost_metrics["accuracy"].append(
        accuracy_score(
            y_valid_fold,
            predictions
        )
    )

    catboost_metrics["precision"].append(
        precision_score(
            y_valid_fold,
            predictions,
            zero_division=0
        )
    )

    catboost_metrics["recall"].append(
        recall_score(
            y_valid_fold,
            predictions,
            zero_division=0
        )
    )

    catboost_metrics["f1"].append(
        f1_score(
            y_valid_fold,
            predictions,
            zero_division=0
        )
    )

    catboost_metrics["roc_auc"].append(
        roc_auc_score(
            y_valid_fold,
            probabilities
        )
    )


results.append({
    "Model": "CatBoost",
    "Accuracy": np.mean(
        catboost_metrics["accuracy"]
    ),
    "Precision": np.mean(
        catboost_metrics["precision"]
    ),
    "Recall": np.mean(
        catboost_metrics["recall"]
    ),
    "F1": np.mean(
        catboost_metrics["f1"]
    ),
    "ROC-AUC": np.mean(
        catboost_metrics["roc_auc"]
    )
})


# ============================================================
# 12. RESULTS
# ============================================================

results_df = pd.DataFrame(results)


# Sort by ROC-AUC
results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("MODEL BENCHMARK RESULTS")
print("=" * 75)

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

print("=" * 75)


# ============================================================
# 14. VALIDATION METHODOLOGY
# ============================================================

print("\nValidation methodology:")
print("  - 5-fold stratified cross-validation")
print("  - Original training distribution used for CV")
print("  - Minority class oversampled ONLY inside each training fold")
print("  - Validation folds remained untouched")
print("  - Test set was not used")
print("  - Random state: 42")

print("\nBenchmark completed successfully.")