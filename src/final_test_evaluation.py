import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.utils import resample

from xgboost import XGBClassifier
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

print("\n" + "=" * 90)
print("FINAL TEST SET EVALUATION")
print("=" * 90)

print(f"\nComplete dataset shape: {df.shape}")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df[SELECTED_FEATURES].copy()
y = df["Churn"].copy()


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
# 6. ORIGINAL STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


print("\nOriginal split:")
print(f"  Training rows : {len(X_train)}")
print(f"  Test rows     : {len(X_test)}")


print("\nOriginal training distribution:")

print(
    y_train.value_counts().sort_index().to_string()
)

print("\nOriginal test distribution:")

print(
    y_test.value_counts().sort_index().to_string()
)


# ============================================================
# 7. BALANCE TRAINING DATA ONLY
# ============================================================

train_data = X_train.copy()

train_data["Churn"] = y_train.values


majority = train_data[
    train_data["Churn"] == 0
]

minority = train_data[
    train_data["Churn"] == 1
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


X_train_balanced = balanced_train.drop(
    columns=["Churn"]
)

y_train_balanced = balanced_train["Churn"]


print("\nBalanced training distribution:")

print(
    y_train_balanced.value_counts().sort_index().to_string()
)


# ============================================================
# 8. SELECTED FINAL CANDIDATES
# ============================================================

models = {

    "CatBoost Shallow": {
        "model": CatBoostClassifier(
            iterations=250,
            depth=4,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False
        ),
        "threshold": 0.60,
        "catboost": True
    },


    "XGBoost Lower LR": {
        "model": XGBClassifier(
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
        "threshold": 0.55,
        "catboost": False
    },


    "Gradient Boosting 150": {
        "model": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        ),
        "threshold": 0.55,
        "catboost": False
    },


    "XGBoost Base": {
        "model": XGBClassifier(
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
        "threshold": 0.55,
        "catboost": False
    }
}


# ============================================================
# 9. CATBOOST CATEGORICAL INDICES
# ============================================================

categorical_indices = [
    X.columns.get_loc(feature)
    for feature in categorical_features
]


# ============================================================
# 10. RESULTS
# ============================================================

results = []


# ============================================================
# 11. TRAIN AND TEST EACH MODEL
# ============================================================

for model_name, config in models.items():

    print("\n")
    print("=" * 90)
    print(f"Evaluating: {model_name}")
    print("=" * 90)

    model = config["model"]
    threshold = config["threshold"]


    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    if config["catboost"]:

        model.fit(
            X_train_balanced,
            y_train_balanced,
            cat_features=categorical_indices
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]


    # --------------------------------------------------------
    # Other models
    # --------------------------------------------------------

    else:

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    create_preprocessor()
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train_balanced,
            y_train_balanced
        )

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]


    # --------------------------------------------------------
    # Apply selected threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(f"\nThreshold : {threshold:.2f}")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix Details:")

    print(f"  True Negatives  : {tn}")
    print(f"  False Positives : {fp}")
    print(f"  False Negatives : {fn}")
    print(f"  True Positives   : {tp}")


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


# ============================================================
# 12. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    by="F1",
    ascending=False
).reset_index(drop=True)


print("\n\n")
print("=" * 110)
print("FINAL TEST SET COMPARISON")
print("=" * 110)


print(
    results_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)


print("=" * 110)


# ============================================================
# 13. TEST SET INFORMATION
# ============================================================

print("\nTest set information:")
print(f"  Total test samples : {len(X_test)}")
print(f"  Actual No Churn    : {(y_test == 0).sum()}")
print(f"  Actual Churn       : {(y_test == 1).sum()}")


print("\nImportant:")
print("  - Test set was untouched during training.")
print("  - Test set was not used during feature selection.")
print("  - Test set was not used during model tuning.")
print("  - Test set was not used during threshold optimization.")
print("  - Final metrics above are based on the untouched test set.")


print("\nFinal test evaluation completed successfully.")