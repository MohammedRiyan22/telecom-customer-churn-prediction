# Candidate and selected features for the churn prediction project

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

SELECTED_FEATURES = [
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

REMOVED_FEATURES = [
    "SeniorCitizen",
    "Partner",
    "PhoneService",
    "StreamingTV",
]