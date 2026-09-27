# Telecom Customer Churn Prediction

An end-to-end machine-learning project that estimates telecom customer churn risk. It includes a CatBoost model and a Streamlit dashboard for single-customer predictions, model insights, and profile-based retention suggestions.

> **Use responsibly:** predictions are estimates from a sample dataset. Retention suggestions are profile-based rules, not causal explanations or guaranteed outcomes.

## Dashboard

The Streamlit app provides:

- Churn probability, risk band, and threshold-based prediction
- Model performance and configuration details
- Profile signals associated with the prediction
- Rule-based retention opportunities for higher-risk profiles

## Production model

The app loads `models/customer_churn_production_model.joblib`; its expected model type, feature order, threshold, and evaluation metrics are recorded in `models/final_model_metadata.json`.

| Held-out metric | Result |
| --- | ---: |
| ROC-AUC | 0.8717 |
| PR-AUC | 0.7150 |
| Accuracy | 79.63% |
| Precision | 59.60% |
| Recall | 72.19% |
| F1 score | 0.6530 |
| Brier score | 0.1245 |

Metrics use the saved artifact and a stratified 20% test split of 1,409 rows. The production threshold is 0.36 and was selected from out-of-fold predictions on the training split. The model was fit on 8,278 training rows after random oversampling.

## Run locally

Python 3.13 is the version used to verify the saved model artifact.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run src/app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The application checks that the model, features, and threshold match the metadata before enabling predictions.

## Dataset and evaluation

The project uses IBM's public Telco Customer Churn sample dataset. The data files are intentionally not included in this repository. Download the CSV from [IBM's sample dataset](https://github.com/IBM/watsonx-ai-samples/blob/master/cpd4.5/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv) and save it as:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

From the project root, recreate the cleaned dataset and training/test split, then verify the saved artifact:

```powershell
python src/data_cleaning.py
python src/data_split.py
python src/verify_final_model.py
```

The split script oversamples the training data only. The verifier compares the artifact's predictions on the saved test split with the metrics in the metadata.

## Project layout

```text
src/
  app.py                  Streamlit dashboard
  data_cleaning.py        Raw-data cleaning
  data_split.py           Stratified split and training-only oversampling
  verify_final_model.py  Artifact, metadata, and held-out metric checks
  package_demo.py         Builds a standalone demo folder and ZIP
models/
  customer_churn_production_model.joblib
  final_model_metadata.json
requirements.txt
README.md
```

The other scripts under `src/` contain exploratory analysis and model-comparison workflows.

## GitHub contents

The production model and its metadata are intended to be committed so the app works after cloning. Local datasets, generated reports, archived experiments, IDE settings, virtual environments, and the duplicate standalone demo package are ignored by Git.

## License

No license has been specified for this project. Add a `LICENSE` file before granting permission for others to reuse or redistribute the code. Check the dataset's terms separately.
