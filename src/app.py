import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "customer_churn_production_model.joblib"
)
METADATA_PATH = BASE_DIR / "models" / "final_model_metadata.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PRODUCTION MODEL CONFIGURATION
# ============================================================

DEFAULT_THRESHOLD = 0.36


# ============================================================
# MODEL FEATURES
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


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    bundle = load_model()

    model = bundle["model"]

    model_features = bundle.get(
        "features",
        ALL_FEATURES,
    )

    model_threshold = float(
        bundle.get(
            "threshold",
            DEFAULT_THRESHOLD,
        )
    )

    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        model_metadata = json.load(metadata_file)

    if type(model).__name__ != model_metadata["model_type"]:
        raise ValueError("Model type does not match production metadata.")

    if list(model_features) != model_metadata["selected_features"]:
        raise ValueError("Model features do not match production metadata.")

    if abs(model_threshold - float(model_metadata["threshold"])) > 1e-9:
        raise ValueError("Model threshold does not match production metadata.")

    MODEL_NAME = model_metadata["model_name"]
    MODEL_PARAMETERS = model_metadata["parameters"]
    MODEL_METRICS = model_metadata["final_test_metrics"]
    ORIGINAL_TRAINING_SAMPLES = model_metadata["training_strategy"]["original_training_samples"]
    TRAINING_SAMPLES = model_metadata["training_strategy"]["balanced_training_samples"]
    FEATURE_COUNT = len(model_features)

    ROC_AUC = MODEL_METRICS["roc_auc"] * 100
    F1_SCORE = MODEL_METRICS["f1"] * 100
    RECALL = MODEL_METRICS["recall"] * 100
    PRECISION = MODEL_METRICS["precision"] * 100
    ACCURACY = MODEL_METRICS["accuracy"] * 100
    PR_AUC = MODEL_METRICS["pr_auc"] * 100
    BRIER_SCORE = MODEL_METRICS["brier"]

    model_loaded = True
    model_error = ""

except Exception as e:

    model = None
    model_features = ALL_FEATURES
    model_threshold = DEFAULT_THRESHOLD
    model_metadata = {}
    MODEL_NAME = "Unavailable"
    MODEL_PARAMETERS = {}
    MODEL_METRICS = {}
    ORIGINAL_TRAINING_SAMPLES = 0
    FEATURE_COUNT = len(ALL_FEATURES)
    TRAINING_SAMPLES = 0
    ROC_AUC = F1_SCORE = RECALL = PRECISION = 0.0
    ACCURACY = PR_AUC = BRIER_SCORE = 0.0

    model_loaded = False
    model_error = str(e)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(59, 130, 246, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(139, 92, 246, 0.08),
                transparent 25%
            ),
            #0b1020;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #080d19;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }


    /* ========================================================
       BRAND
       ======================================================== */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;

        border-radius: 12px;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            linear-gradient(
                135deg,
                #3b82f6,
                #8b5cf6
            );

        color: #ffffff;

        font-size: 20px;
        font-weight: 800;

        box-shadow:
            0 8px 25px rgba(59, 130, 246, 0.25);
    }

    .brand-name {
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    .brand-subtitle {
        color: #7f8aa3;
        font-size: 11px;
        margin-top: 2px;
    }


    /* ========================================================
       NAVIGATION
       ======================================================== */

    .nav-label {
        color: #657089;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 1.5rem 0 0.6rem 0;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        padding: 42px 44px;

        border-radius: 24px;
        margin-bottom: 28px;

        background:
            linear-gradient(
                135deg,
                rgba(30, 64, 175, 0.32),
                rgba(76, 29, 149, 0.25)
            ),
            rgba(17, 24, 39, 0.88);

        border: 1px solid rgba(255, 255, 255, 0.09);

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.25);
    }

    .hero::after {
        content: "";

        position: absolute;

        width: 260px;
        height: 260px;

        right: -90px;
        top: -100px;

        border-radius: 50%;

        background:
            rgba(59, 130, 246, 0.14);

        filter: blur(20px);
    }

    .hero-eyebrow {
        color: #93c5fd;

        font-size: 12px;
        font-weight: 700;

        letter-spacing: 2px;

        text-transform: uppercase;

        margin-bottom: 12px;
    }

    .hero-title {
        color: #ffffff;

        font-size: 42px;

        line-height: 1.1;

        font-weight: 800;

        letter-spacing: -1.5px;
    }

    .hero-title span {
        background:
            linear-gradient(
                90deg,
                #60a5fa,
                #a78bfa
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 760px;

        color: #aab4c8;

        font-size: 16px;

        line-height: 1.7;

        margin-top: 16px;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {
        color: #ffffff;
        font-size: 21px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .section-description {
        color: #78849b;
        font-size: 13px;
        margin-bottom: 18px;
    }


    /* ========================================================
       INPUT GROUP
       ======================================================== */

    .input-group {
        background:
            rgba(17, 24, 39, 0.72);

        border:
            1px solid rgba(255, 255, 255, 0.07);

        border-radius: 18px;

        padding: 20px 24px 8px 24px;

        margin-top: 20px;
        margin-bottom: 5px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.14);
    }

    .input-group-title {
        color: #ffffff;

        font-size: 15px;

        font-weight: 700;

        margin-bottom: 2px;
    }

    .input-group-description {
        color: #66728a;

        font-size: 11px;

        margin-bottom: 5px;
    }


    /* ========================================================
       FORM LABELS
       ======================================================== */

    label {
        color: #aab4c8 !important;

        font-size: 12px !important;

        font-weight: 600 !important;
    }


    /* ========================================================
       SELECTBOX
       ======================================================== */

    /*
       IMPORTANT:
       Only the actual selectbox is targeted here.
       We do NOT use:
           div[data-baseweb="select"] *
       because that can affect unrelated elements.
    */

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #f1f3f7 !important;

        border:
            1px solid rgba(255, 255, 255, 0.12) !important;

        border-radius: 10px !important;

        min-height: 48px !important;
    }

    /* Selected value */

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"]
    div[class*="singleValue"] {
        color: #111827 !important;

        font-weight: 500 !important;
    }

    /* Selectbox input */

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"]
    input {
        color: #111827 !important;

        -webkit-text-fill-color: #111827 !important;
    }

    /* Dropdown */

    div[data-baseweb="popover"] {
        z-index: 999999 !important;
    }

    div[role="listbox"] {
        background-color: #f1f3f7 !important;

        border-radius: 10px !important;
    }

    div[role="option"] {
        color: #111827 !important;
    }

    div[role="option"]:hover {
        background-color: #e5e7eb !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    div[data-testid="stNumberInput"] {
        color: #aab4c8 !important;
    }

    /* Actual number field only */

    div[data-testid="stNumberInput"] input {
        background-color: #f1f3f7 !important;

        color: #111827 !important;

        -webkit-text-fill-color: #111827 !important;

        border-radius: 10px !important;
    }

    /* Number input buttons */

    div[data-testid="stNumberInput"] button {
        color: #111827 !important;
    }


    /* ========================================================
       PLACEHOLDER
       ======================================================== */

    input::placeholder {
        color: #6b7280 !important;

        -webkit-text-fill-color: #6b7280 !important;
    }


    /* ========================================================
       FORM SUBMIT BUTTON
       ======================================================== */

    .stFormSubmitButton > button {

        width: 100%;

        min-height: 50px;

        border: none;

        border-radius: 12px;

        font-weight: 700;

        letter-spacing: 0.2px;

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #7c3aed
            ) !important;

        color: #ffffff !important;

        box-shadow:
            0 10px 25px
            rgba(59, 130, 246, 0.20);

        transition:
            all 0.2s ease;
    }

    .stFormSubmitButton > button:hover {

        transform: translateY(-1px);

        box-shadow:
            0 14px 30px
            rgba(59, 130, 246, 0.30);
    }


    /* ========================================================
       RESULT CARD
       ======================================================== */

    .result-card {

        padding: 32px;

        border-radius: 22px;

        background:
            linear-gradient(
                145deg,
                rgba(17, 24, 39, 0.96),
                rgba(15, 23, 42, 0.90)
            );

        border:
            1px solid rgba(255, 255, 255, 0.09);

        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.25);

        text-align: center;
    }

    .result-label {
        color: #7f8aa3;

        font-size: 11px;

        font-weight: 700;

        letter-spacing: 1.5px;

        text-transform: uppercase;
    }

    .prediction-text {
        color: #ffffff;

        font-size: 17px;

        font-weight: 700;

        margin-top: 18px;
    }

    .threshold-note {
        color: #66728a;

        font-size: 11px;

        margin-top: 10px;
    }


    /* ========================================================
       RISK BADGES
       ======================================================== */

    .risk-badge {

        display: inline-block;

        margin-top: 22px;

        padding: 8px 18px;

        border-radius: 999px;

        font-size: 12px;

        font-weight: 800;

        letter-spacing: 1px;

        text-transform: uppercase;
    }

    .risk-high {

        background:
            rgba(239, 68, 68, 0.14);

        color: #f87171;

        border:
            1px solid rgba(239, 68, 68, 0.25);
    }

    .risk-medium {

        background:
            rgba(245, 158, 11, 0.14);

        color: #fbbf24;

        border:
            1px solid rgba(245, 158, 11, 0.25);
    }

    .risk-low {

        background:
            rgba(34, 197, 94, 0.14);

        color: #4ade80;

        border:
            1px solid rgba(34, 197, 94, 0.25);
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {

        padding: 22px;

        border-radius: 18px;

        background:
            rgba(17, 24, 39, 0.78);

        border:
            1px solid rgba(255, 255, 255, 0.07);

        min-height: 115px;
    }

    .metric-title {

        color: #7f8aa3;

        font-size: 10px;

        font-weight: 700;

        letter-spacing: 1.4px;

        text-transform: uppercase;
    }

    .metric-value {

        color: #ffffff;

        font-size: 28px;

        font-weight: 800;

        margin-top: 8px;
    }

    .metric-note {

        color: #66728a;

        font-size: 11px;

        margin-top: 4px;
    }


    /* ========================================================
       FEATURE CARDS
       ======================================================== */

    .feature-card {

        padding: 18px 20px;

        border-radius: 16px;

        background:
            rgba(17, 24, 39, 0.72);

        border:
            1px solid rgba(255, 255, 255, 0.07);

        margin-bottom: 10px;
    }

    .feature-name {

        color: #ffffff;

        font-size: 13px;

        font-weight: 700;
    }

    .feature-value {

        color: #8290a8;

        font-size: 12px;

        margin-top: 4px;

        line-height: 1.5;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {

        text-align: center;

        color: #505b70;

        font-size: 11px;

        padding-top: 35px;

        margin-top: 40px;

        border-top:
            1px solid rgba(255, 255, 255, 0.06);
    }


    /* ========================================================
       STREAMLIT BRANDING
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="brand">

            <div class="brand-icon">
                ◉
            </div>

            <div>

                <div class="brand-name">
                    Churn Intelligence
                </div>

                <div class="brand-subtitle">
                    AI CUSTOMER ANALYTICS
                </div>

            </div>

        </div>
        """
    )

    st.markdown(
        '<div class="nav-label">Workspace</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Customer Prediction",
            "Model Insights",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if model_loaded:

        st.html(
            f"""
        <div style="padding:10px 4px;">

                <div style="
                    color:#64748b;
                    font-size:10px;
                    font-weight:700;
                    letter-spacing:1.2px;
                ">
                    MODEL STATUS
                </div>

                <div style="
                    color:#4ade80;
                    font-size:13px;
                    font-weight:600;
                    margin-top:7px;
                ">
                    ● Production Ready
                </div>

                <div style="
                    color:#64748b;
                    font-size:11px;
                    margin-top:5px;
                ">
                    {MODEL_NAME} · {FEATURE_COUNT} Features
                </div>

            </div>
            """
        )

    else:

        st.html(
            """
            <div style="padding:10px 4px;">

                <div style="
                    color:#64748b;
                    font-size:10px;
                    font-weight:700;
                    letter-spacing:1.2px;
                ">
                    MODEL STATUS
                </div>

                <div style="
                    color:#f87171;
                    font-size:13px;
                    font-weight:600;
                    margin-top:7px;
                ">
                    ● Model Load Error
                </div>

                <div style="
                    color:#64748b;
                    font-size:10px;
                    margin-top:5px;
                    line-height:1.5;
                ">
                    Check production model
                </div>

            </div>
            """
        )


# ============================================================
# CUSTOMER PREDICTION PAGE
# ============================================================

if page == "Customer Prediction":

    if not model_loaded:

        st.error(
            "The production model could not be loaded."
        )

        st.code(model_error)

        st.stop()


    # ========================================================
    # HERO
    # ========================================================

    st.html(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                CUSTOMER RISK ANALYSIS
            </div>

            <div class="hero-title">
                Understand your
                <span>customer.</span>
            </div>

            <div class="hero-description">
                Enter the customer's current service and subscription
                details to estimate churn probability and identify
                potential retention opportunities.
            </div>

        </div>
        """
    )


    # ========================================================
    # CUSTOMER ANALYSIS
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Customer Analysis
        </div>

        <div class="section-description">
            Provide the customer's current profile and service details.
        </div>
        """
    )


    # ========================================================
    # FORM
    # ========================================================

    with st.form(
        "customer_prediction_form"
    ):

        # ----------------------------------------------------
        # CUSTOMER PROFILE
        # ----------------------------------------------------

        st.html(
            """
            <div class="input-group">

                <div class="input-group-title">
                    👤 Customer Profile
                </div>

                <div class="input-group-description">
                    Basic customer information
                </div>

            </div>
            """
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female",
                ],
            )

        with col2:

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [
                    "No",
                    "Yes",
                ],
            )

        with col3:

            partner = st.selectbox(
                "Partner",
                [
                    "No",
                    "Yes",
                ],
            )

        with col4:

            dependents = st.selectbox(
                "Dependents",
                [
                    "No",
                    "Yes",
                ],
            )


        col1, col2 = st.columns(2)

        with col1:

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
                step=1,
            )

        with col2:

            phone_service = st.selectbox(
                "Phone Service",
                [
                    "Yes",
                    "No",
                ],
            )


        # ----------------------------------------------------
        # SUBSCRIPTION & BILLING
        # ----------------------------------------------------

        st.html(
            """
            <div class="input-group">

                <div class="input-group-title">
                    📄 Subscription & Billing
                </div>

                <div class="input-group-description">
                    Contract, payment and billing information
                </div>

            </div>
            """
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

        with col2:

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col3:

            paperless_billing = st.selectbox(
                "Paperless Billing",
                [
                    "Yes",
                    "No",
                ],
            )


        col1, col2 = st.columns(2)

        with col1:

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.00,
                step=0.50,
            )

        with col2:

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=1500.00,
                step=10.00,
            )


        # ----------------------------------------------------
        # INTERNET & SUPPORT
        # ----------------------------------------------------

        st.html(
            """
            <div class="input-group">

                <div class="input-group-title">
                    🌐 Internet & Support
                </div>

                <div class="input-group-description">
                    Internet services and customer support options
                </div>

            </div>
            """
        )

        col1, col2 = st.columns(2)

        with col1:

            internet_service = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No",
                ],
            )

        with col2:

            online_security = st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )


        col1, col2 = st.columns(2)

        with col1:

            online_backup = st.selectbox(
                "Online Backup",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

        with col2:

            device_protection = st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )


        col1, col2 = st.columns(2)

        with col1:

            tech_support = st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

        with col2:

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service",
                ],
            )


        # ----------------------------------------------------
        # STREAMING SERVICES
        # ----------------------------------------------------

        st.html(
            """
            <div class="input-group">

                <div class="input-group-title">
                    📺 Entertainment Services
                </div>

                <div class="input-group-description">
                    Streaming and entertainment services
                </div>

            </div>
            """
        )

        col1, col2 = st.columns(2)

        with col1:

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

        with col2:

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )


        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # SUBMIT
        # ----------------------------------------------------

        submitted = st.form_submit_button(
            "🔍  Analyze Customer",
            use_container_width=True,
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        customer_data = {

            "gender":
                gender,

            "SeniorCitizen":
                1 if senior_citizen == "Yes" else 0,

            "Partner":
                partner,

            "Dependents":
                dependents,

            "tenure":
                tenure,

            "PhoneService":
                phone_service,

            "MultipleLines":
                multiple_lines,

            "InternetService":
                internet_service,

            "OnlineSecurity":
                online_security,

            "OnlineBackup":
                online_backup,

            "DeviceProtection":
                device_protection,

            "TechSupport":
                tech_support,

            "StreamingTV":
                streaming_tv,

            "StreamingMovies":
                streaming_movies,

            "Contract":
                contract,

            "PaperlessBilling":
                paperless_billing,

            "PaymentMethod":
                payment_method,

            "MonthlyCharges":
                monthly_charges,

            "TotalCharges":
                total_charges,
        }


        input_df = pd.DataFrame(
            [customer_data]
        )


        # Ensure exact production feature order.

        input_df = input_df[
            model_features
        ]


        # ====================================================
        # PREDICT
        # ====================================================

        probability = float(
            model.predict_proba(
                input_df
            )[0][1]
        )

        prediction = int(
            probability >= model_threshold
        )


        # ====================================================
        # RISK LEVEL
        # ====================================================

        if probability >= 0.70:

            risk_level = "High Risk"
            risk_class = "risk-high"

        elif probability >= model_threshold:

            risk_level = "Medium Risk"
            risk_class = "risk-medium"

        else:

            risk_level = "Low Risk"
            risk_class = "risk-low"


        prediction_text = (
            "Customer Likely to Churn"
            if prediction == 1
            else
            "Customer Not Likely to Churn"
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        st.html(
            """
            <div class="section-title">
                Prediction Result
            </div>

            <div class="section-description">
                Model-generated churn assessment
            </div>
            """
        )


        col1, col2 = st.columns(
            [1.15, 0.85]
        )


        with col1:

            st.html(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        CHURN PROBABILITY
                    </div>

                    <div style="
                        color:#ffffff;
                        font-size:58px;
                        font-weight:850;
                        margin-top:14px;
                        letter-spacing:-2px;
                    ">
                        {probability * 100:.1f}%
                    </div>

                    <div class="risk-badge {risk_class}">
                        {risk_level}
                    </div>

                    <div class="prediction-text">
                        {prediction_text}
                    </div>

                    <div class="threshold-note">
                        Classification threshold:
                        {model_threshold * 100:.0f}%
                    </div>

                </div>
                """
            )


        with col2:

            st.html(
                f"""
                <div class="result-card"
                     style="text-align:left;">

                    <div class="result-label">
                        ASSESSMENT
                    </div>

                    <div style="
                        color:#ffffff;
                        font-size:20px;
                        font-weight:750;
                        margin-top:16px;
                    ">
                        {risk_level}
                    </div>

                    <div style="
                        color:#8290a8;
                        font-size:13px;
                        line-height:1.8;
                        margin-top:12px;
                    ">
                        The customer's estimated churn
                        probability is
                        <b style="color:#ffffff;">
                            {probability * 100:.1f}%
                        </b>.
                        The production classification
                        threshold is
                        <b style="color:#ffffff;">
                            {model_threshold * 100:.0f}%
                        </b>.
                    </div>

                </div>
                """
            )


        # ====================================================
        # PROFILE SUMMARY
        # ====================================================

        st.markdown(
            "<br><br>",
            unsafe_allow_html=True,
        )

        st.html(
            """
            <div class="section-title">
                Customer Profile Summary
            </div>

            <div class="section-description">
                Inputs used for this prediction
            </div>
            """
        )


        summary_items = [

            (
                "Contract",
                contract,
            ),

            (
                "Tenure",
                f"{tenure} months",
            ),

            (
                "Internet Service",
                internet_service,
            ),

            (
                "Payment Method",
                payment_method,
            ),

            (
                "Monthly Charges",
                f"${monthly_charges:,.2f}",
            ),

            (
                "Total Charges",
                f"${total_charges:,.2f}",
            ),

            (
                "Online Security",
                online_security,
            ),

            (
                "Tech Support",
                tech_support,
            ),

        ]


        cols = st.columns(4)


        for index, (
            label,
            value,
        ) in enumerate(summary_items):

            with cols[index % 4]:

                st.html(
                    f"""
                    <div class="feature-card">

                        <div class="feature-name">
                            {label}
                        </div>

                        <div class="feature-value">
                            {value}
                        </div>

                    </div>
                    """
                )


        # ====================================================
        # RETENTION INTELLIGENCE
        # ====================================================

        st.markdown(
            "<br><br>",
            unsafe_allow_html=True,
        )

        st.html(
            """
            <div class="section-title">
                Retention Intelligence
            </div>

            <div class="section-description">
                Profile-based opportunities associated with
                the current customer configuration
            </div>
            """
        )


        if prediction == 0:

            st.html(
                """
                <div style="
                    padding:26px 28px;

                    border-radius:18px;

                    background:
                        linear-gradient(
                            135deg,
                            rgba(34,197,94,0.08),
                            rgba(17,24,39,0.72)
                        );

                    border:
                        1px solid
                        rgba(34,197,94,0.18);
                ">

                    <div style="
                        color:#4ade80;
                        font-size:15px;
                        font-weight:750;
                    ">
                        ✓ No Immediate Retention Intervention
                    </div>

                    <div style="
                        color:#8290a8;
                        font-size:12px;
                        line-height:1.7;
                        margin-top:10px;
                    ">
                        The customer is currently below the
                        churn classification threshold.
                        Continue monitoring the profile and
                        consider targeted engagement when
                        future behavior indicates a retention
                        opportunity.
                    </div>

                </div>
                """
            )

        else:

            recommendations = []


            if contract == "Month-to-month":

                recommendations.append(
                    (
                        "📄",
                        "Longer-Term Contract",
                        "Consider a targeted one-year or two-year "
                        "contract offer where appropriate.",
                    )
                )


            if online_security == "No":

                recommendations.append(
                    (
                        "🔐",
                        "Security Service",
                        "Consider an Online Security trial or "
                        "targeted security bundle.",
                    )
                )


            if online_backup == "No":

                recommendations.append(
                    (
                        "☁️",
                        "Backup Service",
                        "Consider an Online Backup promotion "
                        "or bundled service.",
                    )
                )


            if device_protection == "No":

                recommendations.append(
                    (
                        "🛡️",
                        "Device Protection",
                        "Consider a targeted device protection "
                        "offer or service bundle.",
                    )
                )


            if tech_support == "No":

                recommendations.append(
                    (
                        "🛠️",
                        "Technical Support",
                        "Consider a targeted technical support "
                        "package or trial.",
                    )
                )


            if payment_method == "Electronic check":

                recommendations.append(
                    (
                        "💳",
                        "Payment Experience",
                        "Consider promoting automatic payment "
                        "methods where suitable.",
                    )
                )


            if tenure <= 12:

                recommendations.append(
                    (
                        "🤝",
                        "Early-Tenure Engagement",
                        "Consider proactive onboarding and "
                        "early-lifecycle engagement.",
                    )
                )


            if monthly_charges >= 70:

                recommendations.append(
                    (
                        "💰",
                        "Plan Value Review",
                        "Consider reviewing plan value and "
                        "available package options.",
                    )
                )


            if not recommendations:

                recommendations.append(
                    (
                        "📊",
                        "Profile Review",
                        "Review the customer's current services "
                        "and identify relevant engagement options.",
                    )
                )


            for i in range(
                0,
                len(recommendations),
                2,
            ):

                cols = st.columns(2)


                for j in range(2):

                    index = i + j

                    if index >= len(
                        recommendations
                    ):
                        continue


                    icon, title, description = (
                        recommendations[index]
                    )


                    with cols[j]:

                        st.html(
                            f"""
                            <div class="feature-card"
                                 style="
                                    min-height:145px;
                                 ">

                                <div style="
                                    font-size:24px;
                                ">
                                    {icon}
                                </div>

                                <div class="feature-name"
                                     style="
                                        margin-top:10px;
                                     ">
                                    {title}
                                </div>

                                <div class="feature-value"
                                     style="
                                        margin-top:8px;
                                     ">
                                    {description}
                                </div>

                            </div>
                            """
                        )


        st.html(
            """
            <div style="
                color:#59667c;
                font-size:10px;
                line-height:1.6;
                margin-top:12px;
            ">
                <b>Interpretation:</b>
                These are profile-level signals associated
                with churn in the training data and model
                interpretation. They should not be treated
                as causal explanations for an individual
                customer's prediction.
            </div>
            """
        )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.html(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                MODEL INTELLIGENCE
            </div>

            <div class="hero-title">
                Model
                <span>Insights.</span>
            </div>

            <div class="hero-description">
                Production model configuration and
                held-out evaluation metrics.
            </div>

        </div>
        """
    )


    # ========================================================
    # METRICS
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Production Performance
        </div>

        <div class="section-description">
            Evaluation on the untouched held-out test set
        </div>
        """
    )


    metrics = [

        (
            "ROC-AUC",
            f"{ROC_AUC:.2f}%",
            "Held-out test set",
        ),

        (
            "F1 Score",
            f"{F1_SCORE:.2f}%",
            "Churn class",
        ),

        (
            "Recall",
            f"{RECALL:.2f}%",
            "Churn class",
        ),

        (
            "Precision",
            f"{PRECISION:.2f}%",
            "Churn class",
        ),

    ]


    cols = st.columns(4)


    for column, metric in zip(
        cols,
        metrics,
    ):

        label, value, note = metric

        with column:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-note">
                        {note}
                    </div>

                </div>
                """
            )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    st.markdown(
        "<br><br>",
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div class="section-title">
            Model Configuration
        </div>

        <div class="section-description">
            Final production configuration
        </div>
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        st.html(
            f"""
            <div class="result-card"
                 style="text-align:left;">

                <div class="result-label">
                    {MODEL_NAME.upper()} CONFIGURATION
                </div>

                <div style="
                    color:#aab4c8;
                    font-size:13px;
                    line-height:2;
                    margin-top:20px;
                ">

                    <b style="color:#ffffff;">
                        Iterations
                    </b>
                    : {MODEL_PARAMETERS.get('iterations', '—')}

                    <br>

                    <b style="color:#ffffff;">
                        Depth
                    </b>
                    : {MODEL_PARAMETERS.get('depth', '—')}

                    <br>

                    <b style="color:#ffffff;">
                        Learning Rate
                    </b>
                    : {MODEL_PARAMETERS.get('learning_rate', '—')}

                    <br>

                    <b style="color:#ffffff;">
                        L2 Regularization
                    </b>
                    : {MODEL_PARAMETERS.get('l2_leaf_reg', '—')}

                    <br>

                    <b style="color:#ffffff;">
                        Loss Function
                    </b>
                    : Logloss

                    <br>

                    <b style="color:#ffffff;">
                        Random Seed
                    </b>
                    : {MODEL_PARAMETERS.get('random_seed', 42)}

                </div>

            </div>
            """
        )


    with col2:

        st.html(
            f"""
            <div class="result-card"
                 style="text-align:left;">

                <div class="result-label">
                    PREDICTION CONFIGURATION
                </div>

                <div style="
                    color:#aab4c8;
                    font-size:13px;
                    line-height:2;
                    margin-top:20px;
                ">

                    <b style="color:#ffffff;">
                        Features
                    </b>
                    : {FEATURE_COUNT}

                    <br>

                    <b style="color:#ffffff;">
                        Classification Threshold
                    </b>
                    : {model_threshold:.2f}

                    <br>

                    <b style="color:#ffffff;">
                        Original Training Rows
                    </b>
                    : {ORIGINAL_TRAINING_SAMPLES:,}

                    <br>

                    <b style="color:#ffffff;">
                        Balanced Fit Samples
                    </b>
                    : {TRAINING_SAMPLES:,}

                    <br>

                    <b style="color:#ffffff;">
                        Model
                    </b>
                    : {MODEL_NAME}

                    <br>

                    <b style="color:#ffffff;">
                        Training Distribution
                    </b>
                    : 50/50 Random Oversampling

                </div>

            </div>
            """
        )


    # ========================================================
    # FEATURES
    # ========================================================

    st.markdown(
        "<br><br>",
        unsafe_allow_html=True,
    )

    st.html(
        f"""
        <div class="section-title">
            Production Features
        </div>

        <div class="section-description">
            All production predictors in the saved model bundle
        </div>
        """
    )


    for i in range(
        0,
        len(model_features),
        4,
    ):

        cols = st.columns(4)

        for column, feature in zip(
            cols,
            model_features[i:i + 4],
        ):

            with column:

                feature_type = (
                    "Numerical"
                    if feature in [
                        "SeniorCitizen",
                        "tenure",
                        "MonthlyCharges",
                        "TotalCharges",
                    ]
                    else
                    "Categorical"
                )

                st.html(
                    f"""
                    <div class="feature-card">

                        <div class="feature-name">
                            {feature}
                        </div>

                        <div class="feature-value">
                            {feature_type}
                        </div>

                    </div>
                    """
                )


    # ========================================================
    # TEST RESULTS
    # ========================================================

    st.markdown(
        "<br><br>",
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div class="section-title">
            Evaluation Summary
        </div>

        <div class="section-description">
            Untouched test-set results
        </div>
        """
    )


    st.html(
        f"""
        <div class="result-card"
             style="text-align:left;">

            <div style="
                color:#aab4c8;
                font-size:13px;
                line-height:2;
            ">

                <b style="color:#ffffff;">
                    Accuracy
                </b>
                : {ACCURACY:.2f}%

                <br>

                <b style="color:#ffffff;">
                    Precision
                </b>
                : {PRECISION:.2f}%

                <br>

                <b style="color:#ffffff;">
                    Recall
                </b>
                : {RECALL:.2f}%

                <br>

                <b style="color:#ffffff;">
                    F1
                </b>
                : {F1_SCORE:.2f}%

                <br>

                <b style="color:#ffffff;">
                    ROC-AUC
                </b>
                : {ROC_AUC:.2f}%

                <br>

                <b style="color:#ffffff;">
                    PR-AUC
                </b>
                : {PR_AUC:.2f}%

                <br>

                <b style="color:#ffffff;">
                    Brier Score
                </b>
                : {BRIER_SCORE:.3f} (lower is better)

                <br><br>

                <b style="color:#ffffff;">
                    Confusion Matrix
                </b>

                <br>

                TN = {MODEL_METRICS.get('true_negative', '—')}
                &nbsp; | &nbsp;

                FP = {MODEL_METRICS.get('false_positive', '—')}
                &nbsp; | &nbsp;

                FN = {MODEL_METRICS.get('false_negative', '—')}
                &nbsp; | &nbsp;

                TP = {MODEL_METRICS.get('true_positive', '—')}

            </div>

        </div>
        """
    )


# ============================================================
# ABOUT
# ============================================================

else:

    st.html(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                ABOUT THE PROJECT
            </div>

            <div class="hero-title">
                Customer Churn
                <span>Intelligence.</span>
            </div>

            <div class="hero-description">
                A machine-learning based customer churn
                prediction system designed to estimate churn
                probability and support targeted customer
                retention analysis.
            </div>

        </div>
        """
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.html(
            """
            <div class="result-card"
                 style="text-align:left;">

                <div style="font-size:25px;">
                    ◈
                </div>

                <div style="
                    color:#ffffff;
                    font-size:16px;
                    font-weight:700;
                    margin-top:14px;
                ">
                    Machine Learning
                </div>

                <div style="
                    color:#8290a8;
                    font-size:12px;
                    line-height:1.7;
                    margin-top:8px;
                ">
                    Production churn prediction using
                    CatBoost with native categorical
                    feature handling.
                </div>

            </div>
            """
        )


    with col2:

        st.html(
            """
            <div class="result-card"
                 style="text-align:left;">

                <div style="font-size:25px;">
                    ◎
                </div>

                <div style="
                    color:#ffffff;
                    font-size:16px;
                    font-weight:700;
                    margin-top:14px;
                ">
                    Feature Selection
                </div>

                <div style="
                    color:#8290a8;
                    font-size:12px;
                    line-height:1.7;
                    margin-top:8px;
                ">
                    Candidate features were evaluated
                    using leakage-safe validation before
                    the final production configuration.
                </div>

            </div>
            """
        )


    with col3:

        st.html(
            """
            <div class="result-card"
                 style="text-align:left;">

                <div style="font-size:25px;">
                    ✦
                </div>

                <div style="
                    color:#ffffff;
                    font-size:16px;
                    font-weight:700;
                    margin-top:14px;
                ">
                    Retention Intelligence
                </div>

                <div style="
                    color:#8290a8;
                    font-size:12px;
                    line-height:1.7;
                    margin-top:8px;
                ">
                    The application extends churn prediction
                    with profile-based retention opportunities
                    for customers identified as elevated risk.
                </div>

            </div>
            """
        )


    st.markdown(
        "<br><br>",
        unsafe_allow_html=True,
    )


    st.html(
        """
        <div class="section-title">
            System Overview
        </div>

        <div class="section-description">
            From customer data to churn intelligence
        </div>
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        st.html(
            """
            <div class="result-card"
                 style="text-align:left;">

                <div class="result-label">
                    PREDICTION PIPELINE
                </div>

                <div style="
                    color:#aab4c8;
                    font-size:13px;
                    line-height:1.9;
                    margin-top:18px;
                ">

                    Customer profile
                    <br>↓<br>
                    19 predictive features
                    <br>↓<br>
                    CatBoost production model
                    <br>↓<br>
                    Churn probability
                    <br>↓<br>
                    Risk classification

                </div>

            </div>
            """
        )


    with col2:

        st.html(
            """
            <div class="result-card"
                 style="text-align:left;">

                <div class="result-label">
                    TECHNOLOGY STACK
                </div>

                <div style="
                    color:#aab4c8;
                    font-size:13px;
                    line-height:2;
                    margin-top:18px;
                ">

                    Python
                    <br>
                    Pandas
                    <br>
                    CatBoost
                    <br>
                    Joblib
                    <br>
                    Streamlit
                    <br>
                    Scikit-learn

                </div>

            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        Customer Churn Intelligence ·
        CatBoost Prediction System

        <br>

        Built for customer risk analysis
        and retention intelligence

    </div>
    """
)
