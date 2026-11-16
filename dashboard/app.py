
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import sys
import os

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_DIR)

from src.feature_extraction import extract_features

st.set_page_config(
    page_title="PhishGuard-PS",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# 2. CYBERPUNK GLASSMORPHISM DESIGN
# ==========================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background:
        radial-gradient(
            ellipse at 10% 10%,
            rgba(0, 225, 255, 0.13),
            transparent 38%
        ),
        radial-gradient(
            ellipse at 90% 20%,
            rgba(157, 78, 221, 0.16),
            transparent 38%
        ),
        radial-gradient(
            ellipse at 50% 100%,
            rgba(255, 0, 128, 0.08),
            transparent 45%
        ),
        #080B16;

    color: #E8F1FF;
}


/* MAIN CONTAINER */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* GENERAL TEXT */

h1, h2, h3, p, label {
    color: #E8F1FF !important;
}


/* HEADER CARD */

.cyber-header {
    background: linear-gradient(
        135deg,
        rgba(15, 30, 55, 0.92),
        rgba(35, 20, 65, 0.85)
    );

    border: 1px solid rgba(53, 231, 255, 0.35);

    border-radius: 20px;

    padding: 32px;

    box-shadow:
        0 0 25px rgba(53, 231, 255, 0.10),
        inset 0 0 25px rgba(157, 78, 221, 0.08);

    margin-bottom: 30px;
}


/* LOGO */

.cyber-logo {
    font-size: 34px;

    font-weight: 900;

    letter-spacing: 3px;

    background: linear-gradient(
        90deg,
        #35E7FF,
        #A78BFA,
        #FF4FA3
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


/* SUBTITLE */

.cyber-subtitle {
    color: #A9BAD6;

    font-size: 14px;

    letter-spacing: 2px;
}


/* SECTION HEADINGS */

.section-title {
    color: #35E7FF !important;

    font-size: 22px;

    font-weight: 800;

    letter-spacing: 1px;

    margin-top: 25px;

    margin-bottom: 15px;

    text-shadow:
        0 0 12px rgba(53, 231, 255, 0.3);
}


/* GLASS CARDS */

.model-card,
.explanation-card {

    background: rgba(18, 27, 48, 0.72);

    backdrop-filter: blur(15px);

    -webkit-backdrop-filter: blur(15px);

    border: 1px solid rgba(115, 160, 210, 0.22);

    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 8px 30px rgba(0, 0, 0, 0.25),
        inset 0 0 15px rgba(53, 231, 255, 0.025);

    margin-bottom: 15px;
}


/* MODEL CARDS */

.model-card {
    min-height: 230px;

    transition: 0.3s ease;
}

.model-card:hover,
.explanation-card:hover {

    border-color: rgba(53, 231, 255, 0.65);

    box-shadow:
        0 0 20px rgba(53, 231, 255, 0.12);

    transform: translateY(-3px);
}


/* EXPLANATION CARDS */

.explanation-card {

    padding: 17px;

    border-left: 3px solid #35E7FF;
}


/* URL INPUT */

.stTextInput input {

    background: rgba(20, 31, 54, 0.9) !important;

    color: #FFFFFF !important;

    border: 1px solid rgba(53, 231, 255, 0.45) !important;

    border-radius: 12px !important;

    padding: 15px !important;
}

.stTextInput input:focus {

    border-color: #35E7FF !important;

    box-shadow:
        0 0 12px rgba(53, 231, 255, 0.25) !important;
}


/* BUTTONS */

.stButton > button {

    background: linear-gradient(
        90deg,
        #087EAB,
        #6844C8
    );

    color: white !important;

    border: 1px solid rgba(53, 231, 255, 0.65);

    border-radius: 12px;

    padding: 12px 20px;

    font-weight: 800;

    letter-spacing: 1px;

    box-shadow:
        0 0 15px rgba(53, 231, 255, 0.18);

    transition: 0.3s ease;
}

.stButton > button:hover {

    background: linear-gradient(
        90deg,
        #00A6D6,
        #8954F5
    );

    box-shadow:
        0 0 25px rgba(53, 231, 255, 0.4);

    border-color: #FFFFFF;
}


/* METRICS */

[data-testid="stMetric"] {

    background: rgba(20, 31, 54, 0.75);

    border: 1px solid rgba(53, 231, 255, 0.18);

    padding: 15px;

    border-radius: 13px;
}

[data-testid="stMetricValue"] {
    color: #35E7FF !important;
}


/* TABS */

.stTabs [data-baseweb="tab-list"] {

    background: rgba(15, 25, 45, 0.85);

    border-radius: 12px;

    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    color: #AFC4E5 !important;
}

.stTabs [aria-selected="true"] {
    color: #35E7FF !important;
}


/* EXPANDERS */

.streamlit-expanderHeader {

    background: rgba(20, 31, 54, 0.85);

    color: #35E7FF !important;

    border-radius: 10px;
}


/* DIVIDERS */

hr {
    border-color: rgba(53, 231, 255, 0.2) !important;
}


/* FOOTER */

.footer {

    text-align: center;

    color: #8296B8;

    font-size: 12px;

    letter-spacing: 2px;

    padding: 30px;

    border-top: 1px solid rgba(53, 231, 255, 0.2);

    margin-top: 35px;
}


/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 12px;
}


/* SIDEBAR */

[data-testid="stSidebar"] {

    background: #0C1427;

    border-right: 1px solid rgba(53, 231, 255, 0.2);
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():

    models_dir = os.path.join(
        PROJECT_DIR,
        "models"
    )

    rf_model = joblib.load(
        os.path.join(
            models_dir,
            "random_forest_model.pkl"
        )
    )

    xgb_model = joblib.load(
        os.path.join(
            models_dir,
            "xgboost_model.pkl"
        )
    )

    feature_columns = joblib.load(
        os.path.join(
            models_dir,
            "feature_columns.pkl"
        )
    )

    return rf_model, xgb_model, feature_columns


try:

    rf_model, xgb_model, feature_columns = load_models()

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()



# ==========================================
# 4. PHISHGUARD-PS HEADER
# ==========================================

st.title("PHISHGUARD-PS")

st.caption("Website Security Scanner")

st.subheader(
    "Intelligent Phishing Website Detection"
)

st.write(
    "Analyse website URLs using Machine Learning "
    "and Explainable AI to identify potentially "
    "malicious websites."
)

st.divider()

st.markdown(
    "**Created by Pranay and Siddharth**"
)

st.divider()



# ==========================================
# 5. URL INPUT
# ==========================================

st.markdown(
    '<div class="section-title">◈ Website Security Scanner</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter a website URL to analyse its potential phishing characteristics."
)

url = st.text_input(
    "Website URL",
    placeholder="https://example.com",
    label_visibility="collapsed"
)

analyse_button = st.button(
    "🔍 Analyse Website",
    type="primary",
    use_container_width=True
)


# ==========================================
# 6. FEATURE DESCRIPTIONS
# ==========================================

feature_descriptions = {

    "url_length": "URL Length",

    "dot_count": "Number of Dots",

    "hyphen_count": "Number of Hyphens",

    "digit_count": "Number of Digits",

    "special_char_count": "Special Characters",

    "digit_ratio": "Digit Ratio",

    "percent_encoding_count": "Percent Encoding",

    "has_at_symbol": "Presence of @ Symbol",

    "has_https": "HTTPS Presence",

    "has_http": "HTTP Presence",

    "has_fragment": "URL Fragment",

    "path_length": "Path Length",

    "path_segment_count": "Path Segments",

    "query_length": "Query Length",

    "query_param_count": "Query Parameters",

    "domain_length": "Domain Length",

    "subdomain_count": "Subdomain Count",

    "has_ip_address": "IP Address Usage",

    "has_nonstandard_port": "Non-standard Port",

    "suspicious_keyword_count": "Suspicious Keywords",

    "has_shortening_service": "URL Shortening Service"

}


# ==========================================
# 7. PREDICTION FUNCTION
# ==========================================

def predict_website(model, input_df):

    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)

    phishing_probability = probabilities[
        classes.index(0)
    ]

    legitimate_probability = probabilities[
        classes.index(1)
    ]

    return (
        prediction,
        phishing_probability,
        legitimate_probability
    )


# ==========================================
# 8. SHAP EXPLANATION FUNCTION
# ==========================================

def get_shap_explanation(model, input_df, prediction):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_df)

    classes = list(model.classes_)

    class_index = classes.index(prediction)

    # Handle older SHAP versions

    if isinstance(shap_values, list):

        contributions = np.asarray(
            shap_values[class_index]
        )[0]

    else:

        shap_values = np.asarray(shap_values)

        # Handle newer SHAP output formats

        if shap_values.ndim == 3:

            if shap_values.shape[-1] == len(classes):

                contributions = shap_values[
                    0, :, class_index
                ]

            elif shap_values.shape[1] == len(classes):

                contributions = shap_values[
                    0, class_index, :
                ]

            else:

                raise ValueError(
                    "Unexpected SHAP output dimensions."
                )

        else:

            contributions = shap_values[0]

            # XGBoost binary classification
            # uses class 1 as its positive output.
            # Reverse direction for class 0.

            if (
                model.__class__.__name__ == "XGBClassifier"
                and prediction == 0
            ):

                contributions = -contributions

    # Check feature count

    if len(contributions) != len(feature_columns):

        raise ValueError(
            "SHAP contribution count does not match "
            "the number of model features."
        )

    # Create explanation dataframe

    explanation_df = pd.DataFrame({

        "Feature": [
            feature_descriptions.get(
                feature,
                feature
            )
            for feature in feature_columns
        ],

        "Contribution": contributions,

        "Feature Value": [
            input_df.iloc[0][feature]
            for feature in feature_columns
        ]

    })

    explanation_df["Absolute Impact"] = (
        explanation_df["Contribution"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute Impact",
        ascending=False
    ).reset_index(drop=True)

    return explanation_df




# ==========================================
# 9. DISPLAY SHAP EXPLANATIONS
# ==========================================

def display_explanation(model_name, explanation_df):

    st.subheader(f"Why {model_name} made this prediction")

    st.write(
        "The following URL characteristics had "
        "the strongest influence on the prediction."
    )

    top_features = explanation_df.head(7)

    for _, row in top_features.iterrows():

        feature = row["Feature"]
        contribution = row["Contribution"]
        value = row["Feature Value"]

        if contribution > 0:
            impact = "Supports prediction"
        else:
            impact = "Opposes prediction"

        with st.container(border=True):

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**{feature}**")

            with col2:
                st.write(impact)

            st.caption(
                f"Observed value: {value} | "
                f"SHAP contribution: {contribution:.5f}"
            )




# ==========================================
# 10. MAIN WEBSITE ANALYSIS
# ==========================================

if analyse_button:

    if not url.strip():

        st.warning("Please enter a website URL.")

    else:

        try:

            with st.spinner(
                "Scanning website security..."
            ):

                # Extract URL features
                features = extract_features(url)

                # Maintain original training feature order
                input_df = pd.DataFrame(
                    [[features[col] for col in feature_columns]],
                    columns=feature_columns
                )

                # Random Forest prediction
                rf_result = predict_website(
                    rf_model,
                    input_df
                )

                # XGBoost prediction
                xgb_result = predict_website(
                    xgb_model,
                    input_df
                )

            st.divider()

            st.markdown(
                '<div class="section-title">MODEL ANALYSIS RESULTS</div>',
                unsafe_allow_html=True
            )

            # ==================================
            # RANDOM FOREST AND XGBOOST
            # ==================================

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="model-card">',
                    unsafe_allow_html=True
                )

                st.subheader("🌳 Random Forest")

                if rf_result[0] == 0:
                    st.error("⚠ Potential Phishing Website")
                else:
                    st.success("✓ Likely Legitimate Website")

                st.metric(
                    "Phishing Probability",
                    f"{rf_result[1] * 100:.2f}%"
                )

                st.progress(float(rf_result[1]))

                st.metric(
                    "Legitimate Probability",
                    f"{rf_result[2] * 100:.2f}%"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            with col2:

                st.markdown(
                    '<div class="model-card">',
                    unsafe_allow_html=True
                )

                st.subheader("⚡ XGBoost")

                if xgb_result[0] == 0:
                    st.error("⚠ Potential Phishing Website")
                else:
                    st.success("✓ Likely Legitimate Website")

                st.metric(
                    "Phishing Probability",
                    f"{xgb_result[1] * 100:.2f}%"
                )

                st.progress(float(xgb_result[1]))

                st.metric(
                    "Legitimate Probability",
                    f"{xgb_result[2] * 100:.2f}%"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            # ==================================
            # MODEL AGREEMENT
            # ==================================

            st.divider()

            st.markdown(
                '<div class="section-title">MODEL CONSENSUS</div>',
                unsafe_allow_html=True
            )

            if rf_result[0] == xgb_result[0]:

                if rf_result[0] == 0:

                    st.error(
                        "Both machine learning models identify "
                        "this website as potentially phishing."
                    )

                else:

                    st.success(
                        "Both machine learning models classify "
                        "this website as likely legitimate."
                    )

            else:

                st.warning(
                    "The models disagree. Treat this result "
                    "as uncertain and exercise caution."
                )

            # ==================================
            # SHAP EXPLANATIONS
            # ==================================

            st.divider()

            st.markdown(
                '<div class="section-title">EXPLAINABLE AI INTELLIGENCE</div>',
                unsafe_allow_html=True
            )

            st.write(
                "Discover which URL characteristics influenced "
                "each model's prediction."
            )

            try:

                with st.spinner(
                    "Generating AI explanations..."
                ):

                    rf_explanation = get_shap_explanation(
                        rf_model,
                        input_df,
                        rf_result[0]
                    )

                    xgb_explanation = get_shap_explanation(
                        xgb_model,
                        input_df,
                        xgb_result[0]
                    )

                tab1, tab2 = st.tabs([
                    "🌳 Random Forest",
                    "⚡ XGBoost"
                ])

                with tab1:

                    display_explanation(
                        "Random Forest",
                        rf_explanation
                    )

                with tab2:

                    display_explanation(
                        "XGBoost",
                        xgb_explanation
                    )

            except Exception as e:

                st.warning(
                    f"SHAP explanation could not be generated: {e}"
                )

            # ==================================
            # TECHNICAL FEATURE DETAILS
            # ==================================

            st.divider()

            with st.expander(
                "🔬 View Technical Feature Extraction Details"
            ):

                feature_display = pd.DataFrame({

                    "Feature": feature_columns,

                    "Description": [
                        feature_descriptions.get(
                            feature,
                            feature
                        )
                        for feature in feature_columns
                    ],

                    "Extracted Value": [
                        features[feature]
                        for feature in feature_columns
                    ]

                })

                st.dataframe(
                    feature_display,
                    use_container_width=True,
                    hide_index=True
                )

            # ==================================
            # SECURITY DISCLAIMER
            # ==================================

            st.divider()

            st.info(
                """
                **SECURITY DISCLAIMER**

                PhishGuard-PS uses machine learning models to
                estimate phishing risk based on URL characteristics.

                Predictions are not guarantees of website safety.
                False positives and false negatives are possible.

                Always exercise caution before entering personal
                information or credentials on unfamiliar websites.
                """
            )

        except ValueError as e:

            st.error(f"Invalid URL: {e}")

        except Exception as e:

            st.error(f"Analysis failed: {e}")



# ==========================================
# 11. DASHBOARD FOOTER
# ==========================================

st.divider()

st.markdown(
    "### PHISHGUARD-PS"
)

st.caption(
    "Powered by Random Forest, XGBoost and SHAP"
)

st.caption(
    "Created by Pranay and Siddharth"
)

st.caption(
    "Machine Learning for Website Security"
)
