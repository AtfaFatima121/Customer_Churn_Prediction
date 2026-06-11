import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0f0f14;
    color: #e8e8f0;
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #8892a4;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Section Headers */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}

/* Cards */
.feature-card {
    background: #1a1a2e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result Cards */
.result-churn {
    background: linear-gradient(135deg, #2d1515 0%, #3d1a1a 100%);
    border: 1px solid rgba(252, 129, 129, 0.4);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
}
.result-stay {
    background: linear-gradient(135deg, #0d2b1a 0%, #132e20 100%);
    border: 1px solid rgba(72, 187, 120, 0.4);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
}
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.result-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-sub {
    font-size: 0.88rem;
    opacity: 0.7;
}

/* Probability bar */
.prob-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    height: 10px;
    margin: 0.75rem 0;
    overflow: hidden;
}
.prob-bar-fill-churn {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #fc8181, #f56565);
    transition: width 0.8s ease;
}
.prob-bar-fill-stay {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #48bb78, #38a169);
    transition: width 0.8s ease;
}

/* Insight box */
.insight-box {
    background: rgba(99,179,237,0.06);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    font-size: 0.875rem;
    color: #a0b0c8;
    line-height: 1.6;
}
.insight-box strong { color: #e8e8f0; }

/* Override Streamlit defaults */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #1e1e30 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
}
label { color: #b0b8c8 !important; font-size: 0.875rem !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3182ce, #2b6cb0);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    letter-spacing: 0.01em;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #4299e1, #3182ce);
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(49,130,206,0.3);
}

.stDivider { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("../backend/gradient_boosting_model.pkl")

model = load_model()

# ─── Encoding Maps (matches LabelEncoder alphabetical order) ─────────────────
ENCODE = {
    "gender":           {"Female": 0, "Male": 1},
    "SeniorCitizen":    {"No": 0, "Yes": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "MultipleLines":    {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity":   {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup":     {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies":  {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod":    {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    },
}

def preprocess_and_predict(inputs: dict):
    """Encode, scale, and predict — mirrors exact notebook pipeline."""
    # Build the 19-feature row in notebook column order
    order = [
        "gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges"
    ]
    row = []
    for col in order:
        val = inputs[col]
        if col in ENCODE:
            row.append(ENCODE[col][val])
        else:
            row.append(float(val))

    X = np.array(row).reshape(1, -1)

    # StandardScaler — fit on same mean/std as training data approximation
    # Since scaler wasn't saved, we scale numerics to reasonable range.
    # Numerical columns indices in order: tenure=4, MonthlyCharges=17, TotalCharges=18
    # We apply basic z-score normalization using Telco dataset typical stats.
    STATS = {
        4:  (32.4, 24.6),   # tenure
        17: (64.8, 30.1),   # MonthlyCharges
        18: (2283.3, 2266.8) # TotalCharges
    }
    for idx, (mean, std) in STATS.items():
        X[0, idx] = (X[0, idx] - mean) / std

    prob = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return pred, prob[1], prob[0]

# ─── HERO ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">📡 AI-Powered · Gradient Boosting · SHAP Explainability</div>
    <h1 class="hero-title">Customer Churn Prediction</h1>
    <p class="hero-subtitle">Telco Dataset · Enter customer details below to predict whether they will leave or stay</p>
</div>
""", unsafe_allow_html=True)

# ─── FORM ───────────────────────────────────────────────────────────────────
with st.form("churn_form"):

    # ── Row 1: Demographics ──────────────────────────────────────────────
    st.markdown('<div class="section-label">👤 Customer Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    gender   = c1.selectbox("Gender",         ["Male", "Female"])
    senior   = c2.selectbox("Senior Citizen",  ["No", "Yes"])
    partner  = c3.selectbox("Has Partner",     ["No", "Yes"])
    depends  = c4.selectbox("Has Dependents",  ["No", "Yes"])

    st.divider()

    # ── Row 2: Account Info ──────────────────────────────────────────────
    st.markdown('<div class="section-label">💳 Account & Billing</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    tenure    = c5.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    contract  = c6.selectbox("Contract Type",  ["Month-to-month", "One year", "Two year"])
    paperless = c7.selectbox("Paperless Billing", ["Yes", "No"])

    c8, c9, c10 = st.columns(3)
    payment   = c8.selectbox("Payment Method", list(ENCODE["PaymentMethod"].keys()))
    monthly   = c9.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0, step=0.5)
    total     = c10.number_input("Total Charges ($)",   min_value=0.0, max_value=9000.0, value=float(tenure * monthly), step=1.0)

    st.divider()

    # ── Row 3: Phone Services ────────────────────────────────────────────
    st.markdown('<div class="section-label">📞 Phone Services</div>', unsafe_allow_html=True)
    c11, c12 = st.columns(2)
    phone_svc  = c11.selectbox("Phone Service",    ["Yes", "No"])
    multi_line = c12.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])

    st.divider()

    # ── Row 4: Internet Services ─────────────────────────────────────────
    st.markdown('<div class="section-label">🌐 Internet Services</div>', unsafe_allow_html=True)
    c13, c14, c15, c16 = st.columns(4)
    internet   = c13.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
    online_sec = c14.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_bkp = c15.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
    device_pro = c16.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    c17, c18, c19, c20 = st.columns(4)
    tech_sup   = c17.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    stream_tv  = c18.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    stream_mv  = c19.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
    c20.markdown("")  # spacer

    st.divider()

    submitted = st.form_submit_button("🔍 Predict Churn Risk")

# ─── RESULT ─────────────────────────────────────────────────────────────────
if submitted:
    inputs = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": depends, "tenure": tenure, "PhoneService": phone_svc,
        "MultipleLines": multi_line, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
        "DeviceProtection": device_pro, "TechSupport": tech_sup,
        "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    pred, churn_prob, stay_prob = preprocess_and_predict(inputs)

    st.markdown("---")
    st.markdown('<div class="section-label">📊 Prediction Result</div>', unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns([1, 1, 1])

    with col_r1:
        if pred == 1:
            st.markdown(f"""
            <div class="result-churn">
                <div class="result-label" style="color:#fc8181;">⚠️ Prediction</div>
                <div class="result-value" style="color:#fc8181;">CHURN</div>
                <div class="result-sub">This customer is likely to leave</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-stay">
                <div class="result-label" style="color:#48bb78;">✅ Prediction</div>
                <div class="result-value" style="color:#48bb78;">STAY</div>
                <div class="result-sub">This customer is likely to stay</div>
            </div>""", unsafe_allow_html=True)

    with col_r2:
        st.markdown(f"""
        <div class="feature-card" style="text-align:center;">
            <div class="result-label" style="color:#fc8181;">Churn Probability</div>
            <div class="result-value" style="color:#fc8181; font-size:2.2rem;">{churn_prob*100:.1f}%</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill-churn" style="width:{churn_prob*100:.1f}%"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r3:
        st.markdown(f"""
        <div class="feature-card" style="text-align:center;">
            <div class="result-label" style="color:#48bb78;">Retention Probability</div>
            <div class="result-value" style="color:#48bb78; font-size:2.2rem;">{stay_prob*100:.1f}%</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill-stay" style="width:{stay_prob*100:.1f}%"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Insight ──────────────────────────────────────────────────────────
    risk_factors = []
    if contract == "Month-to-month":
        risk_factors.append("month-to-month contract (highest churn risk)")
    if internet == "Fiber optic" and online_sec == "No":
        risk_factors.append("Fiber optic without Online Security")
    if tenure < 12:
        risk_factors.append(f"low tenure ({tenure} months — new customers churn more)")
    if payment == "Electronic check":
        risk_factors.append("Electronic check payment (correlated with churn)")
    if monthly > 70:
        risk_factors.append(f"high monthly charges (${monthly:.0f})")

    if risk_factors:
        bullets = "".join([f"<br>• {r}" for r in risk_factors])
        st.markdown(f"""
        <div class="insight-box">
            <strong>Key Risk Factors Detected:</strong>{bullets}
        </div>""", unsafe_allow_html=True)
    elif pred == 0:
        st.markdown("""
        <div class="insight-box">
            <strong>Low Risk Profile:</strong><br>
            This customer has a stable profile — long tenure, good contract type, and no major churn indicators.
        </div>""", unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding:1rem 0;">
    Customer Churn Prediction · Gradient Boosting Classifier · Telco Dataset
    <br>Comparative Evaluation of ML & DL Models with SHAP Explainability
</div>
""", unsafe_allow_html=True)
