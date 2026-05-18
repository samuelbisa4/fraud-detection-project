import streamlit as st
import pandas as pd
import joblib
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# =====================================================
# LOAD MODEL
# =====================================================
try:
    model = joblib.load("models/best_model.pkl")
except:
    st.error("❌ Failed to load model | Model gagal dimuat")
    st.stop()

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

/* Hide Streamlit Default */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white;
}

/* Main Card */
.main-card {
    background: rgba(255,255,255,0.06);
    padding: 40px;
    border-radius: 28px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 0px 40px rgba(0,0,0,0.4);
}

/* Title */
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Info Box */
.info-box {
    background: rgba(56,189,248,0.08);
    border-left: 5px solid #38bdf8;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 30px;
    color: #e2e8f0;
    line-height: 1.7;
}

/* Section Title */
.section-title {
    font-size: 24px;
    font-weight: bold;
    color: #38bdf8;
    margin-top: 10px;
    margin-bottom: 20px;
}

/* Input Label */
label {
    color: white !important;
    font-weight: 600 !important;
}

/* Input Style */
.stNumberInput input,
.stSelectbox div,
.stSlider {
    border-radius: 12px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 58px;
    border: none;
    border-radius: 16px;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px rgba(59,130,246,0.7);
}

/* Result Box */
.success-box {
    background: rgba(34,197,94,0.12);
    border: 1px solid #22c55e;
    padding: 25px;
    border-radius: 20px;
    margin-top: 25px;
    text-align: center;
    animation: fadeIn 0.5s ease-in-out;
}

.error-box {
    background: rgba(239,68,68,0.12);
    border: 1px solid #ef4444;
    padding: 25px;
    border-radius: 20px;
    margin-top: 25px;
    text-align: center;
    animation: fadeIn 0.5s ease-in-out;
}

/* Result Title */
.result-title {
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Result Text */
.result-text {
    color: #f1f5f9;
    font-size: 18px;
    line-height: 1.8;
}

/* Metric Card */
.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    margin-top: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 40px;
}

/* Animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(12px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="main-card">

<div class="main-title">
💳 AI Fraud Detection
</div>

<div class="subtitle">
Smart Credit Card Fraud Detection System <br>
Sistem Cerdas Deteksi Fraud Kartu Kredit
</div>

<div class="info-box">
<b>English:</b> Fill transaction details to predict fraud probability using Machine Learning. <br><br>

<b>Indonesia:</b> Masukkan detail transaksi untuk memprediksi kemungkinan fraud menggunakan Machine Learning.
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT SECTION
# =====================================================
st.markdown("""
<div class="section-title">
Transaction Information | Informasi Transaksi
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUTS
# =====================================================

amount = st.number_input(
    "Transaction Amount | Jumlah Transaksi",
    min_value=0.0,
    value=100.0
)

transaction_hour = st.slider(
    "Transaction Hour | Jam Transaksi",
    0,
    23,
    12
)

foreign_transaction = st.selectbox(
    "Foreign Transaction | Transaksi Luar Negeri",
    ["No", "Yes"]
)

foreign_transaction = 1 if foreign_transaction == "Yes" else 0

location_mismatch = st.selectbox(
    "Location Mismatch | Ketidaksesuaian Lokasi",
    ["No", "Yes"]
)

location_mismatch = 1 if location_mismatch == "Yes" else 0

device_trust_score = st.slider(
    "Device Trust Score | Skor Kepercayaan Device",
    0.0,
    1.0,
    0.5
)

velocity_last_24h = st.number_input(
    "Transaction Frequency 24h | Frekuensi Transaksi 24 Jam",
    min_value=0,
    value=1
)

cardholder_age = st.number_input(
    "Cardholder Age | Umur Pemilik Kartu",
    min_value=18,
    value=25
)

merchant_category = st.selectbox(
    "Merchant Category | Kategori Merchant",
    [
        "Clothing",
        "Electronics",
        "Food",
        "Grocery",
        "Travel"
    ]
)

# =====================================================
# ONE HOT ENCODING
# =====================================================
merchant_clothing = 1 if merchant_category == "Clothing" else 0
merchant_electronics = 1 if merchant_category == "Electronics" else 0
merchant_food = 1 if merchant_category == "Food" else 0
merchant_grocery = 1 if merchant_category == "Grocery" else 0
merchant_travel = 1 if merchant_category == "Travel" else 0

# =====================================================
# DATAFRAME
# =====================================================
input_data = pd.DataFrame([{
    "amount": amount,
    "transaction_hour": transaction_hour,
    "foreign_transaction": foreign_transaction,
    "location_mismatch": location_mismatch,
    "device_trust_score": device_trust_score,
    "velocity_last_24h": velocity_last_24h,
    "cardholder_age": cardholder_age,
    "merchant_category_Clothing": merchant_clothing,
    "merchant_category_Electronics": merchant_electronics,
    "merchant_category_Food": merchant_food,
    "merchant_category_Grocery": merchant_grocery,
    "merchant_category_Travel": merchant_travel
}])

# =====================================================
# PREDICTION BUTTON
# =====================================================
if st.button("Predict Fraud | Prediksi Fraud"):

    with st.spinner("Analyzing transaction... | Menganalisis transaksi..."):
        time.sleep(2)

    # =================================================
    # PROBABILITY
    # =================================================
    probability = model.predict_proba(input_data)[0][1]

    # =================================================
    # THRESHOLD TUNING
    # =================================================
    threshold = 0.35

    # Manual Prediction
    prediction = 1 if probability > threshold else 0

    # =================================================
    # FRAUD RESULT
    # =================================================
    if prediction == 1:

        st.markdown(f"""
        <div class="error-box">

            <div class="result-title">
            FRAUD DETECTED
            </div>

            <div class="result-text">
            This transaction has a high probability of fraud. <br>
            Transaksi ini memiliki kemungkinan fraud yang tinggi.
            <br><br>

            <b>Fraud Probability:</b> {probability:.2%}
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(float(probability))

        st.markdown(f"""
        <div class="metric-box">

        <b>Risk Level:</b> HIGH RISK <br><br>

        <b>Fraud Score:</b> {probability:.2%} <br><br>

        ⚠ This transaction should be reviewed immediately. <br>
        ⚠ Transaksi ini perlu diperiksa lebih lanjut.

        </div>
        """, unsafe_allow_html=True)

    # =================================================
    # NORMAL RESULT
    # =================================================
    else:

        safe_score = 1 - probability

        st.markdown(f"""
        <div class="success-box">

            <div class="result-title">
            NORMAL TRANSACTION
            </div>

            <div class="result-text">
            This transaction appears safe and legitimate. <br>
            Transaksi terlihat aman dan valid.
            <br><br>

            <b>Confidence Score:</b> {safe_score:.2%}
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(float(safe_score))

        st.markdown(f"""
        <div class="metric-box">

        <b>Risk Level:</b> LOW RISK <br><br>

        <b>Safe Score:</b> {safe_score:.2%} <br><br>

        ✔ No suspicious activity detected. <br>
        ✔ Tidak ditemukan aktivitas mencurigakan.

        </div>
        """, unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.markdown("## About Project")

    st.info("""
### English
This application predicts whether a credit card transaction is fraudulent or legitimate using Machine Learning.

### Indonesia
Aplikasi ini memprediksi apakah transaksi kartu kredit termasuk fraud atau normal menggunakan Machine Learning.
""")

    st.markdown("---")

    st.markdown("## Machine Learning Model")

    st.success("""
✔ Random Forest  
✔ Supervised Learning  
✔ Binary Classification  
✔ Fraud Detection System  
✔ Threshold Tuning
""")

    st.markdown("---")

    st.markdown("## Features")

    st.write("""
- Transaction Amount
- Transaction Hour
- Foreign Transaction
- Location Mismatch
- Device Trust Score
- Transaction Frequency
- Merchant Category
""")

    st.markdown("---")

    st.markdown("## ⚙ Detection Settings")

    st.write("""
Current Fraud Threshold:
""")

    st.code("0.35")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">

AI-Based Fraud Detection System <br><br>

Machine Learning • Random Forest • Streamlit • Python

</div>
""", unsafe_allow_html=True)