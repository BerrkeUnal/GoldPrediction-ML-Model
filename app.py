import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import xgboost as xgb

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Gold Price Prediction Pro", page_icon="🏆", layout="wide")

# --- 2. CUSTOM CSS STYLING (PERFECT DARK & GOLD) ---
st.markdown("""
<style>
    /* GENERAL PAGE BACKGROUND */
    .stApp {
        background-color: #0E1117; 
    }

    /* HEADERS (Gold Color) */
    h1, h2, h3, h4, h5, h6 {
        color: #E0C041 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* LABELS (White text for labels) */
    .stMarkdown p, .stMarkdown label, div[data-testid="stNumberInput"] label p {
        color: #FFFFFF !important;
        font-size: 1rem;
    }
    
    /* --- INPUT BOX STYLING --- */
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #262730 !important;
        border: 1px solid #4C4F56 !important;
        border-radius: 5px;
    }
    div[data-testid="stNumberInput"] input {
        caret-color: #E0C041 !important;
        font-weight: bold;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
        border: 1px solid #E0C041 !important;
        box-shadow: 0 0 5px rgba(224, 192, 65, 0.5);
    }
    
    /* --- SELECTBOX STYLING --- */
    div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        border: 1px solid #4C4F56 !important;
        color: #E0C041 !important;
    }
    div[data-baseweb="popover"] div {
        background-color: #262730 !important;
        color: #FFFFFF !important;
    }

    /* METRIC BOXES */
    [data-testid="stMetricValue"] {
        color: #E0C041;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #FFFFFF;
    }

    /* BUTTON DESIGN */
    div.stButton > button:first-child {
        background-color: #E0C041; 
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.3s;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        background-color: #FFD700;
        color: #000000;
        box-shadow: 0px 0px 15px rgba(224, 192, 65, 0.6);
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-top-color: #E0C041 !important;
        color: #E0C041 !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. TITLE & HEADER ---
col_head1, col_head2 = st.columns([1, 10])
with col_head1:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=80)
with col_head2:
    st.title("Advanced Gold Price Prediction System")
    st.markdown("<p style='color:#E0C041; font-size:1.1rem;'>AI-Powered Analysis using Ensemble & Non-Ensemble Models</p>", unsafe_allow_html=True)

st.markdown("---")

# --- 4. DATA & MODEL LOADING ---
@st.cache_resource
def load_data_and_models():
    # 1. Load CSV
    try:
        if os.path.exists("financial_regression.csv"):
            df = pd.read_csv("financial_regression.csv")
        else:
            url = "https://raw.githubusercontent.com/BerrkeUnal/GoldPrediction-ML-Model/main/financial_regression.csv"
            df = pd.read_csv(url, delimiter=",", encoding='latin1')
        
        df = df.ffill()
        last_row = df.iloc[-1]
    except Exception as e:
        return None, None, None, str(e)

    # 2. Load Models
    model_folder = "Saved_Models"
    try:
        scaler = joblib.load(os.path.join(model_folder, 'scaler.pkl'))
        
        models = {
            "Random Forest (Ensemble)": joblib.load(os.path.join(model_folder, 'rf_model.pkl')),
            "Gradient Boosting (Ensemble)": joblib.load(os.path.join(model_folder, 'gb_model.pkl')),
            "XGBoost (Ensemble)": joblib.load(os.path.join(model_folder, 'xgb_model.pkl')),
            "Linear Regression (Non-Ensemble)": joblib.load(os.path.join(model_folder, 'linear_model.pkl')),
            "Ridge Regression (Non-Ensemble)": joblib.load(os.path.join(model_folder, 'ridge_model.pkl')),
            "SVR (Non-Ensemble)": joblib.load(os.path.join(model_folder, 'svm_model.pkl')),
        }
    except FileNotFoundError:
        return None, None, None, "Model files not found in 'Saved_Models' folder."

    return df, last_row, scaler, models

df, last_row, scaler, models = load_data_and_models()

if models is None:
    st.error("❌ Critical Error: " + str(scaler))
    st.stop()

# --- 5. MAIN INTERFACE ---
tab1, tab2 = st.tabs(["🏠 Prediction Simulator", "📊 Advanced Market Analysis"])

# ================= TAB 1: PREDICTION SIMULATOR =================
with tab1:
    st.markdown("### 📝 Enter Market Indicators")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🌎 Global Markets")
        sp500 = st.number_input("S&P 500 Close", value=float(last_row['sp500 close']))
        nasdaq = st.number_input("Nasdaq Close", value=float(last_row['nasdaq close']))
        us_rates = st.number_input("US Rates %", value=float(last_row['us_rates_%']))
        
    with col2:
        st.markdown("##### 💶 Forex & Macro")
        eur_usd = st.number_input("EUR/USD", value=float(last_row['eur_usd']), step=0.0001, format="%.4f")
        usd_chf = st.number_input("USD/CHF", value=float(last_row['usd_chf']), step=0.0001, format="%.4f")
        cpi = st.number_input("CPI (Inflation)", value=float(last_row['CPI']), step=0.1, format="%.2f")
        gdp = st.number_input("GDP", value=float(last_row['GDP']), step=100.0)

    with col3:
        st.markdown("##### 💎 Commodities")
        gold_input = st.number_input("Reference Gold Price ($)", value=float(last_row['gold close'])*10, step=1.0, format="%.2f", help="Yesterday's Ounce Price")
        silver = st.number_input("Silver Close ($)", value=float(last_row['silver close']))
        oil = st.number_input("Oil Close ($)", value=float(last_row['oil close']))
        platinum = st.number_input("Platinum Close ($)", value=float(last_row['platinum close']))
        palladium = st.number_input("Palladium Close ($)", value=float(last_row['palladium close']))

    st.markdown("---")

    # --- MODEL & PREDICTION ---
    c_left, c_center, c_right = st.columns([1, 2, 1])
    
    with c_center:
        st.markdown("### ⚙️ Model Configuration")
        model_name = st.selectbox("Select Prediction Algorithm:", list(models.keys()))
        selected_model = models[model_name]

        if st.button("PREDICT TARGET PRICE", use_container_width=True):
            with st.spinner('Calculating probabilities...'):
                gold_lag1_model = gold_input / 10 
                input_data = {
                    'sp500 close': sp500, 'nasdaq close': nasdaq, 'us_rates_%': us_rates, 'CPI': cpi,
                    'usd_chf': usd_chf, 'eur_usd': eur_usd, 'GDP': gdp, 'silver close': silver,
                    'oil close': oil, 'platinum close': platinum, 'palladium close': palladium,
                    'gold_lag1': gold_lag1_model 
                }
                input_df = pd.DataFrame(input_data, index=[0])
                input_scaled = scaler.transform(input_df)
                prediction_etf = selected_model.predict(input_scaled)[0]
                real_prediction = prediction_etf * 10 
                change = real_prediction - gold_input
                
                st.markdown("<br>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.info("Reference (Previous Close)")
                    st.metric(label="USD / Ounce", value=f"${gold_input:,.2f}")
                with res_col2:
                    st.success(f"Prediction: {model_name}")
                    st.metric(label="Target Price", value=f"${real_prediction:,.2f}", delta=f"{change:+,.2f} $")

# ================= TAB 2: ANALYSIS =================
with tab2:
    st.markdown("### 📈 Comprehensive Market Analysis")
    st.info("Interactive visualizations and AI Explainability.")
    
    # --- YENİ BÖLÜM: AI EXPLAINABILITY (GÖRSELLER) ---
    st.markdown("---")
    st.subheader("🧠 Model Explainability (AI Logic)")
    st.markdown("These visualizations were generated during the model training phase to explain how models make decisions.")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("#### 1. Feature Importance")
        st.caption("Which economic indicators matter most?")
        
        # Kullanıcı hangi modelin grafiğini görmek istiyor?
        fi_option = st.selectbox("Select Model for Feature Importance:", ["XGBoost", "Random Forest", "Gradient Boosting"])
        
        # Dosya adını oluştur (feature_importance_XGBoost.png gibi)
        fi_filename = f"feature_importance_{fi_option.replace(' ', '_')}.png"
        fi_path = os.path.join("EDA_Results", fi_filename)
        
        if os.path.exists(fi_path):
            st.image(fi_path, caption=f"Feature Importance for {fi_option}", use_container_width=True)
        else:
            st.warning(f"Image not found: {fi_filename}. Please run model.py first.")

    with exp_col2:
        st.markdown("#### 2. SHAP Analysis (XGBoost)")
        st.caption("How features push the price up (red) or down (blue).")
        
        shap_path = os.path.join("EDA_Results", "shap_summary_plot.png")
        if os.path.exists(shap_path):
            st.image(shap_path, caption="SHAP Summary Plot", use_container_width=True)
        else:
            st.warning("SHAP image not found. Ensure SHAP analysis ran in model.py.")

    st.markdown("---")

    # --- MEVCUT PLOTLY GRAFİKLERİ ---
    if df is not None:
        st.subheader("📊 Interactive Data Exploration")
        
        # 3. TIME SERIES
        st.markdown("#### 3. Historical Trends")
        selected_col = st.selectbox("Select Feature to Visualize:", df.columns, index=len(df.columns)-1)
        fig_line = px.line(df, x='date', y=selected_col, title=f'{selected_col} Over Time')
        fig_line.update_traces(line_color='#E0C041')
        st.plotly_chart(fig_line, use_container_width=True)

        col_viz1, col_viz2 = st.columns(2)

        # 4. HEATMAP
        with col_viz1:
            st.markdown("#### 4. Correlation Heatmap")
            corr_cols = ['gold close', 'silver close', 'oil close', 'sp500 close', 'eur_usd', 'us_rates_%']
            corr = df[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdYlGn', title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

        # 5. HISTOGRAM
        with col_viz2:
            st.markdown("#### 5. Data Distribution")
            hist_col = st.selectbox("Select Feature for Histogram:", ['gold close', 'us_rates_%', 'eur_usd'])
            fig_hist = px.histogram(df, x=hist_col, nbins=50, title=f"Frequency of {hist_col}")
            fig_hist.update_traces(marker_color='#FFFFFF')
            st.plotly_chart(fig_hist, use_container_width=True)