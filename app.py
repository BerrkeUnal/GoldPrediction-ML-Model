import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    
    /* 1. CONTAINER (BACKGROUND GREY) */
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #262730 !important; /* Selectbox Grisi */
        border: 1px solid #4C4F56 !important; /* Kenarlık */
        border-radius: 5px;
    }
    
    /* 2. THE NUMBER TEXT INSIDE */
    div[data-testid="stNumberInput"] input {
        caret-color: #E0C041 !important;
        font-weight: bold;
    }

    /* Focus State: Border turns Gold when clicked */
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
    st.markdown("<p style='color:#E0C041; font-size:1.1rem;'>AI-Powered Analysis for XAU/USD & Market Indicators</p>", unsafe_allow_html=True)

st.markdown("---")

# --- 4. DATA LOADING ---
try:
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv("financial_regression.csv")
    df = df.ffill()
    last_row = df.iloc[-1]
except FileNotFoundError:
    st.error("❌ Critical Error: Model files (.pkl) not found. Please run 'model.py' first.")
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
        model_option = st.selectbox(
            "Select Prediction Algorithm:",
            ("Random Forest (Best Performance)", "Ridge Regression (Linear)", "SVR (Support Vector)")
        )

        if "Random" in model_option:
            model = joblib.load('rf_model.pkl')
        elif "Ridge" in model_option:
            model = joblib.load('ridge_model.pkl')
        else:
            model = joblib.load('svm_model.pkl')

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
                prediction_etf = model.predict(input_scaled)[0]
                real_prediction = prediction_etf * 10 
                change = real_prediction - gold_input
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.info("Reference (Previous Close)")
                    st.metric(label="USD / Ounce", value=f"${gold_input:,.2f}")
                with res_col2:
                    st.success(f"Model Prediction ({model_option.split(' ')[0]})")
                    st.metric(label="Target Price", value=f"${real_prediction:,.2f}", delta=f"{change:+,.2f} $")

# ================= TAB 2: ANALYSIS =================
with tab2:
    st.markdown("### 📈 Comprehensive Market Analysis")
    st.info("Interactive visualizations generated from historical data.")

    # 1. TIME SERIES
    st.markdown("#### 1. Historical Trends (Time Series)")
    selected_col = st.selectbox("Select Feature to Visualize:", df.columns, index=len(df.columns)-1)
    fig_line = px.line(df, x='date', y=selected_col, title=f'{selected_col} Over Time')
    fig_line.update_traces(line_color='#E0C041')
    st.plotly_chart(fig_line, use_container_width=True)

    # 2. HEATMAP
    st.markdown("#### 2. Correlation Heatmap")
    corr_cols = ['gold close', 'silver close', 'oil close', 'sp500 close', 'eur_usd', 'us_rates_%']
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdYlGn', title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    col_viz1, col_viz2 = st.columns(2)

    # 3. BOX PLOTS
    with col_viz1:
        st.markdown("#### 3. Outlier Analysis (Box Plot)")
        box_col = st.selectbox("Select Feature for Box Plot:", ['gold close', 'silver close', 'oil close', 'sp500 close'])
        fig_box = px.box(df, y=box_col, title=f"Distribution of {box_col}")
        fig_box.update_traces(marker_color='#E0C041')
        st.plotly_chart(fig_box, use_container_width=True)

    # 4. HISTOGRAM
    with col_viz2:
        st.markdown("#### 4. Data Distribution (Histogram)")
        hist_col = st.selectbox("Select Feature for Histogram:", ['gold close', 'us_rates_%', 'eur_usd'])
        fig_hist = px.histogram(df, x=hist_col, nbins=50, title=f"Frequency of {hist_col}")
        fig_hist.update_traces(marker_color='#FFFFFF')
        st.plotly_chart(fig_hist, use_container_width=True)

    # 5. PAIR PLOT
    st.markdown("#### 5. Asset Relations (Scatter Matrix)")
    fig_pair = px.scatter_matrix(
        df, 
        dimensions=['gold close', 'silver close', 'oil close', 'sp500 close'],
        title="Multi-Asset Scatter Matrix",
        opacity=0.5
    )
    fig_pair.update_traces(diagonal_visible=False, marker=dict(color='#E0C041', size=3))
    st.plotly_chart(fig_pair, use_container_width=True)