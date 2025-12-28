Markdown

# 🏆 Gold Price Prediction Pro

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-red)
![SHAP](https://img.shields.io/badge/AI-SHAP%20Analysis-green)

**Gold Price Prediction Pro** is an advanced Machine Learning application designed to forecast the price of Gold (XAU/USD). It goes beyond simple regression by utilizing **Ensemble Learning**, **Hyperparameter Tuning**, and **Explainable AI (XAI)** to provide accurate market insights.

The application features a **Premium Dark & Gold UI**, offering interactive charts, real-time simulations, and deep market analysis with model interpretability.

## 🌟 Key Features

### 🧠 Advanced Machine Learning

- **6 Different Models:** Compare performance between **Ensemble Models** (Random Forest, XGBoost, Gradient Boosting) and **Non-Ensemble Models** (Linear Regression, Ridge, SVR).
- **Hyperparameter Tuning:** Models are optimized using `GridSearchCV` for best performance.
- **Robust Preprocessing:** Includes automated data cleaning, scaling (StandardScaler), and lag-feature generation.

### 📊 Interactive Analysis & AI Explainability

- **Explainable AI (XAI):** Visualizes **SHAP (SHapley Additive exPlanations)** values to show _why_ the model made a specific prediction.
- **Feature Importance:** Dynamic charts showing which economic indicators (e.g., Oil, Silver, CPI) drive the gold price.
- **Deep Market Analysis:** Interactive Time Series, Correlation Heatmaps, and Distribution Plots powered by **Plotly**.

### 🎨 Premium UI/UX

- **Scenario Analysis:** Modify inputs like S&P 500, Oil, Silver, and US Rates to see real-time impacts on Gold prices.
- **Dark & Gold Theme:** A custom-designed interface for a professional financial terminal feel.

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Python)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Model Interpretability:** SHAP
- **Visualization:** Plotly, Matplotlib, Seaborn

## 🚀 Installation & Setup

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/BerrkeUnal/GoldPrediction-ML-Model.git
cd GoldPrediction-ML-Model

2. Install Dependencies
Make sure you have Python installed. Then run:

Bash:
pip install -r requirements.txt

3. Train the Models
Before running the app, you need to generate the pre-trained model files and analysis charts. Run the model script once:

Bash:
python model.py
This script will:

Train and optimize 6 ML models.

Create Saved_Models/ folder and save .pkl files.

Create EDA_Results/ folder and save SHAP/Feature Importance images.

4. Run the Application
Start the Streamlit interface:

Bash:

streamlit run app.py

📂 Project Structure
Plaintext

├── app.py                   # Main Streamlit Application (Frontend)
├── model.py                 # ML Model Training, Tuning, SHAP Analysis Script
├── financial_regression.csv # Historical Dataset
├── requirements.txt         # Project Dependencies
├── Saved_Models/            # Folder containing trained models (scaler.pkl, xgb_model.pkl, etc.)
├── EDA_Results/             # Folder containing generated charts (SHAP, Feature Importance, Histograms)
└── README.md                # Project Documentation
📊 Dataset Indicators
The model uses the following features for prediction:

Global Markets: S&P 500, Nasdaq, US Rates (Interests)

Currencies: EUR/USD, USD/CHF

Commodities: Silver, Oil, Platinum, Palladium, Gold (Lag1)

Macro: CPI (Inflation), GDP

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📜 License
This project is open-source and available under the MIT License.

Developed  Berke Ünal
```
