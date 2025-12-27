# 🏆 Gold Price Prediction Pro

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

**Gold Price Prediction Pro** is an advanced Machine Learning application designed to forecast the price of Gold (XAU/USD). It utilizes historical financial data and macroeconomic indicators to predict future prices using multiple algorithms.

The application features a **Premium Dark & Gold UI**, offering interactive charts, real-time simulations, and deep market analysis.

## 🌟 Features

- **Multi-Model Prediction:** Choose between **Random Forest**, **Ridge Regression**, and **SVR** for forecasts.
- **Interactive Visualizations:** Powered by **Plotly**, analyze Time Series, Correlations, and Asset Relations.
- **Smart Data Handling:** Automatically loads the latest dataset and handles missing values.
- **Premium UI/UX:** A custom-designed Dark Mode with Gold accents for a professional financial terminal feel.
- **Scenario Analysis:** Modify inputs like S&P 500, Oil, Silver, and US Rates to see how they impact Gold prices.

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Python)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest, Ridge, SVR)
- **Visualization:** Plotly, Matplotlib, Seaborn

## 🚀 Installation & Setup

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/BerrkeUnal/GoldPrediction-ML-Model.git
cd GoldPrediction-ML-Model
2. Install Dependencies
Make sure you have Python installed. Then run:

Bash

pip install -r requirements.txt
3. Train the Models
Before running the app, you need to generate the pre-trained model files (.pkl). Run the model script once:

Bash

python model.py
This will create rf_model.pkl, scaler.pkl, etc.

4. Run the Application
Start the Streamlit interface:

Bash

streamlit run app.py
📂 Project Structure
├── app.py                   # Main Streamlit Application (Frontend)
├── model.py                 # ML Model Training & Evaluation Script
├── financial_regression.csv # Historical Dataset
├── requirements.txt         # Project Dependencies
├── *.pkl                    # Saved Model Files (Generated after running model.py)
└── README.md                # Project Documentation
📊 Dataset Indicators
The model uses the following features for prediction:

Global Markets: S&P 500, Nasdaq, US Rates (Interests)

Currencies: EUR/USD, USD/CHF, Dollar Index

Commodities: Silver, Oil, Platinum, Palladium

Macro: CPI (Inflation), GDP

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📜 License
This project is open-source and available under the MIT License.

Developed by Berke ÜNAL


---
```
