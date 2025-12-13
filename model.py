import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import Ridge 
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# --- STEP 1: UPLOADING FILE ---
try:
    
    df = pd.read_csv("financial_regression.csv", delimiter=",", encoding='latin1')
    print("File Uploaded.")
except FileNotFoundError:
    print("Error: File was not found.")
    exit()

#Selecting Important Columns
keep_columns = [
    'date', 'sp500 close', 'nasdaq close', 'us_rates_%', 'CPI', 'usd_chf', 
    'eur_usd', 'GDP', 'silver close', 'oil close', 'platinum close', 
    'palladium close', 'gold close'
]
df = df[keep_columns]


# --- STEP 2: CLEANING DATE COLUMN ---
numeric_dates = pd.to_numeric(df['date'], errors='coerce')
excel_dates = pd.to_datetime(numeric_dates[numeric_dates.notna()], unit='D', origin='1899-12-30')
text_dates = pd.to_datetime(df['date'][numeric_dates.isna()], errors='coerce')
df['date'] = pd.concat([excel_dates, text_dates])
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# --- STEP 3: CONVERTING TO NUMERICAL FORM ---
cols_to_convert = [col for col in df.columns if col != 'date']
for col in cols_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- STEP 4: FILLING MISSING VALUES ---
df.interpolate(method='time', inplace=True)
df.fillna(method='ffill', inplace=True)
df.dropna(axis=1, how='all', inplace=True) 

df['gold_lag1'] = df['gold close'].shift(1)
df.dropna(inplace=True)

# ---STEP 5: VISILUATION (EDA) ---

# 1. Histograms
try:
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Distribution of All Variables (Histogram)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("histograms.png")
    print("- histograms.png saved.")
except Exception as e:
    print(f"Histogram Error: {e}")
plt.close('all') 

# 2. Box Plots
try:
    df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(20, 16), sharex=False, sharey=False)
    plt.suptitle("Outlier Detection (Box Plot)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("box_plots.png")
    print("- box_plots.png saved.")
except Exception as e:
    print(f"Box Plot Error: {e}")
plt.close('all')

# 3. Correlation Heat Map
try:
    plt.figure(figsize=(12, 9))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8})
    plt.title('Correlation Heat Map between Variables')
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    print("- correlation_heatmap.png saved.")
except Exception as e:
    print(f"Correlation Error: {e}")
plt.close('all')

# 4. Pair Plot / Scatter Plot
try:
    target_col = "gold close"

    # Draw selection of top 4 features with the highest visibility.
    top_corr_features = corr_matrix[target_col].abs().sort_values(ascending=False).head(5).index
    
    sns.pairplot(df[top_corr_features], diag_kind='kde')
    plt.suptitle(f"Pair Plot of {target_col} vs Top Correlated Features", y=1.02)
    plt.savefig("pair_plot.png")
    print("- pair_plot.png saved.")
except Exception as e:
    print(f"Pair Plot Error: {e}")
plt.close('all')

# 5. Time Series Line Plot
try:
    df.plot(subplots=True, figsize=(15, 20), layout=(5, 3), sharex=True)
    plt.suptitle("Time Series Trend of All Variables", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("time_series_trends.png")
    print("- time_series_trends.png saved.")
except Exception as e:
    print(f"Time Series Plot Error: {e}")
plt.close('all')


# --- STEP 6: DATA PREPARATION  ---

# Separate into X (Features) and y (Target)
X = df.drop(columns=['gold close']) 
y = df['gold close'] 

print("\n--- Data Preparation ---")
print(f"Features (X) Shape: {X.shape}")
print(f"Target (y) Shape: {y.shape}")

# --- STEP 7: MODEL BUILDING & EVALUATION  ---

# 1. Train-Test Split (80% Train, 20% Test)
train_size = int(len(X) * 0.8)
X_train_raw, X_test_raw = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"\nTrain Indices: {X_train_raw.index[0]} -> {X_train_raw.index[-1]}")
print(f"Test Indices:  {X_test_raw.index[0]} -> {X_test_raw.index[-1]}")

# 2. Scaling (Fit on TRAIN, Transform TRAIN & TEST)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), index=X_train_raw.index, columns=X_train_raw.columns)
X_test = pd.DataFrame(scaler.transform(X_test_raw), index=X_test_raw.index, columns=X_test_raw.columns)

print("Data Scaled Successfully.")

# 3. Defining Models
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "SVR": SVR(kernel='rbf'),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# 4. Performance Evaluation Function (Template requires RMSE!)
def evaluate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    
    return {
        "Model": model_name, 
        "R2 Value": r2, 
        "RMSE": rmse,
        "MSE": mse, 
        "MAE": mae, 
        "MBE": mbe
    }

# 5. Cross-Validation Setup
# Template: "Technique: K-Fold"
kf = KFold(n_splits=5, shuffle=False)

train_results_list = []
test_results_list = []

print("\n--- MODEL PERFORMANCE RESULTS ---")

for name, model in models.items():
    print(f"Running {name}...")
    
    # A) TRAINING RESULTS
    y_train_cv_pred = cross_val_predict(model, X_train, y_train, cv=kf)
    train_metrics = evaluate_metrics(y_train, y_train_cv_pred, name)
    train_results_list.append(train_metrics)

    # B) TESTING RESULTS.
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_metrics(y_test, y_test_pred, name)
    test_results_list.append(test_metrics)


df_train_results = pd.DataFrame(train_results_list)
df_test_results = pd.DataFrame(test_results_list)

cols = ["Model", "R2 Value", "RMSE", "MSE", "MAE", "MBE"]
df_train_results = df_train_results[cols]
df_test_results = df_test_results[cols]

print("\n--- TRAINING RESULTS (Cross-Validation) ---")
print(df_train_results.round(4).to_string(index=False))

print("\n--- TESTING RESULTS (Hold-out Set) ---")
print(df_test_results.round(4).to_string(index=False))

# --- STEP 8: SAVE ALL MODELS ---
import joblib

print("\n--- Models Saving... ---")

# 1. Save Scaler
joblib.dump(scaler, 'scaler.pkl')

# 2. We retrain the models (with all the data) and save them
# Random Forest
rf_final = models["Random Forest"]
rf_final.fit(X_train, y_train)
joblib.dump(rf_final, 'rf_model.pkl')

# Ridge Regression
ridge_final = models["Ridge Regression"]
ridge_final.fit(X_train, y_train)
joblib.dump(ridge_final, 'ridge_model.pkl')

# SVM
svm_final = models["SVR"]
svm_final.fit(X_train, y_train)
joblib.dump(svm_final, 'svm_model.pkl')

print("All models (RF, Ridge, SVM) and Scaler saved successfully!")