import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression 
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import joblib
import shap

# =============================
# --- STEP 0: FOLDER SETUP ---
# =============================

# Defining Folders
eda_folder = "EDA_Results"
model_folder = "Saved_Models"

# Creating folders if they dont exist
os.makedirs(eda_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

print(f"Directories created/checked: '{eda_folder}' and '{model_folder}'")

# ===============================
# --- STEP 1: UPLOADING FILE ---
# ===============================

try:
    url = "https://raw.githubusercontent.com/BerrkeUnal/GoldPrediction-ML-Model/main/financial_regression.csv" 
    df = pd.read_csv(url, delimiter=",", encoding='latin1')
    print("File Uploaded.")
except FileNotFoundError:
    print("Error: File was not found.")
    exit()

# Selecting Important Columns
keep_columns = [
    'date', 'sp500 close', 'nasdaq close', 'us_rates_%', 'CPI', 'usd_chf', 
    'eur_usd', 'GDP', 'silver close', 'oil close', 'platinum close', 
    'palladium close', 'gold close'
]
df = df[keep_columns]

# =====================================
# --- STEP 2: CLEANING DATE COLUMN ---
# =====================================

numeric_dates = pd.to_numeric(df['date'], errors='coerce')
excel_dates = pd.to_datetime(numeric_dates[numeric_dates.notna()], unit='D', origin='1899-12-30')
text_dates = pd.to_datetime(df['date'][numeric_dates.isna()], errors='coerce')
df['date'] = pd.concat([excel_dates, text_dates])
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# ============================================
# --- STEP 3: CONVERTING TO NUMERICAL FORM ---
# ============================================

cols_to_convert = [col for col in df.columns if col != 'date']
for col in cols_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================
# --- STEP 4: FILLING MISSING VALUES ---
# ============================================

df.interpolate(method='time', inplace=True)
df.fillna(method='ffill', inplace=True)
df.dropna(axis=1, how='all', inplace=True) 

df['gold_lag1'] = df['gold close'].shift(1)
df.dropna(inplace=True)

# ============================================
# --- STEP 5: VISUALIZATION (EDA) ---
# ============================================

print("\n--- Saving EDA Images to folder... ---")

# Statistical Summary
stats = df.describe().transpose()
stats.to_csv(os.path.join(eda_folder, "statistical_summary.csv"))
print(f"- Saved: {os.path.join(eda_folder, 'statistical_summary.csv')} (For Report Table)")

# 1. Histograms
try:
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Distribution of All Variables (Histogram)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(eda_folder, "histograms.png")
    plt.savefig(save_path)
    print(f"- Saved: {save_path}")
except Exception as e:
    print(f"Histogram Error: {e}")
plt.close('all') 

# 2. Box Plots
try:
    df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(20, 16), sharex=False, sharey=False)
    plt.suptitle("Outlier Detection (Box Plot)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(eda_folder, "box_plots.png")
    plt.savefig(save_path)
    print(f"- Saved: {save_path}")
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
    
    save_path = os.path.join(eda_folder, "correlation_heatmap.png")
    plt.savefig(save_path)
    print(f"- Saved: {save_path}")
except Exception as e:
    print(f"Correlation Error: {e}")
plt.close('all')

# 4. Pair Plot / Scatter Plot
try:
    target_col = "gold close"
    top_corr_features = corr_matrix[target_col].abs().sort_values(ascending=False).head(5).index
    
    sns.pairplot(df[top_corr_features], diag_kind='kde')
    plt.suptitle(f"Pair Plot of {target_col} vs Top Correlated Features", y=1.02)
    
    save_path = os.path.join(eda_folder, "pair_plot.png")
    plt.savefig(save_path)
    print(f"- Saved: {save_path}")
except Exception as e:
    print(f"Pair Plot Error: {e}")
plt.close('all')

# 5. Time Series Line Plot
try:
    df.plot(subplots=True, figsize=(15, 20), layout=(5, 3), sharex=True)
    plt.suptitle("Time Series Trend of All Variables", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(eda_folder, "time_series_trends.png")
    plt.savefig(save_path)
    print(f"- Saved: {save_path}")
except Exception as e:
    print(f"Time Series Plot Error: {e}")
plt.close('all')


# ============================================
# --- STEP 6: DATA PREPARATION ---
# ============================================

# Separate into X (Features) and y (Target)
X = df.drop(columns=['gold close']) 
y = df['gold close'] 

print("\n--- Data Preparation ---")
print(f"Features (X) Shape: {X.shape}")
print(f"Target (y) Shape: {y.shape}")

# ============================================
# --- STEP 7: MODEL BUILDING & EVALUATION ---
# ============================================

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

# GROUP 1: Non-Ensemble Models
non_ensemble_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "SVR": SVR(kernel='rbf')
}

# GROUP 2: Ensemble Models
ensemble_models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
}

# 4. Performance Evaluation Function
def evaluate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "Model": model_name, 
        "RMSE": rmse,
        "R2": r2, 
        "MAE": mae, 
        "MAPE": mape
    }

# 5. Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=False)

def run_models(model_dict, group_name):
    results_list = []
    print(f"\n--- Running {group_name} ---")
    
    for name, model in model_dict.items():
        print(f"Training {name}...")
        
        # A) Cross-Validation
        try:
            y_train_cv_pred = cross_val_predict(model, X_train, y_train, cv=kf)
            cv_metrics = evaluate_metrics(y_train, y_train_cv_pred, name)
            print(f"   > CV Score (RMSE): {cv_metrics['RMSE']:.4f}") # Konsola bilgi basar
        except Exception as e:
            print(f"   > CV Error: {e}")


        # B) Hold-out Test
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        
        metrics = evaluate_metrics(y_test, y_test_pred, name)
        results_list.append(metrics)
    
    return pd.DataFrame(results_list)

# Running the models
df_non_ensemble = run_models(non_ensemble_models, "Non-Ensemble Models")
df_ensemble = run_models(ensemble_models, "Ensemble Models")

print("\n" + "="*50)
print("REPORT TABLE 1: NON-ENSEMBLE MODELS RESULTS")
print("="*50)
cols = ["Model", "RMSE", "R2", "MAE", "MAPE"]
print(df_non_ensemble[cols].round(4).to_string(index=False))

print("\n" + "="*50)
print("REPORT TABLE 2: ENSEMBLE MODELS RESULTS (Before Tuning)")
print("="*50)
print(df_ensemble[cols].round(4).to_string(index=False))


# ==========================================================
# --- STEP 7.5: HYPERPARAMETER TUNING (ENSEMBLE MODELS) ---
# ==========================================================

print("\n" + "="*50)
print("STARTING HYPERPARAMETER TUNING (GridSearchCV)...")
print("="*50)

# 1. Define Parameter Grids for Each Ensemble Model
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5]
    },
    "XGBoost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5]
    }
}

tuned_results_list = []
best_params_dict = {}

# 2. Loop Through Ensemble Models and Perform GridSearch
for name, model in ensemble_models.items():
    print(f"Tuning {name}...")
    
    # Initialize GridSearchCV
    # cv=3 is used to save time, you can increase to 5 for better results
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], 
                               cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=0)
    
    # Fit GridSearch
    grid_search.fit(X_train, y_train)
    
    # Get Best Model and Params
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_params_dict[name] = best_params
    
    # Update the model in our dictionary to the tuned version for saving later
    ensemble_models[name] = best_model
    
    # Predict using the best model
    y_test_pred_tuned = best_model.predict(X_test)
    
    # Evaluate Metrics
    metrics = evaluate_metrics(y_test, y_test_pred_tuned, name)
    tuned_results_list.append(metrics)

# 3. Print Best Hyperparameters (For Report Table: "Hyperparameters Tuned")
print("\n" + "="*50)
print("REPORT TABLE: BEST HYPERPARAMETERS FOUND")
print("="*50)
for name, params in best_params_dict.items():
    print(f"Model: {name}")
    print(f"Best Values: {params}")
    print("-" * 30)

# 4. Print Performance After Tuning (For Report Table: "After Hyperparameter tuning")
df_tuned_results = pd.DataFrame(tuned_results_list)
print("\n" + "="*50)
print("REPORT TABLE 3: ENSEMBLE MODELS RESULTS (AFTER TUNING)")
print("="*50)
print(df_tuned_results[cols].round(4).to_string(index=False))

print("\nOptimization Complete. Proceeding to Save Models...")


# =============================================
# --- STEP 7.6: FEATURE IMPORTANCE ANALYSIS ---
# =============================================
print("\n" + "="*50)
print("GENERATING FEATURE IMPORTANCE GRAPHS...")
print("="*50)

feature_names = X.columns

for name, model in ensemble_models.items():
    try:
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Create Plot
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance - {name}")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        
        # Save Plot
        filename = f"feature_importance_{name.replace(' ', '_')}.png"
        save_path = os.path.join(eda_folder, filename)
        plt.savefig(save_path)
        print(f"- Saved: {save_path}")
        plt.close()
        
    except AttributeError:
        print(f"Warning: {name} does not provide feature importances.")
    except Exception as e:
        print(f"Error plotting feature importance for {name}: {e}")

print("\nFeature Analysis Complete.")

# ============================================
# --- STEP 7.7: SHAP ANALYSIS ---
# ============================================

print("\n" + "="*50)
print("STARTING SHAP ANALYSIS (BONUS TASK)...")
print("="*50)

shap_model_name = "XGBoost" 
best_model_for_shap = ensemble_models[shap_model_name]

print(f"Generating SHAP plots for: {shap_model_name}...")

try:
    # 1. Create Explainer
    explainer = shap.TreeExplainer(best_model_for_shap)
    
    # 2. Calculate SHAP Values 
    shap_values = explainer.shap_values(X_test)

    # 3. Summary Plot (Beeswarm)
    plt.figure()
    plt.title(f"SHAP Summary Plot - {shap_model_name}")
    shap.summary_plot(shap_values, X_test, show=False)
    
    save_path = os.path.join(eda_folder, "shap_summary_plot.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"- Saved: {save_path}")
    plt.close()

    # 4. Feature Importance Bar Plot (SHAP based)
    plt.figure()
    plt.title(f"SHAP Feature Importance - {shap_model_name}")
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    
    save_path = os.path.join(eda_folder, "shap_feature_importance.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"- Saved: {save_path}")
    plt.close()

    print("SHAP Analysis Completed Successfully.")

except Exception as e:
    print(f"Error during SHAP Analysis: {e}")
    print("Ensure 'shap' library is installed via 'pip install shap'")

# ============================================
# --- STEP 8: SAVE ALL MODELS ---
# ============================================
print("\n--- Saving Models to folder... ---")

# 1. Save Scaler
joblib.dump(scaler, os.path.join(model_folder, 'scaler.pkl'))
print(f"- Scaler saved: {os.path.join(model_folder, 'scaler.pkl')}")

# 2. Save Non-Ensemble Models
# Ridge Regression
ridge_final = non_ensemble_models["Ridge Regression"]
ridge_final.fit(X_train, y_train)
joblib.dump(ridge_final, os.path.join(model_folder, 'ridge_model.pkl'))
print(f"- Ridge saved: {os.path.join(model_folder, 'ridge_model.pkl')}")

# SVR
svm_final = non_ensemble_models["SVR"]
svm_final.fit(X_train, y_train)
joblib.dump(svm_final, os.path.join(model_folder, 'svm_model.pkl'))
print(f"- SVR saved: {os.path.join(model_folder, 'svm_model.pkl')}")

# Linear Regression
lr_final = non_ensemble_models["Linear Regression"]
lr_final.fit(X_train, y_train)
joblib.dump(lr_final, os.path.join(model_folder, 'linear_model.pkl'))
print(f"- Linear Regression saved: {os.path.join(model_folder, 'linear_model.pkl')}")

# 3. Save Ensemble Models
# Random Forest
rf_final = ensemble_models["Random Forest"]
rf_final.fit(X_train, y_train)
joblib.dump(rf_final, os.path.join(model_folder, 'rf_model.pkl'))
print(f"- Random Forest saved: {os.path.join(model_folder, 'rf_model.pkl')}")

# Gradient Boosting
gb_final = ensemble_models["Gradient Boosting"]
gb_final.fit(X_train, y_train)
joblib.dump(gb_final, os.path.join(model_folder, 'gb_model.pkl'))
print(f"- Gradient Boosting saved: {os.path.join(model_folder, 'gb_model.pkl')}")

# XGBoost
xgb_final = ensemble_models["XGBoost"]
xgb_final.fit(X_train, y_train)
joblib.dump(xgb_final, os.path.join(model_folder, 'xgb_model.pkl'))
print(f"- XGBoost saved: {os.path.join(model_folder, 'xgb_model.pkl')}")

print("\nExecution Completed. Check 'EDA_Results' and 'Saved_Models' folders!")