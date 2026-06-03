import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')

# 1. PREPROCESSING & SPLIT
# ==========================================

# Load Data
df_raw = pd.read_csv('data_hiv.csv', sep=';', parse_dates=['tahun'], dayfirst=True, index_col='tahun')
print("=== Head of Raw Data ===")
print(df_raw.head())

missing_values = df_raw.isnull().sum()
print('\nMissing values:', missing_values)

# Pisahkan Train & Test di AWAL untuk menghindari kebocoran data (Anti-Leakage)
train_size = int(len(df_raw) * 0.8)
train = df_raw.iloc[:train_size].copy()
test = df_raw.iloc[train_size:].copy()

target_scaler = MinMaxScaler()
exog_scaler = MinMaxScaler()

# Fit & Transform Target secara terisolasi
train['hiv_nasional_scaled'] = target_scaler.fit_transform(train[['hiv_nasional']])
test['hiv_nasional_scaled'] = target_scaler.transform(test[['hiv_nasional']])

# Fit & Transform Exogenous Variables secara terisolasi
exog_vars = ['layanan_konseling_tes', 'tes', 'layanan_pengobatan', 'pasien_arv']
train[exog_vars] = exog_scaler.fit_transform(train[exog_vars])
test[exog_vars] = exog_scaler.transform(test[exog_vars])

# Feature Engineering dilakukan mandiri pada masing-masing subset
for dataset in [train, test]:
    dataset['lag_1'] = dataset['hiv_nasional_scaled'].shift(1)
    dataset['lag_2'] = dataset['hiv_nasional_scaled'].shift(2)
    dataset['lag_3'] = dataset['hiv_nasional_scaled'].shift(3)
    dataset['rolling_mean_4'] = dataset['hiv_nasional_scaled'].rolling(window=4).mean()
    dataset['rolling_std_4'] = dataset['hiv_nasional_scaled'].rolling(window=4).std()
    dataset['quarter'] = dataset.index.quarter
    dataset['year'] = dataset.index.year

# Drop NaN akibat shift & rolling secara lokal
train = train.dropna()
test = test.dropna()

# Gabungkan kembali hanya untuk keperluan plot korelasi & EDA 
df_engineered = pd.concat([train, test])

print("\n=== Head of Processed Data ===")
print(df_engineered.head())

# Plot all scaled columns
plt.figure(figsize=(15, 10))
for col in df_engineered.columns:
    if 'scaled' in col or col in exog_vars:
        plt.plot(df_engineered.index, df_engineered[col], label=col)
plt.xlabel('Year')
plt.ylabel('Scaled Values')
plt.title('Scaled Data')
plt.legend()
plt.grid(True)
plt.show()

# Correlation Matrix
correlation_matrix = df_engineered.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Scaled Data')
plt.show()

# Define exogenous variables for SARIMAX and XGBoost
sarimax_exog_vars = ['layanan_konseling_tes', 'tes', 'layanan_pengobatan', 'pasien_arv']
xgb_exog_vars = sarimax_exog_vars + ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_4', 'rolling_std_4', 'quarter', 'year']

# Split the target variable (scaled)
train_target = train['hiv_nasional_scaled']
test_target = test['hiv_nasional_scaled']

# Split exogenous variables
sarimax_exog_train = train[sarimax_exog_vars]
sarimax_exog_test = test[sarimax_exog_vars]
xgb_exog_train = train[xgb_exog_vars]
xgb_exog_test = test[xgb_exog_vars]

print(f"\nTraining set size: {len(train)}")
print(f"Testing set size: {len(test)}")



# 2. SARIMAX ANALYSIS
# ==========================================

# Decompose the time series
decomposition = seasonal_decompose(df_engineered['hiv_nasional_scaled'], model='additive', period=4)
decomposition.plot()
plt.show()

# ADF Test function
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

print('\n--- ADF Test for hiv_nasional_scaled ---')
adf_test(df_engineered['hiv_nasional_scaled'])

# Differencing
df_engineered['hiv_nasional_diff'] = df_engineered['hiv_nasional_scaled'].diff().dropna()

print('\n--- ADF Test for differenced hiv_nasional_scaled ---')
adf_test(df_engineered['hiv_nasional_diff'].dropna())

# ACF & PACF Plots
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(df_engineered['hiv_nasional_diff'].dropna(), lags=15, ax=plt.gca())
plt.title('ACF for Differenced Series')

plt.subplot(2, 1, 2)
plot_pacf(df_engineered['hiv_nasional_diff'].dropna(), lags=15, ax=plt.gca())
plt.title('PACF for Differenced Series')
plt.tight_layout()
plt.show()

# Fit SARIMAX Model
model = SARIMAX(train_target,
                exog=sarimax_exog_train,
                order=(1, 1, 1),  
                seasonal_order=(1, 0, 1, 4),  
                enforce_stationarity=False,
                enforce_invertibility=False)

sarimax_results = model.fit(disp=False)
print("\n=== SARIMAX Model Summary ===")
print(sarimax_results.summary())

# Residual Analysis
residuals = sarimax_results.resid

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(residuals, lags=10, ax=plt.gca())
plt.title('ACF of SARIMAX Residuals')

plt.subplot(2, 1, 2)
plot_pacf(residuals, lags=10, ax=plt.gca())
plt.title('PACF of SARIMAX Residuals')
plt.tight_layout()
plt.show()

# Plot residual distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of SARIMAX Residuals')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.show()

# Forecast the test set using SARIMAX
forecast = sarimax_results.get_forecast(steps=len(test), exog=sarimax_exog_test)
forecast_mean_scaled = forecast.predicted_mean
forecast_ci_scaled = forecast.conf_int()

# Inverse transform to original scale
forecast_mean = target_scaler.inverse_transform(forecast_mean_scaled.values.reshape(-1, 1)).flatten()
forecast_ci_lower = target_scaler.inverse_transform(forecast_ci_scaled.iloc[:, 0].values.reshape(-1, 1)).flatten()
forecast_ci_upper = target_scaler.inverse_transform(forecast_ci_scaled.iloc[:, 1].values.reshape(-1, 1)).flatten()

actual_values = target_scaler.inverse_transform(test_target.values.reshape(-1, 1)).flatten()

# Plot Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(train.index, target_scaler.inverse_transform(train_target.values.reshape(-1, 1)), label='Train')
plt.plot(test.index, target_scaler.inverse_transform(test_target.values.reshape(-1, 1)), label='Test')
plt.plot(test.index, forecast_mean, label='Forecast')
plt.fill_between(test.index, forecast_ci_lower, forecast_ci_upper, color='k', alpha=0.1)
plt.title('HIV Cases Forecast with Exogenous Variables (Original Scale)')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()

print(f'\nSARIMAX Nilai prediksi: {forecast_mean}')
print(f'SARIMAX Nilai asli: {actual_values}')

rmse = np.sqrt(mean_squared_error(actual_values, forecast_mean))
mape = mean_absolute_percentage_error(actual_values, forecast_mean)
print(f'SARIMAX RMSE: {rmse}')
print(f'SARIMAX MAPE: {mape}')

# Residuals Tracking
train_pred_sarimax = sarimax_results.predict(start=train_target.index[0], end=train_target.index[-1], exog=sarimax_exog_train)
train_residuals = train_target - train_pred_sarimax
test_pred_sarimax = sarimax_results.predict(start=test_target.index[0], end=test_target.index[-1], exog=sarimax_exog_test)
test_residuals = test_target - test_pred_sarimax

plt.figure(figsize=(10, 6))
plt.plot(train_residuals, label='Training Residuals')
plt.plot(test_residuals, label='Test Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('SARIMAX Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Shapiro-Wilk Test
shapiro_train = shapiro(train_residuals)
print("\n--- Shapiro-Wilk test on training residuals ---")
print(f"Statistic: {shapiro_train.statistic}, P-value: {shapiro_train.pvalue}")

shapiro_test = shapiro(test_residuals)
print("--- Shapiro-Wilk test on testing residuals ---")
print(f"Statistic: {shapiro_test.statistic}, P-value: {shapiro_test.pvalue}")


# ==========================================
# 3. XGBOOST ANALYSIS
# ==========================================

# Add SARIMAX residuals as a feature to the XGBoost data securely
xgb_exog_train_extended = xgb_exog_train.copy()
xgb_exog_test_extended = xgb_exog_test.copy()
xgb_exog_train_extended['sarimax_residuals'] = train_residuals
xgb_exog_test_extended['sarimax_residuals'] = test_residuals

# Hyperparameter Tuning with TimeSeriesSplit
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model_base = XGBRegressor(random_state=42)
tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(estimator=xgb_model_base, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=tscv, verbose=0)
grid_search.fit(xgb_exog_train_extended, train_target)

best_params = grid_search.best_params_
print(f'\nBest parameters for XGBoost: {best_params}')

# Re-train with Best Params
xgb_model = XGBRegressor(**best_params, random_state=42)
xgb_model.fit(xgb_exog_train_extended, train_target)

# Predict using XGBoost
xgb_pred_scaled = xgb_model.predict(xgb_exog_test_extended)
xgb_pred = target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).flatten()

# Plot XGBoost Result
plt.figure(figsize=(12, 6))
plt.plot(test.index, actual_values, label='Actual', color='black')
plt.plot(test.index, xgb_pred, label='XGBoost Forecast', linestyle='--')
plt.title('HIV Cases Forecast with XGBoost')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()

print(f'\nXGBoost Nilai prediksi: {xgb_pred}')
print(f'XGBoost Nilai asli: {actual_values}')

rmse_xgb = np.sqrt(mean_squared_error(actual_values, xgb_pred))
mape_xgb = mean_absolute_percentage_error(actual_values, xgb_pred)
print(f'XGBoost RMSE: {rmse_xgb}')
print(f'XGBoost MAPE: {mape_xgb}')

# Feature Importance Analysis
importance_df = pd.DataFrame({
    'Feature': xgb_exog_train_extended.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance for XGBoost Model')
plt.show()


# 4. HYBRID MODEL & OPTIMIZATION
# ==========================================

# Test Alpha Weights Simulation (Murni Evaluasi)
alphas = np.arange(0.0, 1.05, 0.05)
results = []
for alpha in alphas:
    hybrid_pred_test = alpha * forecast_mean + (1 - alpha) * xgb_pred
    err_rmse = np.sqrt(mean_squared_error(actual_values, hybrid_pred_test))
    err_mape = mean_absolute_percentage_error(actual_values, hybrid_pred_test)
    results.append((alpha, err_rmse, err_mape))
results = np.array(results)

# Plot Alpha Optimization Optimization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(results[:, 0], results[:, 1], marker='o')
plt.title('RMSE vs Alpha')
plt.xlabel('Alpha (Weight SARIMAX)')
plt.ylabel('RMSE')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(results[:, 0], results[:, 2], marker='o', color='orange')
plt.title('MAPE vs Alpha')
plt.xlabel('Alpha (Weight SARIMAX)')
plt.ylabel('MAPE')
plt.grid(True)
plt.tight_layout()
plt.show()

# Kombinasi Ensembel Prediksi Validasi (Menggunakan bobot 0.55 SARIMAX + 0.45 XGBoost)
hybrid_forecast_mean_scaled = (forecast_sarimax_scaled * 0.55) + (xgb_pred_scaled * 0.45)
hybrid_forecast_mean = target_scaler.inverse_transform(hybrid_forecast_mean_scaled.values.reshape(-1, 1)).flatten()

rmse_hybrid = np.sqrt(mean_squared_error(actual_values, hybrid_forecast_mean))
mape_hybrid = mean_absolute_percentage_error(actual_values, hybrid_forecast_mean)

print(f'\nHybrid Model Prediction: {hybrid_forecast_mean}')
print(f'Hybrid Model Actual Test Data: {actual_values}')

# Future 8-Steps Out-of-Sample Forecast Preparation
future_exog_sarimax = sarimax_exog_test.tail(8).copy()
future_exog_xgb = xgb_exog_test.tail(8).copy()
future_exog_xgb['sarimax_residuals'] = 0.0 

# Future Forecast SARIMAX
future_forecast = sarimax_results.get_forecast(steps=8, exog=future_exog_sarimax)
future_forecast_mean_scaled = future_forecast.predicted_mean

# Future Forecast XGBoost
future_xgb_pred_scaled = xgb_model.predict(future_exog_xgb)

# Future Hybrid Forecast
future_hybrid_forecast_mean_scaled = (future_forecast_mean_scaled * 0.5) + (future_xgb_pred_scaled * 0.5)
future_hybrid_forecast_mean = target_scaler.inverse_transform(future_hybrid_forecast_mean_scaled.values.reshape(-1, 1)).flatten()

# Extend Dates
last_date = df_raw.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=8, freq='Q')

# Plot Final Forecast Comparison
plt.figure(figsize=(12, 6))
plt.plot(df_raw.index, df_raw['hiv_nasional'], label='Historical Data Data')
plt.plot(test.index, actual_values, label='Actual Test Data', color='black')
plt.plot(test.index, forecast_mean, label='SARIMAX Forecast', linestyle='-')
plt.plot(test.index, xgb_pred, label='XGBoost Predictions', linestyle='-')
plt.plot(test.index, hybrid_forecast_mean, label='Hybrid Forecast', linestyle=':')
plt.plot(future_dates, future_hybrid_forecast_mean, label='Future Hybrid Forecast', linestyle='--')
plt.title('HIV Cases Forecast with Hybrid Model (8 steps)')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()

print(f'\nFuture Hybrid Forecast Values (8 steps):\n{future_hybrid_forecast_mean}')

# 5. RECAP METRICS ALL MODELS
# ==========================================
print("\n" + "="*40)
print("       RECAP MODEL PERFORMANCE METRICS")
print("="*40)
print(f'SARIMAX      -> RMSE: {rmse:<12.4f} | MAPE: {mape:.4f}')
print(f'XGBoost      -> RMSE: {rmse_xgb:<12.4f} | MAPE: {mape_xgb:.4f}')
print(f'Hybrid Model -> RMSE: {rmse_hybrid:<12.4f} | MAPE: {mape_hybrid:.4f}')
print("="*40)