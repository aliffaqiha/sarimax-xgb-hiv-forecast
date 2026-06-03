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

# ==========================================
# 0. DATA GENERATOR ENGINE (REPLACING CSV)
# ==========================================
def generate_hiv_synthetic_dataset():
    np.random.seed(42)
    date_range = pd.date_range(start='2010-01-01', end='2026-12-31', freq='QE')
    n_periods = len(date_range)
    
    penduduk_nasional = np.linspace(240000000, 280000000, n_periods).astype(int)
    
    layanan_konseling_tes = np.linspace(1000, 5000, n_periods) + np.random.normal(0, 150, n_periods)
    tes = np.linspace(50000, 300000, n_periods) + np.random.normal(0, 8000, n_periods)
    layanan_pengobatan = np.linspace(500, 3500, n_periods) + np.random.normal(0, 100, n_periods)
    pasien_arv = np.linspace(20000, 180000, n_periods) + np.random.normal(0, 4000, n_periods)
    
    time_trend = np.linspace(5000, 14000, n_periods)
    seasonality = 800 * np.sin(2 * np.pi * date_range.quarter / 4)
    noise = np.random.normal(0, 400, n_periods)
    hiv_nasional = (time_trend + seasonality + noise).astype(int)
    
    tingkat_kematian = (hiv_nasional * 0.15) - (pasien_arv * 0.02) + np.random.normal(100, 20, n_periods)
    tingkat_kematian = np.clip(tingkat_kematian, 50, None).astype(int)
    
    hiv_jawa = (hiv_nasional * 0.56 + np.random.normal(0, 100, n_periods)).astype(int)
    hiv_sumatra = (hiv_nasional * 0.22 + np.random.normal(0, 50, n_periods)).astype(int)
    hiv_sulawesi = (hiv_nasional * 0.07 + np.random.normal(0, 20, n_periods)).astype(int)
    hiv_kalimantan = (hiv_nasional * 0.06 + np.random.normal(0, 15, n_periods)).astype(int)
    
    generated_df = pd.DataFrame({
        'hiv_nasional': hiv_nasional,
        'penduduk_nasional': penduduk_nasional,
        'layanan_konseling_tes': np.clip(layanan_konseling_tes, 100, None).astype(int),
        'tes': np.clip(tes, 10000, None).astype(int),
        'layanan_pengobatan': np.clip(layanan_pengobatan, 100, None).astype(int),
        'pasien_arv': np.clip(pasien_arv, 1000, None).astype(int),
        'tingkat_kematian': tingkat_kematian,
        'hiv_jawa': np.clip(hiv_jawa, 0, None),
        'hiv_sumatra': np.clip(hiv_sumatra, 0, None),
        'hiv_sulawesi': np.clip(hiv_sulawesi, 0, None),
        'hiv_kalimantan': np.clip(hiv_kalimantan, 0, None),
        'penduduk_jawa': (penduduk_nasional * 0.56).astype(int),
        'penduduk_sumatra': (penduduk_nasional * 0.22).astype(int),
        'penduduk_sulawesi': (penduduk_nasional * 0.07).astype(int),
        'penduduk_kalimantan': (penduduk_nasional * 0.06).astype(int)
    }, index=date_range)
    
    generated_df.index.name = 'tahun'
    return generated_df

df_raw = generate_hiv_synthetic_dataset()

# ==========================================
# 1. PIPELINE PREPROCESSING & SPLIT SECURE
# ==========================================

# Pembuatan Fitur Lag & Rolling secara GLOBAL di awal agar baris TEST tidak terpotong
df_raw['lag_1'] = df_raw['hiv_nasional'].shift(1)
df_raw['lag_2'] = df_raw['hiv_nasional'].shift(2)
df_raw['lag_3'] = df_raw['hiv_nasional'].shift(3)
df_raw['rolling_mean_4'] = df_raw['hiv_nasional'].rolling(window=4).mean()
df_raw['rolling_std_4'] = df_raw['hiv_nasional'].rolling(window=4).std()
df_raw['quarter'] = df_raw.index.quarter
df_raw['year'] = df_raw.index.year

# Drop baris awal yang bernilai NaN akibat pergeseran window secara terpusat
df_raw = df_raw.dropna()

# Pemisahan Train & Test Dataset (80:20) pasca pembersihan fitur
train_size = int(len(df_raw) * 0.8)
train = df_raw.iloc[:train_size].copy()
test = df_raw.iloc[train_size:].copy()

# Penskalaan Fitur secara terisolasi untuk menghindari Data Leakage
target_scaler = MinMaxScaler()
exog_scaler = MinMaxScaler()

train['hiv_nasional_scaled'] = target_scaler.fit_transform(train[['hiv_nasional']])
test['hiv_nasional_scaled'] = target_scaler.transform(test[['hiv_nasional']])

# Daftar variabel eksogen
sarimax_exog_vars = ['layanan_konseling_tes', 'tes', 'layanan_pengobatan', 'pasien_arv']
xgb_exog_vars = sarimax_exog_vars + ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_4', 'rolling_std_4', 'quarter', 'year']

# Fit & Transform Variabel Eksogen
train[xgb_exog_vars] = exog_scaler.fit_transform(train[xgb_exog_vars])
test[xgb_exog_vars] = exog_scaler.transform(test[xgb_exog_vars])

# Pisahkan Target & Eksogen untuk Model
train_target = train['hiv_nasional_scaled']
test_target = test['hiv_nasional_scaled']

sarimax_exog_train = train[sarimax_exog_vars]
sarimax_exog_test = test[sarimax_exog_vars]
xgb_exog_train = train[xgb_exog_vars]
xgb_exog_test = test[xgb_exog_vars]

print(f"Training set size: {len(train)}")
print(f"Testing set size: {len(test)}")

# ==========================================
# 2. SARIMAX ANALYSIS
# ==========================================
decomposition = seasonal_decompose(train['hiv_nasional_scaled'], model='additive', period=4)
decomposition.plot()
plt.show()

model = SARIMAX(train_target,
                exog=sarimax_exog_train,
                order=(1, 1, 1),  
                seasonal_order=(1, 0, 1, 4),  
                enforce_stationarity=False,
                enforce_invertibility=False)

sarimax_results = model.fit(disp=False)
print("\n=== SARIMAX Model Summary ===")
print(sarimax_results.summary())

# Menghitung Nilai Residual Secara Presisi & Selaras
train_pred_sarimax = sarimax_results.predict(start=train_target.index[0], end=train_target.index[-1], exog=sarimax_exog_train)
train_residuals = train_target - train_pred_sarimax

test_pred_sarimax = sarimax_results.predict(start=test_target.index[0], end=test_target.index[-1], exog=sarimax_exog_test)
test_residuals = test_target - test_pred_sarimax

# ==========================================
# 3. XGBOOST ANALYSIS
# ==========================================
xgb_exog_train_extended = xgb_exog_train.copy()
xgb_exog_test_extended = xgb_exog_test.copy()
xgb_exog_train_extended['sarimax_residuals'] = train_residuals
xgb_exog_test_extended['sarimax_residuals'] = test_residuals

param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
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

xgb_model = XGBRegressor(**best_params, random_state=42)
xgb_model.fit(xgb_exog_train_extended, train_target)

xgb_pred_scaled = xgb_model.predict(xgb_exog_test_extended)
xgb_pred = target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).flatten()

actual_values = target_scaler.inverse_transform(test_target.values.reshape(-1, 1)).flatten()
rmse_xgb = np.sqrt(mean_squared_error(actual_values, xgb_pred))
mape_xgb = mean_absolute_percentage_error(actual_values, xgb_pred)

# ==========================================
# 4. HYBRID MODEL & FUTURE FORECAST (8 STEPS)
# ==========================================
forecast_sarimax_scaled = sarimax_results.get_forecast(steps=len(test), exog=sarimax_exog_test).predicted_mean
forecast_mean = target_scaler.inverse_transform(forecast_sarimax_scaled.values.reshape(-1, 1)).flatten()

hybrid_forecast_mean_scaled = (forecast_sarimax_scaled * 0.55) + (xgb_pred_scaled * 0.45)
hybrid_forecast_mean = target_scaler.inverse_transform(hybrid_forecast_mean_scaled.values.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(actual_values, forecast_mean))
mape = mean_absolute_percentage_error(actual_values, forecast_mean)

rmse_hybrid = np.sqrt(mean_squared_error(actual_values, hybrid_forecast_mean))
mape_hybrid = mean_absolute_percentage_error(actual_values, hybrid_forecast_mean)

# Proyeksi ke Depan
future_exog_sarimax = sarimax_exog_test.tail(8).copy()
future_exog_xgb = xgb_exog_test.tail(8).copy()
future_exog_xgb['sarimax_residuals'] = 0.0  

future_forecast = sarimax_results.get_forecast(steps=8, exog=future_exog_sarimax)
future_forecast_mean_scaled_out = future_forecast.predicted_mean
future_xgb_pred_scaled_out = xgb_model.predict(future_exog_xgb)

future_hybrid_forecast_mean_scaled = (future_forecast_mean_scaled_out * 0.5) + (future_xgb_pred_scaled_out * 0.5)
future_hybrid_forecast_mean = target_scaler.inverse_transform(future_hybrid_forecast_mean_scaled.values.reshape(-1, 1)).flatten()

last_date = df_raw.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=8, freq='QE')

# Plot Visualisasi
plt.figure(figsize=(12, 6))
plt.plot(df_raw.index, df_raw['hiv_nasional'], label='Historical Data (Generated)')
plt.plot(test.index, actual_values, label='Actual Test Data', color='black')
plt.plot(test.index, hybrid_forecast_mean, label='Hybrid Forecast (Test Segment)', linestyle=':')
plt.plot(future_dates, future_hybrid_forecast_mean, label='Future Hybrid Forecast (8 Steps Ahead)', linestyle='--')
plt.title('HIV Cases Forecast with Hybrid Model (Fixed Reshape Alignment)')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()

print(f'\nFuture Hybrid Forecast Values (8 steps ahead):\n{future_hybrid_forecast_mean}')

# ==========================================
# 5. RECAP METRICS ALL MODELS
# ==========================================
print("\n" + "="*50)
print("       RECAP MODEL PERFORMANCE METRICS")
print("="*50)
print(f'SARIMAX      -> RMSE: {rmse:<12.4f} | MAPE: {mape:.4f}')
print(f'XGBoost      -> RMSE: {rmse_xgb:<12.4f} | MAPE: {mape_xgb:.4f}')
print(f'Hybrid Model -> RMSE: {rmse_hybrid:<12.4f} | MAPE: {mape_hybrid:.4f}')
print("="*50)