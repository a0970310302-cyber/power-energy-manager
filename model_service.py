# model_service.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import json
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras

# ==========================================
# 🚑 [設定] 抑制警告 & 相容性設定
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ==========================================
# ⚙️ 設定與常數
# ==========================================
MODEL_FILES = {
    "lgbm": "lgbm_model.pkl",
    "lstm": "lstm_model.keras",
    "scaler_seq": "scaler_seq.pkl",
    "scaler_dir": "scaler_dir.pkl",
    "scaler_target": "scaler_target.pkl",
    "weights": "ensemble_weights.pkl",
    "history_data": "final_training_data_with_humidity.csv"
}

LOOKBACK_HOURS = 168

# ==========================================
# 🛠️ 特徵工程 (維持原樣)
# ==========================================
def get_taiwan_holidays():
    """
    回傳台灣國定假日列表 (格式: YYYY-MM-DD)
    """
    holidays = [
        # --- 2024 ---
        "2024-01-01", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", 
        "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-28", "2024-04-04", 
        "2024-04-05", "2024-05-01", "2024-06-10", "2024-09-17", "2024-10-10",

        # --- 2025 ---
        "2025-01-01", 
        "2025-01-25", "2025-01-26", "2025-01-27", "2025-01-28", "2025-01-29", 
        "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02",
        "2025-02-28", "2025-03-01", "2025-03-02",
        "2025-04-03", "2025-04-04", "2025-04-05", "2025-04-06",
        "2025-05-01",
        "2025-05-30", "2025-05-31", "2025-06-01",
        "2025-09-27", "2025-09-28", "2025-09-29",
        "2025-10-04", "2025-10-05", "2025-10-06",
        "2025-10-10", "2025-10-11", "2025-10-12",
        "2025-10-24", "2025-10-25", "2025-10-26",
        "2025-12-25"
    ]
    return holidays

def add_lgbm_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    
    tw_holidays = get_taiwan_holidays()
    date_strs = df.index.strftime("%Y-%m-%d")
    
    df["is_holiday_or_weekend"] = ((df["day_of_week"] >= 5) | (date_strs.isin(tw_holidays))).astype(int)
    df["is_weekend"] = df["is_holiday_or_weekend"]
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    
    peak_hours = [10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
    df["is_peak"] = df["hour"].isin(peak_hours).astype(int)
    
    for lag in [24, 168, 720]:
         if f'lag_{lag}h' not in df.columns: df[f'lag_{lag}h'] = df["power"].shift(lag)
    for i in [1, 2, 3]:
        if f'temp_lag_{i}' not in df.columns: df[f'temp_lag_{i}'] = df["temperature"].shift(i)
    for window_days in [7, 14, 30]:
        window_hours = window_days * 24
        df[f'ma_{window_days}d'] = df["power"].shift(1).rolling(window=window_hours, min_periods=1).mean()
        df[f'std_{window_days}d'] = df["power"].shift(1).rolling(window=window_hours, min_periods=1).std()
    
    df["temp_x_peak"] = df["temperature"] * df["is_peak"]
    df["temp_squared"] = df["temperature"] ** 2
    return df

def add_lstm_features(df):
    df = df.copy()
    df["hour"] = df.index.hour.astype(float)
    
    tw_holidays = get_taiwan_holidays()
    date_strs = df.index.strftime("%Y-%m-%d")
    
    df["is_weekend"] = ((df.index.dayofweek >= 5) | (date_strs.isin(tw_holidays))).astype(float)
    df["day_of_week"] = df.index.dayofweek.astype(float)
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["temp_squared"] = df["temperature"] ** 2
    
    df["lag_24h"] = df["power"].shift(24)
    df["lag_168h"] = df["power"].shift(168)
    df["rolling_mean_24h_safe"] = df["power"].shift(24).rolling(window=24, min_periods=1).mean()
    df["rolling_std_24h_safe"] = df["power"].shift(24).rolling(window=24, min_periods=1).std()
    df["rolling_mean_168h"] = df["power"].shift(24).rolling(window=168, min_periods=1).mean()
    df["rolling_std_168h"] = df["power"].shift(24).rolling(window=168, min_periods=1).std()
    return df

# ==========================================
# 🧠 主預測流程 (Debug 顯影版)
# ==========================================
def load_resources_and_predict(full_data_df=None):
    """
    Debug 版：無 Try-Except 保護，錯誤會直接顯示在前端。
    """
    # 1. 載入模型資源
    resources = {}
    print("📥 [Model Service] 開始載入模型...")
    resources['lgbm'] = joblib.load(MODEL_FILES['lgbm'])
    resources['lstm'] = keras.models.load_model(MODEL_FILES['lstm'])
    resources['scaler_seq'] = joblib.load(MODEL_FILES['scaler_seq'])
    resources['scaler_dir'] = joblib.load(MODEL_FILES['scaler_dir'])
    resources['scaler_target'] = joblib.load(MODEL_FILES['scaler_target'])
    resources['weights'] = joblib.load(MODEL_FILES['weights'])
    
    # 2. 準備數據
    combined_df = None
    
    if full_data_df is not None and not full_data_df.empty:
        print("📥 [Model Service] 使用記憶體中的 DataFrame 進行預測...")
        combined_df = full_data_df.copy()
    else:
        print("⚠️ [Model Service] 未收到數據，啟動 Fallback 模式...")
        if not os.path.exists(MODEL_FILES['history_data']):
            raise FileNotFoundError(f"找不到檔案 {MODEL_FILES['history_data']}")
        
        hist_df = pd.read_csv(MODEL_FILES['history_data'])
        if 'datetime' in hist_df.columns:
                hist_df['timestamp'] = pd.to_datetime(hist_df['datetime'])
        elif 'timestamp' in hist_df.columns:
                hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
        hist_df = hist_df.set_index('timestamp').sort_index()
        if 'power' in hist_df.columns: 
            hist_df = hist_df.rename(columns={'power': 'power_kW'})
        combined_df = hist_df
        
    # 3. 資料對齊
    if 'power_kW' in combined_df.columns:
        combined_df['power'] = pd.to_numeric(combined_df['power_kW'], errors='coerce')
    elif 'power' in combined_df.columns:
        combined_df['power'] = pd.to_numeric(combined_df['power'], errors='coerce')
    else:
        raise ValueError("數據中找不到 power 或 power_kW 欄位")
        
    combined_df = combined_df.dropna(subset=['power'])
    
    # 4. 預測準備
    buffer_size = 2000 
    df_ready = combined_df.iloc[-buffer_size:].copy()
    last_time = df_ready.index[-1]
    
    future_dates = [last_time + timedelta(hours=i+1) for i in range(24)]
    future_df = pd.DataFrame(index=future_dates, columns=df_ready.columns)
    
    if 'temperature' in df_ready.columns:
        future_df['temperature'] = df_ready['temperature'].iloc[-1]
    else:
        future_df['temperature'] = 25.0
        
    if 'humidity' in df_ready.columns:
        future_df['humidity'] = df_ready['humidity'].iloc[-1]
    else:
            future_df['humidity'] = 70.0
    
    full_context = pd.concat([df_ready, future_df])
    
    # 5. 特徵工程
    df_lgbm = add_lgbm_features(full_context)
    df_lstm = add_lstm_features(full_context)
    
    target_feat_lgbm = df_lgbm.iloc[-24:]
    target_feat_lstm = df_lstm.iloc[-24:] # 輔助變數
    
    # --- LGBM 推論 ---
    lgbm_feature_names = resources['lgbm'].feature_name()
    for col in lgbm_feature_names:
        if col not in target_feat_lgbm.columns:
            target_feat_lgbm[col] = 0
            
    X_lgbm = target_feat_lgbm[lgbm_feature_names]
    pred_lgbm = resources['lgbm'].predict(X_lgbm)
    
    # --- LSTM 推論 ---
    current_idx = -25
    
    # 根據你的 .pkl 檔確認，這裡包含了 humidity，所以維持 6 個特徵
    seq_cols = ["power", "temperature", "humidity", "hour_sin", "hour_cos", "is_weekend"]
    dir_cols = ["lag_24h", "lag_168h", "temperature", "humidity", "hour_sin", "hour_cos", "week_sin", "week_cos", "is_weekend", "temp_squared", "rolling_mean_24h_safe", "rolling_std_24h_safe", "rolling_mean_168h", "rolling_std_168h"]
    
    for c in seq_cols + dir_cols:
        if c not in df_lstm.columns: df_lstm[c] = 0
    
    seq_data = df_lstm[seq_cols].iloc[current_idx-LOOKBACK_HOURS+1 : current_idx+1]
    dir_data = df_lstm[dir_cols].iloc[current_idx+1 : current_idx+2]
    
    X_seq = resources['scaler_seq'].transform(seq_data).reshape(1, LOOKBACK_HOURS, -1)
    X_dir = resources['scaler_dir'].transform(dir_data)
    
    pred_lstm_scaled = resources['lstm'].predict([X_seq, X_dir], verbose=0)
    pred_lstm_val = resources['scaler_target'].inverse_transform(pred_lstm_scaled).flatten()[0]
    pred_lstm = np.full(24, pred_lstm_val) 
    
    # --- 集成 ---
    pred_final = (pred_lgbm * resources['weights']['w_lgbm']) + (pred_lstm * resources['weights']['w_lstm'])
    
    result_df = pd.DataFrame({
        "時間": future_dates,
        "預測值": pred_final,
        "LGBM": pred_lgbm,
        "LSTM": pred_lstm
    }).set_index("時間")
    
    return result_df, combined_df