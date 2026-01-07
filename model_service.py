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
# 🛠️ 特徵工程 (已更新 2025 完整假日)
# ==========================================
def get_taiwan_holidays():
    """
    回傳台灣國定假日列表 (格式: YYYY-MM-DD)
    包含 2025 年新制恢復的七天假 (行憲、光復、教師節等)
    """
    holidays = [
        # --- 2024 (保留舊資料供模型參考) ---
        "2024-01-01", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", 
        "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-28", "2024-04-04", 
        "2024-04-05", "2024-05-01", "2024-06-10", "2024-09-17", "2024-10-10",

        # --- 2025 (依據新制修訂) ---
        "2025-01-01", # 元旦
        
        # 農曆春節 (1/25-2/2，共9天)
        # 1/27(一)調整放假, 1/28(除夕)-1/31(初三)
        "2025-01-25", "2025-01-26", "2025-01-27", "2025-01-28", "2025-01-29", 
        "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02",
        
        # 228 和平紀念日 (週五)
        "2025-02-28", "2025-03-01", "2025-03-02",
        
        # 兒童節與清明節 (4/3-4/6)
        # 4/4兒童節適逢週五，4/3(四)補假
        "2025-04-03", "2025-04-04", "2025-04-05", "2025-04-06",
        
        # 勞動節 (5/1) - 新制全國放假
        "2025-05-01",
        
        # 端午節 (5/30-6/1)
        # 5/31(六)端午節，5/30(五)補假
        "2025-05-30", "2025-05-31", "2025-06-01",
        
        # 教師節 (9/27-9/29)
        # 9/28(日)教師節，9/29(一)補假 (新制)
        "2025-09-27", "2025-09-28", "2025-09-29",
        
        # 中秋節 (10/4-10/6)
        # 10/6(一)中秋節
        "2025-10-04", "2025-10-05", "2025-10-06",
        
        # 國慶日 (10/10-10/12)
        # 10/10(五)國慶日
        "2025-10-10", "2025-10-11", "2025-10-12",
        
        # 台灣光復節 (10/24-10/26)
        # 10/25(六)光復節，10/24(五)補假 (新制)
        "2025-10-24", "2025-10-25", "2025-10-26",
        
        # 行憲紀念日 (12/25)
        # 12/25(四) (新制恢復放假)
        "2025-12-25"
    ]
    return holidays

def add_lgbm_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    
    # 呼叫更新後的假日函式
    tw_holidays = get_taiwan_holidays()
    date_strs = df.index.strftime("%Y-%m-%d")
    
    # 判斷是否為假日 (週末 OR 國定假日)
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
    
    # LSTM 也需要同步使用新的假日邏輯
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
# 🧠 主預測流程 (離線版)
# ==========================================
def load_resources_and_predict():
    resources = {}
    try:
        # 1. 載入模型
        resources['lgbm'] = joblib.load(MODEL_FILES['lgbm'])
        resources['lstm'] = keras.models.load_model(MODEL_FILES['lstm'])
        resources['scaler_seq'] = joblib.load(MODEL_FILES['scaler_seq'])
        resources['scaler_dir'] = joblib.load(MODEL_FILES['scaler_dir'])
        resources['scaler_target'] = joblib.load(MODEL_FILES['scaler_target'])
        resources['weights'] = joblib.load(MODEL_FILES['weights'])
        
        # 2. 準備數據 (僅使用 CSV)
        print("📥 [Offline Mode] 讀取本地歷史數據...")
        
        if not os.path.exists(MODEL_FILES['history_data']):
            print(f"❌ 錯誤：找不到檔案 {MODEL_FILES['history_data']}")
            return None, None
            
        hist_df = pd.read_csv(MODEL_FILES['history_data'])
        
        if 'datetime' in hist_df.columns:
             hist_df['timestamp'] = pd.to_datetime(hist_df['datetime'])
        elif 'timestamp' in hist_df.columns:
             hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
        else:
             print("❌ 錯誤：CSV 中找不到 datetime 或 timestamp 欄位")
             return None, None

        hist_df = hist_df.set_index('timestamp').sort_index()
        
        if 'power' in hist_df.columns: 
            hist_df = hist_df.rename(columns={'power': 'power_kW'})
            
        if 'power_kW' not in hist_df.columns:
             print("❌ 錯誤：CSV 中找不到 power 或 power_kW 欄位")
             return None, None

        combined_df = hist_df.copy()
        combined_df['power'] = combined_df['power_kW']
        combined_df['power'] = pd.to_numeric(combined_df['power'], errors='coerce')
        combined_df = combined_df.dropna(subset=['power'])

        print(f"🎉 [Total] 資料載入完畢！資料範圍: {combined_df.index.min()} ~ {combined_df.index.max()}")

        # 3. 預測準備 (基準時間為 CSV 最後一筆)
        buffer_size = 2000
        df_ready = combined_df.iloc[-buffer_size:].copy()
        
        last_time = df_ready.index[-1]
        print(f"🔮 預測基準時間 (Last Data Point): {last_time}")
        
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
        
        # 4. 特徵工程 (使用更新後的假日列表)
        df_lgbm = add_lgbm_features(full_context)
        df_lstm = add_lstm_features(full_context)
        
        target_feat_lgbm = df_lgbm.iloc[-24:]
        target_feat_lstm = df_lstm.iloc[-24:]
        
        # --- LGBM ---
        lgbm_feature_names = resources['lgbm'].feature_name()
        X_lgbm = target_feat_lgbm[lgbm_feature_names]
        pred_lgbm = resources['lgbm'].predict(X_lgbm)
        
        # --- LSTM ---
        current_idx = -25
        seq_cols = ["power", "temperature", "humidity", "hour_sin", "hour_cos", "is_weekend"]
        dir_cols = ["lag_24h", "lag_168h", "temperature", "humidity", "hour_sin", "hour_cos", "week_sin", "week_cos", "is_weekend", "temp_squared", "rolling_mean_24h_safe", "rolling_std_24h_safe", "rolling_mean_168h", "rolling_std_168h"]
        
        for c in seq_cols + dir_cols:
            if c not in df_lstm.columns: df_lstm[c] = 0
        
        seq_data = df_lstm[seq_cols].iloc[current_idx-LOOKBACK_HOURS+1 : current_idx+1]
        dir_data = df_lstm[dir_cols].iloc[current_idx+1 : current_idx+2]
        
        X_seq = resources['scaler_seq'].transform(seq_data).reshape(1, LOOKBACK_HOURS, -1)
        X_dir = resources['scaler_dir'].transform(dir_data)
        
        pred_lstm_scaled = resources['lstm'].predict([X_seq, X_dir], verbose=0)
        pred_lstm = resources['scaler_target'].inverse_transform(pred_lstm_scaled).flatten()
        
        # --- 集成 ---
        pred_final = (pred_lgbm * resources['weights']['w_lgbm']) + (pred_lstm * resources['weights']['w_lstm'])
        
        result_df = pd.DataFrame({
            "時間": future_dates,
            "預測值": pred_final,
            "LGBM": pred_lgbm,
            "LSTM": pred_lstm
        }).set_index("時間")
        
        return result_df, combined_df
        
    except Exception as e:
        print(f"❌ [Model Service Error]: {e}")
        import traceback
        traceback.print_exc()
        return None, None