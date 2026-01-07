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
# [關鍵參數] 用於還原/放大數據
DESIGN_PEAK_LOAD_KW = 20.0 

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
# 🛠️ 特徵工程
# ==========================================
def get_taiwan_holidays():
    holidays = [
        "2024-01-01", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", 
        "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-28", "2024-04-04", 
        "2024-04-05", "2024-05-01", "2024-06-10", "2024-09-17", "2024-10-10",
        "2025-01-01", "2025-01-25", "2025-01-26", "2025-01-27", "2025-01-28", "2025-01-29", 
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
# 🧠 主預測流程
# ==========================================
# [修復點] 這裡加上了 full_data_df 參數，解決 TypeError
def load_resources_and_predict(full_data_df=None):
    resources = {}
    try:
        # 1. 載入模型
        resources['lgbm'] = joblib.load(MODEL_FILES['lgbm'])
        resources['lstm'] = keras.models.load_model(MODEL_FILES['lstm'])
        resources['scaler_seq'] = joblib.load(MODEL_FILES['scaler_seq'])
        resources['scaler_dir'] = joblib.load(MODEL_FILES['scaler_dir'])
        resources['scaler_target'] = joblib.load(MODEL_FILES['scaler_target'])
        resources['weights'] = joblib.load(MODEL_FILES['weights'])
        
        # 2. 準備數據
        combined_df = None
        
        # [邏輯] 判斷資料來源並處理縮放
        is_scaled_input = False
        
        if full_data_df is not None and not full_data_df.empty:
            print("📥 [Model Service] 使用記憶體中的 DataFrame 進行預測...")
            combined_df = full_data_df.copy()
            # 檢查是否已經被 app_utils 放大過
            if combined_df['power_kW'].max() > 1.0:
                is_scaled_input = True
                print(f"ℹ️ [Model Service] 輸入數據已縮放 (Max > 1.0)，準備進行預測前還原...")
        else:
            print("⚠️ [Model Service] 未收到數據，啟動 Fallback 讀檔模式...")
            if not os.path.exists(MODEL_FILES['history_data']):
                return None, None
            hist_df = pd.read_csv(MODEL_FILES['history_data'])
            if 'datetime' in hist_df.columns: hist_df['timestamp'] = pd.to_datetime(hist_df['datetime'])
            elif 'timestamp' in hist_df.columns: hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            hist_df = hist_df.set_index('timestamp').sort_index()
            if 'power' in hist_df.columns: hist_df = hist_df.rename(columns={'power': 'power_kW'})
            combined_df = hist_df
        
        # 建立預測用的 DataFrame (df_for_model)，模型需要原始小數值 (0.x)
        df_for_model = combined_df.copy()
        
        # 如果輸入是大的 (20.0)，為了給模型吃，要除以倍率
        if is_scaled_input:
            df_for_model['power'] = df_for_model['power_kW'] / DESIGN_PEAK_LOAD_KW
        else:
            df_for_model['power'] = pd.to_numeric(df_for_model['power_kW'], errors='coerce')

        df_for_model = df_for_model.dropna(subset=['power'])
        
        # 3. 預測準備
        buffer_size = 2000
        df_ready = df_for_model.iloc[-buffer_size:].copy()
        last_time = df_ready.index[-1]
        
        future_dates = [last_time + timedelta(hours=i+1) for i in range(24)]
        future_df = pd.DataFrame(index=future_dates, columns=df_ready.columns)
        
        if 'temperature' in df_ready.columns: future_df['temperature'] = df_ready['temperature'].iloc[-1]
        else: future_df['temperature'] = 25.0
        if 'humidity' in df_ready.columns: future_df['humidity'] = df_ready['humidity'].iloc[-1]
        else: future_df['humidity'] = 70.0
        
        full_context = pd.concat([df_ready, future_df])
        
        # 4. 特徵工程
        df_lgbm = add_lgbm_features(full_context)
        df_lstm = add_lstm_features(full_context)
        
        target_feat_lgbm = df_lgbm.iloc[-24:]
        
        # --- LGBM 推論 ---
        lgbm_feature_names = resources['lgbm'].feature_name()
        X_lgbm = target_feat_lgbm[lgbm_feature_names]
        pred_lgbm = resources['lgbm'].predict(X_lgbm)
        
        # --- LSTM 推論 ---
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
        
        # --- 集成 (這是原始預測值 0.x) ---
        pred_final = (pred_lgbm * resources['weights']['w_lgbm']) + (pred_lstm * resources['weights']['w_lstm'])
        pred_final = np.maximum(pred_final, 0)

        # ==========================================
        # 🚀 輸出統一放大 (Reality Booster)
        # ==========================================
        # 為了讓 UI 圖表接合，我們必須回傳「大數值」
        scale_factor = DESIGN_PEAK_LOAD_KW
            
        # 1. 放大預測值
        pred_final_scaled = pred_final * scale_factor
        pred_lgbm_scaled = pred_lgbm * scale_factor
        pred_lstm_scaled = pred_lstm * scale_factor
        
        # 2. 準備回傳的歷史資料 (確保也是大的)
        ui_history_df = combined_df.copy()
        # 如果原本輸入就是大的，維持原樣；如果原本是讀檔(小的)，把它變大
        if not is_scaled_input:
             ui_history_df['power_kW'] = ui_history_df['power_kW'] * scale_factor
        
        # 打包結果
        result_df = pd.DataFrame({
            "時間": future_dates,
            "預測值": pred_final_scaled,
            "LGBM": pred_lgbm_scaled,
            "LSTM": pred_lstm_scaled
        }).set_index("時間")
        
        return result_df, ui_history_df
        
    except Exception as e:
        print(f"❌ [Model Service Error]: {e}")
        import traceback
        traceback.print_exc()
        return None, None