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
# 🚑 [設定] 抑制警告
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ==========================================
# ⚙️ 設定與常數
# ==========================================
DESIGN_PEAK_LOAD_KW = 20.0 

MODEL_FILES = {
    "config": "hybrid_residual.pkl",
    "lgbm": "lgbm_residual.pkl",
    "lstm": "lstm_hybrid.keras",
    "history_data": "final_training_data_with_humidity.csv"
}

LOOKBACK_HOURS = 168

# ==========================================
# 🛠️ 特徵工程
# ==========================================
def get_taiwan_holidays():
    holidays = [
        "2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", 
        "2022-02-04", "2022-02-28", "2022-04-04", "2022-04-05", "2022-05-01", 
        "2022-06-03", "2022-09-09", "2022-10-10",

        "2023-01-01", "2023-01-02", "2023-01-20", "2023-01-21", "2023-01-22", 
        "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
        "2023-02-27", "2023-02-28", "2023-04-03", "2023-04-04", 
        "2023-04-05", "2023-05-01", "2023-06-22", "2023-06-23",
        "2023-09-29", "2023-10-09", "2023-10-10",

        "2024-01-01", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", 
        "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-28", "2024-04-04", 
        "2024-04-05", "2024-05-01", "2024-06-10", "2024-09-17", "2024-10-10",

        "2025-01-01", "2025-01-25", "2025-01-26", "2025-01-27", "2025-01-28", 
        "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02",
        "2025-02-28", "2025-04-03", "2025-04-04", "2025-05-01", "2025-05-31", 
        "2025-10-06", "2025-10-10", "2025-10-11", "2025-10-12", "2025-10-24", 
        "2025-10-25", "2025-10-26", "2025-12-25"
    ]
    return holidays

def create_hybrid_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    tw_holidays = get_taiwan_holidays()
    date_strs = df.index.strftime("%Y-%m-%d")
    df["is_weekend"] = ((df["day_of_week"] >= 5) | (date_strs.isin(tw_holidays))).astype(int)
    
    df["temp_squared"] = df["temperature"] ** 2
    df["temp_humidity"] = df["temperature"] * df["humidity"]
    
    for w in [24, 72]:
        df[f'temp_roll_{w}'] = df['temperature'].rolling(window=w, min_periods=1).mean()
        
    df['rolling_mean_24h'] = df['power'].shift(1).rolling(window=24, min_periods=1).mean()
    df['rolling_max_24h'] = df['power'].shift(1).rolling(window=24, min_periods=1).max()
    df['rolling_min_24h'] = df['power'].shift(1).rolling(window=24, min_periods=1).min()
    df['rolling_mean_7d'] = df['power'].shift(1).rolling(window=168, min_periods=1).mean()
    df['rolling_mean_3h'] = df['power'].shift(1).rolling(window=3, min_periods=1).mean() 
    
    for lag in [24, 48, 168]:
        df[f'lag_{lag}'] = df['power'].shift(lag)
        df[f'lag_{lag}h'] = df['power'].shift(lag) 
        
    df['diff_24_48'] = df['lag_24'] - df['lag_48']
    return df

# ==========================================
# 🧠 主預測流程
# ==========================================
def load_resources_and_predict(full_data_df=None):
    resources = {}
    try:
        # 1. 載入資源
        config = joblib.load(MODEL_FILES['config'])
        resources['lgbm'] = joblib.load(MODEL_FILES['lgbm'])
        resources['lstm'] = keras.models.load_model(MODEL_FILES['lstm'])
        resources['scaler_seq'] = config['scaler_seq']
        resources['scaler_direct'] = config.get('scaler_direct', None)
        
        # 2. 準備數據
        combined_df = None
        is_scaled_input = False
        
        if full_data_df is not None and not full_data_df.empty:
            combined_df = full_data_df.copy()
            if combined_df['power_kW'].max() > 1.0:
                is_scaled_input = True
        else:
            if not os.path.exists(MODEL_FILES['history_data']): return None, None
            hist_df = pd.read_csv(MODEL_FILES['history_data'])
            if 'datetime' in hist_df.columns: hist_df['timestamp'] = pd.to_datetime(hist_df['datetime'])
            elif 'timestamp' in hist_df.columns: hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            hist_df = hist_df.set_index('timestamp').sort_index()
            if 'power' in hist_df.columns: hist_df = hist_df.rename(columns={'power': 'power_kW'})
            combined_df = hist_df

        df_model = combined_df.copy()
        if is_scaled_input:
            df_model['power'] = df_model['power_kW'] / DESIGN_PEAK_LOAD_KW
        else:
            df_model['power'] = pd.to_numeric(df_model['power_kW'], errors='coerce')
        
        df_model = df_model.dropna(subset=['power'])
        
        # 3. 預測準備
        buffer_size = 2000
        df_ready = df_model.iloc[-buffer_size:].copy()
        last_time = df_ready.index[-1]
        future_dates = [last_time + timedelta(hours=i+1) for i in range(24)]
        future_df = pd.DataFrame(index=future_dates, columns=df_ready.columns)
        
        if 'temperature' in df_ready.columns: future_df['temperature'] = df_ready['temperature'].iloc[-1]
        else: future_df['temperature'] = 25.0
        if 'humidity' in df_ready.columns: future_df['humidity'] = df_ready['humidity'].iloc[-1]
        else: future_df['humidity'] = 70.0
        
        full_context = pd.concat([df_ready, future_df])
        full_feat = create_hybrid_features(full_context)
        
        # ======================================
        # Step A: LSTM 預測 (Base Trend)
        # ======================================
        current_idx = -25
        lstm_seq_cols = config['lstm_seq_cols']
        seq_data = full_feat[lstm_seq_cols].iloc[current_idx-LOOKBACK_HOURS+1 : current_idx+1].values
        X_seq = resources['scaler_seq'].transform(seq_data).reshape(1, LOOKBACK_HOURS, -1)
        
        lstm_dir_cols = config.get('lstm_direct_cols', [])
        for c in lstm_dir_cols:
            if c not in full_feat.columns: full_feat[c] = 0
        dir_data = full_feat[lstm_dir_cols].iloc[current_idx+1 : current_idx+2].values
        
        if resources['scaler_direct']: X_dir = resources['scaler_direct'].transform(dir_data)
        else: X_dir = dir_data
            
        pred_lstm_val = resources['lstm'].predict([X_seq, X_dir], verbose=0).flatten()[0]
        
        # 將 LSTM 結果填回特徵
        full_feat['lstm_pred'] = 0.0
        full_feat.iloc[-24:, full_feat.columns.get_loc('lstm_pred')] = pred_lstm_val
        
        # ======================================
        # Step B: LightGBM 預測 (Residual Correction)
        # ======================================
        lgbm_cols = config['lgbm_feature_cols']
        final_lgbm_cols = list(lgbm_cols)
        if 'lstm_pred' not in final_lgbm_cols: final_lgbm_cols.append('lstm_pred')
            
        target_feat = full_feat.iloc[-24:].copy()
        for c in final_lgbm_cols:
            if c not in target_feat.columns: target_feat[c] = 0
            
        X_lgbm = target_feat[final_lgbm_cols]
        pred_lgbm_residual = resources['lgbm'].predict(X_lgbm)
        
        # ======================================
        # 🚀 [關鍵修正] 混合邏輯：LSTM + LGBM
        # ======================================
        # 根據檔名 'residual'，LGBM 預測的是「殘差」(誤差修正量)
        # 所以最終預測 = LSTM(基礎趨勢) + LGBM(微調)
        
        # 這裡需要將 pred_lstm_val (單點) 擴展到 24 點，或者如果您的 LSTM 是多步預測則不需要
        # 假設是單點預測，我們將其視為這段時間的基準水位
        pred_base = np.full(24, pred_lstm_val)
        
        # 最終疊加
        pred_final = pred_base + pred_lgbm_residual
        pred_final = np.maximum(pred_final, 0) # 負值修正
        
        # --- 🔍 終端機診斷 (印出數值供您確認) ---
        print(f"📊 [Diagnosis] LSTM Base (Scaled): {pred_lstm_val:.4f}")
        print(f"📊 [Diagnosis] LGBM Residual (Scaled): {np.mean(pred_lgbm_residual):.4f}")
        print(f"📊 [Diagnosis] Final Combined (Scaled): {np.mean(pred_final):.4f}")
        
        # ======================================
        # 🚀 輸出放大 (Restoration)
        # ======================================
        scale_factor = DESIGN_PEAK_LOAD_KW
        
        pred_final_scaled = pred_final * scale_factor
        pred_lstm_scaled = pred_base * scale_factor
        pred_lgbm_scaled = pred_lgbm_residual * scale_factor # 這是修正量，可能是負的
        
        ui_history_df = combined_df.copy()
        if not is_scaled_input:
             ui_history_df['power_kW'] = ui_history_df['power_kW'] * scale_factor
        
        result_df = pd.DataFrame({
            "時間": future_dates,
            "預測值": pred_final_scaled,
            "LSTM (基準)": pred_lstm_scaled,
            "LGBM (修正)": pred_lgbm_scaled
        }).set_index("時間")
        
        return result_df, ui_history_df
        
    except Exception as e:
        print(f"❌ [Model Service Error]: {e}")
        import traceback
        traceback.print_exc()
        return None, None