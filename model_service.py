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

# [更新] 使用新的檔案路徑
MODEL_FILES = {
    "config": "hybrid_residual.pkl",
    "lgbm": "lgbm_residual.pkl",
    "lstm": "lstm_hybrid.keras",
    "history_data": "final_training_data_with_humidity.csv"
}

LOOKBACK_HOURS = 168

# ==========================================
# 🛠️ 進階特徵工程 (適配 Residual 模型)
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

def create_hybrid_features(df):
    """
    產生 Hybrid Model 所需的所有特徵
    """
    df = df.copy()
    
    # 時間特徵
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    
    # 週期性編碼
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    # 假日
    tw_holidays = get_taiwan_holidays()
    date_strs = df.index.strftime("%Y-%m-%d")
    df["is_weekend"] = ((df["day_of_week"] >= 5) | (date_strs.isin(tw_holidays))).astype(int)
    
    # 互動特徵 (溫度相關)
    df["temp_squared"] = df["temperature"] ** 2
    df["temp_humidity"] = df["temperature"] * df["humidity"]
    
    # 滾動特徵 (Rolling)
    for w in [24, 72]:
        df[f'temp_roll_{w}'] = df['temperature'].rolling(window=w, min_periods=1).mean()
        
    df['rolling_mean_24h'] = df['power'].shift(1).rolling(window=24, min_periods=1).mean()
    df['rolling_max_24h'] = df['power'].shift(1).rolling(window=24, min_periods=1).max()
    df['rolling_min_24h'] = df['power'].shift(1).rolling(window=24, min_periods=1).min()
    df['rolling_mean_7d'] = df['power'].shift(1).rolling(window=168, min_periods=1).mean()
    df['rolling_mean_3h'] = df['power'].shift(1).rolling(window=3, min_periods=1).mean() # LSTM 用
    
    # Lag 特徵
    for lag in [24, 48, 168]:
        df[f'lag_{lag}'] = df['power'].shift(lag)
        df[f'lag_{lag}h'] = df['power'].shift(lag) # 兼容命名
        
    df['diff_24_48'] = df['lag_24'] - df['lag_48']
    
    return df

# ==========================================
# 🧠 主預測流程
# ==========================================
def load_resources_and_predict(full_data_df=None):
    resources = {}
    try:
        # 1. 載入模型與設定
        print("📥 [Model Service] 正在載入混合模型資源...")
        config = joblib.load(MODEL_FILES['config']) # hybrid_residual.pkl
        resources['lgbm'] = joblib.load(MODEL_FILES['lgbm'])
        resources['lstm'] = keras.models.load_model(MODEL_FILES['lstm'])
        resources['scaler_seq'] = config['scaler_seq']
        # resources['scaler_direct'] = config.get('scaler_direct', None) # 視情況使用
        
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

        # 還原為小數值 (Normalized Scale)
        df_model = combined_df.copy()
        if is_scaled_input:
            df_model['power'] = df_model['power_kW'] / DESIGN_PEAK_LOAD_KW
        else:
            df_model['power'] = pd.to_numeric(df_model['power_kW'], errors='coerce')
        
        df_model = df_model.dropna(subset=['power'])
        
        # 3. 準備預測區間 (Buffer)
        buffer_size = 2000
        df_ready = df_model.iloc[-buffer_size:].copy()
        last_time = df_ready.index[-1]
        
        future_dates = [last_time + timedelta(hours=i+1) for i in range(24)]
        future_df = pd.DataFrame(index=future_dates, columns=df_ready.columns)
        
        # 簡單填補未來環境特徵
        if 'temperature' in df_ready.columns: future_df['temperature'] = df_ready['temperature'].iloc[-1]
        else: future_df['temperature'] = 25.0
        if 'humidity' in df_ready.columns: future_df['humidity'] = df_ready['humidity'].iloc[-1]
        else: future_df['humidity'] = 70.0
        
        full_context = pd.concat([df_ready, future_df])
        
        # 4. 產生基礎特徵
        full_feat = create_hybrid_features(full_context)
        
        # ======================================
        # Step A: LSTM 預測 (第一層)
        # ======================================
        # LSTM 輸入準備 (Sequence)
        # 根據 config 中的 lstm_seq_cols: ['power', 'temperature', 'humidity']
        lstm_cols = config['lstm_seq_cols']
        
        # 取得最後一段歷史作為輸入
        # LSTM input shape: (1, 168, 3)
        current_idx = -25 # 未來的第一個點的前一個位置
        seq_data = full_feat[lstm_cols].iloc[current_idx-LOOKBACK_HOURS+1 : current_idx+1].values
        
        # Scaling
        X_seq = resources['scaler_seq'].transform(seq_data).reshape(1, LOOKBACK_HOURS, -1)
        
        # Predict
        # 這裡假設 LSTM 輸出的是單點預測，我們簡單用來當作未來趨勢的基準
        # 為了產生 24 小時的預測特徵，我們這裡做一個簡化：
        # 用 LSTM 預測出來的值，填滿未來的 lstm_pred 欄位
        pred_lstm_val = resources['lstm'].predict(X_seq, verbose=0).flatten()[0]
        
        # 將 LSTM 預測值放入特徵中 (給 LGBM 用)
        # 這裡假設未來 24 小時的 LSTM 預測值是一個平滑的趨勢或定值
        # 若你的 LSTM 是輸出 24 小時序列，則直接填入；若是單點，則廣播
        full_feat['lstm_pred'] = 0.0
        full_feat.iloc[-24:, full_feat.columns.get_loc('lstm_pred')] = pred_lstm_val
        
        # ======================================
        # Step B: LightGBM 預測 (第二層 / 殘差修正)
        # ======================================
        lgbm_cols = config['lgbm_feature_cols']
        target_feat = full_feat.iloc[-24:].copy()
        
        # 補齊缺失欄位 (防呆)
        for c in lgbm_cols:
            if c not in target_feat.columns: target_feat[c] = 0
            
        X_lgbm = target_feat[lgbm_cols]
        pred_final = resources['lgbm'].predict(X_lgbm)
        pred_final = np.maximum(pred_final, 0)
        
        # ======================================
        # 🚀 輸出放大
        # ======================================
        scale_factor = DESIGN_PEAK_LOAD_KW
        pred_final_scaled = pred_final * scale_factor
        
        # 為了畫圖，我們也把 LSTM 的中間產物輸出出來看
        pred_lstm_scaled = np.full(24, pred_lstm_val * scale_factor)
        
        ui_history_df = combined_df.copy()
        if not is_scaled_input:
             ui_history_df['power_kW'] = ui_history_df['power_kW'] * scale_factor
        
        result_df = pd.DataFrame({
            "時間": future_dates,
            "預測值": pred_final_scaled,
            "LSTM (特徵)": pred_lstm_scaled, # 這是 LSTM 給 LGBM 的參考值
            "LGBM (最終)": pred_final_scaled # 在此架構下，LGBM 輸出即為最終結果
        }).set_index("時間")
        
        return result_df, ui_history_df
        
    except Exception as e:
        print(f"❌ [Model Service Error]: {e}")
        import traceback
        traceback.print_exc()
        return None, None