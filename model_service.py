# model_service.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import tensorflow as tf
from datetime import timedelta

# ==========================================
# 🚑 [設定] 抑制警告與環境設定
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 預測時使用 CPU 即可
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 設定常數
# ==========================================
# 必須與 app_utils.py 中的設定一致，用來還原數值
DESIGN_PEAK_LOAD_KW = 20.0 

MODEL_FILES = {
    "config": "hybrid_residual.pkl",    # 存放 Scaler, Columns, Params
    "lgbm": "lgbm_residual.pkl",        # 殘差修正模型
    "lstm": "lstm_hybrid.keras",        # 趨勢預測模型
    "history_data": "final_training_data_with_humidity.csv" # 預設歷史資料
}

# ==========================================
# 🛠️ 特徵工程 (完全複製訓練腳本邏輯)
# ==========================================

def add_strict_features(df):
    """LightGBM 用的特徵 (對應訓練程式碼)"""
    df = df.copy()
    
    # 交互作用與統計
    df["temp_squared"] = df["temperature"] ** 2
    df["temp_humidity"] = df["temperature"] * df["humidity"]
    
    # 滾動平均 (溫度)
    df["temp_roll_24"] = df["temperature"].rolling(window=24, min_periods=1).mean()
    df["temp_roll_72"] = df["temperature"].rolling(window=72, min_periods=1).mean()
    
    # 滯後特徵 (Lags)
    df["lag_24"] = df["power"].shift(24)
    df["lag_48"] = df["power"].shift(48)
    df["lag_168"] = df["power"].shift(168)
    
    # 滾動特徵 (電力)
    # 訓練代碼中使用 .shift(24) 避免 Data Leakage
    df["rolling_mean_24h"] = df["power"].shift(24).rolling(window=24, min_periods=1).mean()
    df["rolling_max_24h"] = df["power"].shift(24).rolling(window=24, min_periods=1).max()
    df["rolling_min_24h"] = df["power"].shift(24).rolling(window=24, min_periods=1).min()
    df["rolling_mean_7d"] = df["power"].shift(24).rolling(window=168, min_periods=1).mean()
    
    df["diff_24_48"] = df["power"].shift(24) - df["power"].shift(48)
    
    return df

def add_engineering_features(df):
    """LSTM 用的特徵 (對應訓練程式碼)"""
    df = df.copy()
    
    df["temp_squared"] = df["temperature"] ** 2
    
    df["lag_24h"] = df["power"].shift(24)
    df["lag_168h"] = df["power"].shift(168)
    
    # LSTM 特徵：使用 shift(1) 代表看的是「上一小時」以前的數據
    df["rolling_mean_3h"] = df["power"].shift(1).rolling(window=3, min_periods=1).mean()
    df["rolling_mean_24h"] = df["power"].shift(1).rolling(window=24, min_periods=1).mean()
    
    return df

# ==========================================
# 🧠 核心預測邏輯
# ==========================================

def load_resources_and_predict(input_df=None):
    """
    載入模型並執行未來 24 小時的預測
    包含：資料清洗、頻率重取樣(Resampling)、數值還原、Autoregressive 預測
    """
    print("🚀 Starting Hybrid Prediction Service...")
    
    # 1. 檢查檔案是否存在
    missing_files = [f for n, f in MODEL_FILES.items() if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return None, None

    try:
        # 2. 載入模型與設定檔
        config = joblib.load(MODEL_FILES['config'])
        lgbm_model = joblib.load(MODEL_FILES['lgbm'])
        lstm_model = tf.keras.models.load_model(MODEL_FILES['lstm'])
        
        # 從 config 還原 Scaler 和 欄位名稱
        scaler_seq = config['scaler_seq']
        scaler_direct = config['scaler_direct']
        scaler_target = config['scaler_target']
        
        lstm_seq_cols = config['lstm_seq_cols']
        lstm_direct_cols = config['lstm_direct_cols']
        lgbm_feature_cols = config['lgbm_feature_cols']
        lookback_hours = config['lookback_hours'] # 通常是 168

        print("✅ Models and Config loaded successfully.")

        # 3. 準備歷史資料 (Context)
        if input_df is not None and not input_df.empty:
            history_df = input_df.copy()
        else:
            # 若無外部輸入，讀取預設 CSV
            history_df = pd.read_csv(MODEL_FILES['history_data'])
            if 'datetime' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['datetime'])
            elif 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.set_index('timestamp').sort_index()

        # -----------------------------------------------------------
        # 🚑 [資料清洗區]
        # -----------------------------------------------------------
        
        # A. 欄位名稱映射 (UI: power_kW -> Model: power)
        if 'power_kW' in history_df.columns:
            history_df = history_df.rename(columns={'power_kW': 'power'})
        
        # 確保必要欄位存在
        required_cols = ['power', 'temperature', 'humidity']
        for col in required_cols:
            if col not in history_df.columns:
                if col == 'temperature': history_df[col] = 25.0
                elif col == 'humidity': history_df[col] = 70.0
                else: raise ValueError(f"Missing column: {col}")
        
        history_df = history_df[required_cols]

        # B. 頻率重取樣 (Resampling) - 解決 15min 資料問題
        history_df = history_df.resample('H').mean().ffill()

        # C. 數值縮放檢測 (Scaling Check) - 解決 x20 倍率問題
        is_ui_scaled = False
        if history_df['power'].mean() > 2.0: 
            print("⚠️ Detected scaled input (UI scale). Reverting to model scale...")
            history_df['power'] = history_df['power'] / DESIGN_PEAK_LOAD_KW
            is_ui_scaled = True
        
        # -----------------------------------------------------------
        
        # 4. 預測迴圈準備
        buffer_size = 500
        current_df = history_df.iloc[-buffer_size:].copy()
        future_predictions = []
        last_timestamp = current_df.index[-1]
        
        print(f"⏱️ Predicting future from: {last_timestamp}")

        last_temp = current_df['temperature'].iloc[-1]
        last_hum = current_df['humidity'].iloc[-1]

        # 5. 逐小時預測未來 24 小時
        for i in range(1, 25): 
            next_time = last_timestamp + timedelta(hours=i)
            
            # --- 建立暫存 DataFrame ---
            next_row = pd.DataFrame({
                'temperature': [last_temp], 
                'humidity': [last_hum],
                'power': [np.nan] # 待預測
            }, index=[next_time])
            
            temp_df = pd.concat([current_df, next_row])
            
            # --- Step A: LSTM 預測 ---
            df_lstm_feat = add_engineering_features(temp_df)
            
            target_idx = -1
            
            # Sequence Input (過去 168 筆)
            seq_data = df_lstm_feat[lstm_seq_cols].iloc[target_idx-lookback_hours : target_idx].values
            
            # Direct Input (當下這一筆)
            # 修正處：使用 [[target_idx]] 確保取出的是 2D DataFrame (1 row, N cols)
            # 舊寫法 iloc[-1:0] 會變空值，這裡改用 [[-1]] 就能正確取出最後一列
            direct_data = df_lstm_feat[lstm_direct_cols].iloc[[target_idx]].values
            
            if len(seq_data) < lookback_hours:
                print("⚠️ Not enough history for LSTM lookback.")
                break

            # 正規化
            X_seq = scaler_seq.transform(seq_data).reshape(1, lookback_hours, -1)
            X_direct = scaler_direct.transform(direct_data)
            
            # 預測
            lstm_pred_scaled = lstm_model.predict([X_seq, X_direct], verbose=0).flatten()[0]
            lstm_pred_real = scaler_target.inverse_transform([[lstm_pred_scaled]])[0][0]
            
            # --- Step B: LightGBM 殘差修正 ---
            df_lgbm_feat = add_strict_features(temp_df)
            current_lgbm_feat = df_lgbm_feat.iloc[[target_idx]].copy()
            
            current_lgbm_feat['lstm_pred'] = lstm_pred_real
            
            X_lgbm = current_lgbm_feat[lgbm_feature_cols]
            lgbm_residual = lgbm_model.predict(X_lgbm)[0]
            
            # --- Step C: 最終融合 ---
            final_pred = lstm_pred_real + lgbm_residual
            final_pred = max(0.0, final_pred)
            
            # 將結果填回 current_df
            current_df = pd.concat([current_df, pd.DataFrame({
                'temperature': [last_temp],
                'humidity': [last_hum],
                'power': [final_pred]
            }, index=[next_time])])
            
            # 儲存結果 (若輸入被縮小過，輸出要放大回 UI 用的倍率)
            display_factor = DESIGN_PEAK_LOAD_KW if is_ui_scaled else 1.0
            
            future_predictions.append({
                "時間": next_time,
                "預測值": final_pred * display_factor,
                "LSTM基礎": lstm_pred_real * display_factor,
                "殘差修正": lgbm_residual * display_factor
            })

        # 6. 整理輸出
        result_df = pd.DataFrame(future_predictions).set_index("時間")
        
        ui_history_df = history_df.copy()
        if is_ui_scaled:
            ui_history_df['power'] = ui_history_df['power'] * DESIGN_PEAK_LOAD_KW
            
        ui_history_df = ui_history_df.rename(columns={'power': 'power_kW'})
        ui_history_df = ui_history_df.iloc[-72:][['power_kW']]
        
        print("✅ Prediction complete.")
        return result_df, ui_history_df

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
if __name__ == "__main__":
    # 測試用
    res, hist = load_resources_and_predict()
    if res is not None:
        print(res.head())