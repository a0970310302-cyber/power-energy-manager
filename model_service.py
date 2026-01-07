# model_service.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import tensorflow as tf
from datetime import timedelta, datetime
import calendar

# ==========================================
# 🚑 [設定] 抑制警告與環境設定
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 設定常數
# ==========================================
# [修正] 同步設定為 3.6
DESIGN_PEAK_LOAD_KW = 3.6 

MODEL_FILES = {
    "config": "hybrid_residual.pkl",    
    "lgbm": "lgbm_residual.pkl",        
    "lstm": "lstm_hybrid.keras",        
    "history_data": "final_training_data_with_humidity.csv"
}

# ==========================================
# 🌤️ 天氣模擬器 (解決未來天氣未知的問題)
# ==========================================
class WeatherSimulator:
    def __init__(self, history_df):
        # 建立一個快速查詢表：(Month, Day, Hour) -> (Avg Temp, Avg Hum)
        self.lookup = {}
        temp_df = history_df.copy()
        temp_df['month'] = temp_df.index.month
        temp_df['day'] = temp_df.index.day
        temp_df['hour'] = temp_df.index.hour
        
        # 計算歷史平均值作為未來的期望值
        stats = temp_df.groupby(['month', 'day', 'hour'])[['temperature', 'humidity']].mean()
        self.lookup = stats.to_dict('index')
        
        # 備用：全域平均
        self.fallback_temp = temp_df['temperature'].mean()
        self.fallback_hum = temp_df['humidity'].mean()

    def get_forecast(self, target_time):
        key = (target_time.month, target_time.day, target_time.hour)
        if key in self.lookup:
            data = self.lookup[key]
            return data['temperature'], data['humidity']
        else:
            return self.fallback_temp, self.fallback_hum

# ==========================================
# 🛠️ 特徵工程
# ==========================================
def add_strict_features(df):
    df = df.copy()
    df["temp_squared"] = df["temperature"] ** 2
    df["temp_humidity"] = df["temperature"] * df["humidity"]
    df["temp_roll_24"] = df["temperature"].rolling(window=24, min_periods=1).mean()
    df["temp_roll_72"] = df["temperature"].rolling(window=72, min_periods=1).mean()
    df["lag_24"] = df["power"].shift(24)
    df["lag_48"] = df["power"].shift(48)
    df["lag_168"] = df["power"].shift(168)
    df["rolling_mean_24h"] = df["power"].shift(24).rolling(window=24, min_periods=1).mean()
    df["rolling_max_24h"] = df["power"].shift(24).rolling(window=24, min_periods=1).max()
    df["rolling_min_24h"] = df["power"].shift(24).rolling(window=24, min_periods=1).min()
    df["rolling_mean_7d"] = df["power"].shift(24).rolling(window=168, min_periods=1).mean()
    df["diff_24_48"] = df["power"].shift(24) - df["power"].shift(48)
    return df

def add_engineering_features(df):
    df = df.copy()
    df["temp_squared"] = df["temperature"] ** 2
    df["lag_24h"] = df["power"].shift(24)
    df["lag_168h"] = df["power"].shift(168)
    df["rolling_mean_3h"] = df["power"].shift(1).rolling(window=3, min_periods=1).mean()
    df["rolling_mean_24h"] = df["power"].shift(1).rolling(window=24, min_periods=1).mean()
    return df

# ==========================================
# 🧠 核心預測邏輯 (長程馬拉松版)
# ==========================================
def load_resources_and_predict(input_df=None):
    print("🚀 Starting Hybrid Prediction Service (Full Cycle Mode)...")
    
    missing_files = [f for n, f in MODEL_FILES.items() if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return None, None

    try:
        # 2. 載入模型
        config = joblib.load(MODEL_FILES['config'])
        lgbm_model = joblib.load(MODEL_FILES['lgbm'])
        lstm_model = tf.keras.models.load_model(MODEL_FILES['lstm'])
        
        scaler_seq = config['scaler_seq']
        scaler_direct = config['scaler_direct']
        scaler_target = config['scaler_target']
        
        lstm_seq_cols = config['lstm_seq_cols']
        lstm_direct_cols = config['lstm_direct_cols']
        lgbm_feature_cols = config['lgbm_feature_cols'] 
        lookback_hours = config['lookback_hours']

        print("✅ Models and Config loaded successfully.")

        # 3. 準備歷史資料
        if input_df is not None and not input_df.empty:
            history_df = input_df.copy()
        else:
            history_df = pd.read_csv(MODEL_FILES['history_data'])
            if 'datetime' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['datetime'])
            elif 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.set_index('timestamp').sort_index()

        # -----------------------------------------------------------
        # 🚑 [資料清洗區]
        # -----------------------------------------------------------
        if 'power_kW' in history_df.columns:
            history_df = history_df.rename(columns={'power_kW': 'power'})
        
        required_cols = ['power', 'temperature', 'humidity']
        for col in required_cols:
            if col not in history_df.columns:
                if col == 'temperature': history_df[col] = 25.0
                elif col == 'humidity': history_df[col] = 70.0
                else: raise ValueError(f"Missing column: {col}")
        
        history_df = history_df[required_cols]
        history_df = history_df.resample('H').mean().ffill()

        # 縮放檢測 (0.2 門檻)
        is_ui_scaled = False
        if history_df['power'].mean() > 0.2: 
            print("⚠️ Detected scaled input (UI scale). Reverting to model scale...")
            history_df['power'] = history_df['power'] / DESIGN_PEAK_LOAD_KW
            is_ui_scaled = True
        
        # 初始化天氣模擬器
        weather_sim = WeatherSimulator(history_df)
        
        # -----------------------------------------------------------
        # 🏃 [預測規劃] 計算還需要跑多遠 (到本期帳單結束)
        # -----------------------------------------------------------
        last_timestamp = history_df.index[-1]
        
        # 簡單計算帳單週期結束日 (複製 app_utils 邏輯以免循環引用)
        curr_mon = last_timestamp.month
        start_mon = curr_mon if curr_mon % 2 != 0 else curr_mon - 1
        end_mon = start_mon + 1
        last_day = calendar.monthrange(last_timestamp.year, end_mon)[1]
        cycle_end_date = datetime(last_timestamp.year, end_mon, last_day, 23, 0, 0)
        
        # 計算剩餘小時數
        hours_to_predict = int((cycle_end_date - last_timestamp).total_seconds() / 3600)
        if hours_to_predict <= 0:
            # 如果已經是最後一天，預測未來 24 小時即可
            hours_to_predict = 24
            
        print(f"⏱️ Predicting from {last_timestamp} to {cycle_end_date} ({hours_to_predict} hours)")

        # 4. 預測迴圈
        buffer_size = 500
        current_df = history_df.iloc[-buffer_size:].copy()
        future_predictions = []
        
        # 為了效能，每 24 小時印一次進度
        for i in range(1, hours_to_predict + 1): 
            next_time = last_timestamp + timedelta(hours=i)
            
            # --- 模擬未來天氣 ---
            sim_temp, sim_hum = weather_sim.get_forecast(next_time)
            
            next_row = pd.DataFrame({
                'temperature': [sim_temp], 
                'humidity': [sim_hum],
                'power': [np.nan] 
            }, index=[next_time])
            
            temp_df = pd.concat([current_df, next_row])
            
            # --- Step A: LSTM ---
            df_lstm_feat = add_engineering_features(temp_df)
            target_idx = -1
            
            seq_data = df_lstm_feat[lstm_seq_cols].iloc[target_idx-lookback_hours : target_idx].values
            direct_data = df_lstm_feat[lstm_direct_cols].iloc[[target_idx]].values
            
            if len(seq_data) < lookback_hours:
                break

            X_seq = scaler_seq.transform(seq_data).reshape(1, lookback_hours, -1)
            X_direct = scaler_direct.transform(direct_data)
            
            lstm_pred_scaled = lstm_model.predict([X_seq, X_direct], verbose=0).flatten()[0]
            lstm_pred_real = scaler_target.inverse_transform([[lstm_pred_scaled]])[0][0]
            
            # --- Step B: LightGBM ---
            df_lgbm_feat = add_strict_features(temp_df)
            current_lgbm_feat = df_lgbm_feat.iloc[[target_idx]].copy()
            
            current_lgbm_feat['lstm_pred'] = lstm_pred_real
            
            final_feature_cols = list(lgbm_feature_cols)
            if 'lstm_pred' not in final_feature_cols:
                final_feature_cols.append('lstm_pred')
            
            X_lgbm = current_lgbm_feat[final_feature_cols]
            lgbm_residual = lgbm_model.predict(X_lgbm)[0]
            
            # --- Step C: 融合 ---
            final_pred = lstm_pred_real + lgbm_residual
            final_pred = max(0.0, final_pred)
            
            current_df = pd.concat([current_df, pd.DataFrame({
                'temperature': [sim_temp],
                'humidity': [sim_hum],
                'power': [final_pred]
            }, index=[next_time])])
            
            display_factor = DESIGN_PEAK_LOAD_KW
            
            future_predictions.append({
                "時間": next_time,
                "預測值": final_pred * display_factor,
                "LSTM基礎": lstm_pred_real * display_factor,
                "殘差修正": lgbm_residual * display_factor
            })
            
            if i % 48 == 0:
                print(f"   ... Progress: {i}/{hours_to_predict} hours predicted")

        # 6. 整理輸出
        result_df = pd.DataFrame(future_predictions).set_index("時間")
        
        ui_history_df = history_df.copy()
        ui_history_df['power'] = ui_history_df['power'] * DESIGN_PEAK_LOAD_KW
        ui_history_df = ui_history_df.rename(columns={'power': 'power_kW'})
        ui_history_df = ui_history_df[['power_kW']]
        
        # 回傳合併後的完整數據 (歷史 + 預測)，方便前端切割
        # 但為了相容性，我們還是分開回傳
        print(f"✅ Prediction complete. Generated {len(result_df)} future points.")
        return result_df, ui_history_df

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    res, hist = load_resources_and_predict()
    if res is not None:
        print(res.head())