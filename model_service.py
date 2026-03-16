import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st

# 引入 app_utils 的 load_data，確保資料源頭「唯一化」
from app_utils import load_data

# ================= 設定區 =================
# 模型路徑設定
LSTM_MODEL_PATH = "lstm_hybrid_seq2seq3.h5"
RESIDUAL_MODEL_PATH = "lgbm_residual_seq2seq3.pkl"
HYBRID_MODEL_PATH = "hybrid_residual_seq2seq3.pkl"

class ModelService:
    def __init__(self):
        self.model_lstm = None
        self.model_residual = None
        self.lookback_hours = 168
        
        # 🌟 修改點：將 self.df 移除，讓 ModelService 變得純粹，只負責管理模型與預測邏輯
        self.load_models()
        
    def load_models(self):
        print("Loading models...")
        if os.path.exists(LSTM_MODEL_PATH):
            self.model_lstm = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
            print(f"LSTM model loaded from {LSTM_MODEL_PATH}")
        
        if os.path.exists(RESIDUAL_MODEL_PATH):
            self.model_residual = joblib.load(RESIDUAL_MODEL_PATH)
            print(f"Residual model loaded from {RESIDUAL_MODEL_PATH}")

        if os.path.exists(HYBRID_MODEL_PATH):
            payload = joblib.load(HYBRID_MODEL_PATH)
            self.lookback_hours = payload.get("lookback_hours", 168)
            self.features_lgbm = payload.get("lgbm_feature_cols", [])
            self.seq_cols = payload.get("lstm_seq_cols", [])
            self.direct_cols = payload.get("lstm_direct_cols", [])
            self.scaler_seq = payload.get("scaler_seq")
            self.scaler_direct = payload.get("scaler_direct")
            self.scaler_target = payload.get("scaler_target")
            print(f"Hybrid artifacts loaded from {HYBRID_MODEL_PATH}")

    def prepare_input(self, df_window):
        df = df_window.copy()
        
        # 1. 電力 Lag 與 Rolling 特徵
        # 這是 LSTM 縮放器認得的名稱 (加回來這四行！)
        df['lag_24h'] = df['power'].shift(24)
        df['lag_168h'] = df['power'].shift(168)
        df['rolling_mean_3h'] = df['power'].shift(1).rolling(window=3).mean()
        df['rolling_mean_24h'] = df['power'].shift(1).rolling(window=24).mean()

        # 這是可能給 LGBM 用到的名稱
        df['lag_24'] = df['power'].shift(24)
        df['lag_48'] = df['power'].shift(48)
        df['lag_168'] = df['power'].shift(168)
        
        df["rolling_max_24h"] = df["power"].shift(24).rolling(window=24).max()
        df["rolling_min_24h"] = df["power"].shift(24).rolling(window=24).min()
        df["rolling_mean_7d"] = df["power"].shift(24).rolling(window=168).mean()
        df["diff_24_48"] = df["power"].shift(24) - df["power"].shift(48)
        
        # 2. 時間特徵與週期性編碼
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # 3. 確保基礎天氣欄位存在
        if 'temperature' not in df.columns:
            df['temperature'] = 25.0
        if 'humidity' not in df.columns:
            df['humidity'] = 70.0

        # 4. 天氣衍生特徵
        df['temp_squared'] = df['temperature'] ** 2
        df['humidity_squared'] = df['humidity'] ** 2
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['temp_roll_24'] = df['temperature'].rolling(window=24).mean()
        df['temp_roll_72'] = df['temperature'].rolling(window=72).mean()

        # 🌟 修改點：先針對整個 DataFrame 進行空值填補，再取出最後一筆，確保 rolling 特徵不會變成 NaN
        df = df.bfill().ffill().fillna(0)
        last_row = df.iloc[[-1]].copy()
        
        return last_row

    def generate_rolling_predictions(self, hist_df, target_time=None, steps=48):
        """
        利用歷史資料與自迴歸(Auto-regressive)技術，產生未來 steps 小時的連續預測 DataFrame。
        🌟 修改點：改為從外部接收 hist_df，而不是依賴 self.df
        """
        if self.model_lstm is None or self.model_residual is None:
            print("Models not loaded properly. Returning empty prediction.")
            return pd.DataFrame()

        if hist_df.empty:
            print("No historical data available. Cannot perform predictions.")
            return pd.DataFrame()

        if target_time is None:
            target_time = hist_df.index[-1]
            
        target_time = pd.Timestamp(target_time)
        print(f"Starting rolling prediction for {steps} hours from {target_time}...")

        needed_hours = 350
        full_df = hist_df.copy()

        try:
            idx = full_df.index.get_loc(target_time)
            start_idx = max(0, idx - needed_hours)
            working_df = full_df.iloc[start_idx : idx + 1].copy()
        except KeyError:
            working_df = full_df[full_df.index <= target_time].tail(needed_hours).copy()

        predictions = []

        for step in range(steps):
            if len(working_df) < self.lookback_hours:
                print("Not enough data to continue rolling prediction.")
                break
                
            input_row = self.prepare_input(working_df)
            
            lstm_seq_raw = working_df[self.seq_cols].iloc[-self.lookback_hours:].values
            lstm_seq_scaled = self.scaler_seq.transform(lstm_seq_raw).reshape(1, self.lookback_hours, -1)
            direct_input = self.scaler_direct.transform(input_row[self.direct_cols])
            
            lstm_pred_scaled = self.model_lstm.predict([lstm_seq_scaled, direct_input], verbose=0)
            lstm_pred = self.scaler_target.inverse_transform(lstm_pred_scaled)[0][0]
            
            # 把算出來的 LSTM 預測值加進特徵表裡
            input_row['lstm_pred'] = lstm_pred
            # 1. 優先使用從 payload 載入的特徵清單
            if hasattr(self, 'features_lgbm') and len(self.features_lgbm) > 0:
                correct_features = self.features_lgbm
            else:
                # 2. 若無清單，則安全地從模型提取 (處理 MultiOutputRegressor 包裝的情況)
                target_model = self.model_residual
                if hasattr(target_model, 'estimators_'):
                    # 打開 MultiOutputRegressor 這個大箱子，拿出裡面的第一個模型
                    target_model = target_model.estimators_[0]
                
                correct_features = getattr(target_model, 'feature_name_', None)
                if correct_features is None and hasattr(target_model, 'booster_'):
                    correct_features = target_model.booster_.feature_name()
                
            try:
                lgbm_input = input_row[correct_features]
            except KeyError:
                missing_cols = [col for col in correct_features if col not in input_row.columns]
                raise ValueError(f"🚨 抓到漏網之魚！模型需要這特徵，但目前缺少了：{missing_cols}。")
                
            raw_residual = self.model_residual.predict(lgbm_input)
            residual_pred = float(np.ravel(raw_residual)[0])
            
            final_pred = lstm_pred + residual_pred
            final_pred = max(0.0, float(final_pred)) 
            
            next_time = working_df.index[-1] + pd.Timedelta(hours=1)
            
            predictions.append({
                "datetime": next_time,
                "lstm_pred": lstm_pred,
                "residual_pred": residual_pred,
                "預測值": final_pred
            })
            
            # 氣象特徵抓取 24 小時前的資料進行合理插補
            new_row = pd.DataFrame({"power": [final_pred]}, index=[next_time])
            for col in working_df.columns:
                if col != "power":
                    if col in ['temperature', 'humidity'] and len(working_df) >= 24:
                        new_row[col] = working_df.iloc[-24][col]
                    else:
                        new_row[col] = working_df.iloc[-1][col]
                    
            working_df = pd.concat([working_df, new_row])

        pred_df = pd.DataFrame(predictions)
        if not pred_df.empty:
            pred_df = pred_df.set_index("datetime")
            
        return pred_df

# ==========================================
# 🚀 資源與資料快取工廠 (徹底分離)
# ==========================================

@st.cache_resource 
def get_model_service():
    """
    只負責快取模型物件。無論使用者怎麼重整，模型都只會載入一次。
    """
    print("🧠 [Cache Miss] 正在初始化 ModelService 並載入 AI 模型...")
    return ModelService()

@st.cache_data(ttl=600) # 🌟 修改點：設定 TTL (例如 600 秒 = 10 分鐘)，時間到了自動去抓新資料
def get_latest_data():
    """
    獨立負責載入與更新資料。
    """
    print("📊 [Data Update] Loading historical data via app_utils...")
    try:
        raw_df = load_data()
        if not raw_df.empty:
            # 基礎的時間特徵可以在一開始就先建立好
            raw_df['hour'] = raw_df.index.hour
            raw_df['dayofweek'] = raw_df.index.dayofweek
            raw_df['hour_sin'] = np.sin(2 * np.pi * raw_df['hour'] / 24)
            raw_df['hour_cos'] = np.cos(2 * np.pi * raw_df['hour'] / 24)
            raw_df['day_sin'] = np.sin(2 * np.pi * raw_df['dayofweek'] / 7)
            raw_df['day_cos'] = np.cos(2 * np.pi * raw_df['dayofweek'] / 7)
            print(f"Data loaded and base features initialized. Shape: {raw_df.shape}")
            return raw_df
        else:
            print("Error: Loaded data is empty.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# ==========================================
# 🚀 給前端呼叫的統一入口函數
# ==========================================
def load_resources_and_predict(steps=48):
    """
    支援動態步數的預測入口。
    """
    print(f"🧠 [AI Core] 啟動預測任務，目標步數：{steps} 小時")
    
    # 1. 取得模型 (快取)
    service = get_model_service()
    
    # 2. 取得最新資料 (快取，但會定時更新)
    hist_df = get_latest_data()
    
    if hist_df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 3. 把資料丟進去開始預測
    pred_df = service.generate_rolling_predictions(hist_df, steps=steps)
    
    if 'power' in hist_df.columns and 'power_kW' not in hist_df.columns:
        hist_df = hist_df.rename(columns={'power': 'power_kW'})
        
    return pred_df, hist_df

if __name__ == "__main__":
    pred, hist = load_resources_and_predict()
    print("=== History Head ===")
    print(hist.tail())
    print("\n=== Prediction Head ===")
    print(pred.head())
    
    # 🌟 修改點：測試區塊也改用 get_model_service()，避免又消耗記憶體重新載入模型
    service = get_model_service() 
    if service.model_lstm is not None:
        print("=== Scaler 的真實範圍 ===")
        print("Target Max:", service.scaler_target.data_max_)
        print("Target Min:", service.scaler_target.data_min_)