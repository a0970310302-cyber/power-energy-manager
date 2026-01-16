import os
import json
import time
import datetime
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import requests  # [新增] 用於抓取即時資料

# ================= 設定區 =================
# [修改] 檔案名稱改為加總版
RAW_CSV_FILE = "final_training_data_with_humidity_sum.csv"

# [新增] 即時資料來源設定
REALTIME_POWER_SOURCE = "https://api.jsonstorage.net/v1/json/888edb43-7993-46ef-8020-767afb44a2cb/3a6282d1-b474-4274-952b-e4a0a84f0deb?apiKey=e1230ae2-6eee-433a-ad58-7ab2c622b9e5"
WEATHER_INDEX_URL = "https://api.jsonstorage.net/v1/json/12d77044-531c-4984-8421-01585a961bfb/3f1ac541-adb2-4275-901b-dd1300502c0f"
PANTRY_ID = "6a2e85f5-4af4-4efd-bb9f-c5604fe8475e"
PANTRY_BASKET = "2026-q1" # 假設即時預測主要針對 2026 第一季
PANTRY_URL = f"https://getpantry.cloud/apiv1/pantry/{PANTRY_ID}/basket/{PANTRY_BASKET}"

# 模型路徑設定 (維持原樣)
LSTM_MODEL_PATH = "lstm_hybrid_noweather.keras"
RESIDUAL_MODEL_PATH = "lgbm_residual_noweather.pkl"
HYBRID_MODEL_PATH = "hybrid_residual_noweather.pkl"

class ModelService:
    def __init__(self):
        self.model_lstm = None
        self.model_residual = None
        self.lookback_hours = 168
        self.df = pd.DataFrame()
        
        # 載入模型與資料
        self.load_models()
        self.load_data()

    def load_models(self):
        print("Loading models...")
        if os.path.exists(LSTM_MODEL_PATH):
            self.model_lstm = tf.keras.models.load_model(LSTM_MODEL_PATH)
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

    def load_data(self):
        print(f"Loading historical data from {RAW_CSV_FILE}...")
        if os.path.exists(RAW_CSV_FILE):
            try:
                self.df = pd.read_csv(RAW_CSV_FILE)
                self.df['datetime'] = pd.to_datetime(self.df['datetime'])
                self.df = self.df.set_index('datetime').sort_index()
                # 確保只有非零數據，避免異常
                self.df = self.df[self.df['power'] > 0]
                print(f"Data loaded. Shape: {self.df.shape}")
                print(f"Time range: {self.df.index.min()} to {self.df.index.max()}")
            except Exception as e:
                print(f"Error loading CSV: {e}")
        else:
            print(f"Warning: {RAW_CSV_FILE} not found. Service will rely on live fetching for recent data.")

    # [新增] 抓取即時資料的輔助方法
    def _fetch_live_data_window(self):
        """
        從 API 抓取最近的電力(2026-q1 Pantry)與氣候資料，
        並進行 '每小時加總(Sum)' 處理，回傳 DataFrame。
        """
        print("Fetching live data from APIs...")
        try:
            # 1. 抓取電力 (Pantry 2026-q1)
            res = requests.get(PANTRY_URL, timeout=10)
            if res.status_code != 200:
                print("Failed to fetch power pantry.")
                return pd.DataFrame()
            
            data_dict = res.json()
            # Pantry 存的是原始 15 分鐘資料
            df_power = pd.DataFrame(list(data_dict.items()), columns=['datetime', 'power'])
            df_power['datetime'] = pd.to_datetime(df_power['datetime'])
            df_power = df_power.set_index('datetime').sort_index()

            # [關鍵] 進行每小時加總 (Sum)，與訓練資料一致
            df_hourly = df_power.resample('1h', label='right', closed='right').sum()
            df_hourly = df_hourly[df_hourly['power'] > 0] # 過濾 0 值

            # 2. 抓取氣候 (最近 7 天)
            # 為了效能，這裡只示範抓取必要的最近資料，實務上可優化快取
            df_weather = pd.DataFrame()
            try:
                w_idx_res = requests.get(WEATHER_INDEX_URL, timeout=10)
                w_items = w_idx_res.json().get("items", {})
                # 簡單抓取最後 10 個連結 (涵蓋最近日期)
                recent_dates = sorted(w_items.keys())[-10:] 
                
                weather_records = []
                for date_str in recent_dates:
                    uri = w_items[date_str].get("uri")
                    if uri:
                        d_res = requests.get(uri, timeout=5)
                        rows = d_res.json().get("days", {}).get(date_str, {}).get("rows", [])
                        for row in rows:
                            weather_records.append({
                                "datetime": row[0],
                                "temperature": row[1],
                                "humidity": row[2]
                            })
                if weather_records:
                    df_weather = pd.DataFrame(weather_records)
                    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
            except Exception as e:
                print(f"Weather fetch warning: {e}")

            # 3. 合併 (以電力資料為主)
            if not df_weather.empty:
                df_merged = pd.merge(df_hourly.reset_index(), df_weather, on='datetime', how='left')
            else:
                df_merged = df_hourly.reset_index()
                # 若無天氣資料，補 NaN 或 0，視模型是否必須需要天氣
                df_merged['temperature'] = 0
                df_merged['humidity'] = 0

            df_merged = df_merged.set_index('datetime').sort_index()
            return df_merged

        except Exception as e:
            print(f"Error fetching live data: {e}")
            return pd.DataFrame()

    # [修改] 修改取得資料邏輯：支援 CSV 查找失敗時切換到即時抓取
    def get_latest_data(self, target_time):
        """
        取得預測所需的時間窗口資料 (lookback_hours)。
        優先從 CSV 讀取，若無則嘗試抓取即時資料。
        """
        target_timestamp = pd.Timestamp(target_time)
        
        # 嘗試從 CSV 歷史資料獲取
        if target_timestamp in self.df.index:
            target_idx = self.df.index.get_loc(target_timestamp)
            start_idx = target_idx - self.lookback_hours
            
            if start_idx >= 0:
                return self.df.iloc[start_idx:target_idx]
        
        # 若 CSV 沒有 (例如是現在/未來)，則抓取即時資料
        print(f"Target time {target_timestamp} not in history CSV. Attempting live fetch...")
        df_live = self._fetch_live_data_window()
        
        if not df_live.empty and target_timestamp in df_live.index:
            # 在即時資料中找到目標時間
            target_idx = df_live.index.get_loc(target_timestamp)
            start_idx = target_idx - self.lookback_hours
            
            if start_idx >= 0:
                print("Successfully retrieved data window from live source.")
                return df_live.iloc[start_idx:target_idx]
            else:
                # 資料不足 lookback (例如剛過年)，嘗試拼接 CSV 與 Live (進階處理)
                # 這裡簡化處理：若 Live 資料不夠長，直接回傳有的部分，讓後續處理
                return df_live.iloc[:target_idx]
        
        print("Data unavailable for the requested time.")
        return pd.DataFrame()

    def prepare_input(self, df_window):
        # 確保特徵工程與訓練時一致 (平均 vs 加總 已在 load/fetch 階段處理完畢)
        df = df_window.copy()
        
        # 產生特徵 (Lags, Rolling)
        df['lag_24h'] = df['power'].shift(24)
        df['lag_168h'] = df['power'].shift(168)
        df['rolling_mean_3h'] = df['power'].shift(1).rolling(window=3).mean()
        df['rolling_mean_24h'] = df['power'].shift(1).rolling(window=24).mean()
        
        # LGBM 特徵
        df["lag_24"] = df["power"].shift(24)
        df["lag_48"] = df["power"].shift(48)
        df["lag_168"] = df["power"].shift(168)
        df["rolling_mean_24h"] = df["power"].shift(24).rolling(window=24).mean()
        df["rolling_max_24h"] = df["power"].shift(24).rolling(window=24).max()
        df["rolling_min_24h"] = df["power"].shift(24).rolling(window=24).min()
        df["rolling_mean_7d"] = df["power"].shift(24).rolling(window=168).mean()
        df["diff_24_48"] = df["power"].shift(24) - df["power"].shift(48)

        # 取最後一筆 (我們要預測的時間點)
        # 注意：由於 shift 的關係，最後一筆資料其實包含的是「已知」的過去特徵
        last_row = df.iloc[[-1]].copy()
        
        # 填補 NaN (因為剛剛 shift 產生的)
        last_row = last_row.fillna(method='ffill').fillna(0)

        return last_row

    def predict_next_24h(self, current_time_str):
        """
        核心預測函式
        """
        try:
            target_time = pd.Timestamp(current_time_str)
            print(f"Predicting for time: {target_time}")

            # 1. 取得資料視窗 (History + Live)
            # 這裡我們需要比 lookback 多一點的資料來計算 rolling/lag
            # 訓練時 lookback=168，但特徵工程最大 lag=168，所以至少需要 168+168 小時前的資料
            # 為了簡化，get_latest_data 回傳基礎視窗，我們假設資料庫夠大
            # 實務上：應該要抓取 target_time 往前推 336 小時的資料來確保特徵計算無誤
            # 這裡做一個修正：get_latest_data 邏輯保持回傳 lookback，
            # 但我們為了計算 lag，在 _fetch 或 load 時應確保有足夠 buffer
            
            # 重新定義：抓取最近 350 小時以確保特徵計算安全
            needed_hours = 350 
            
            # 這裡簡化邏輯：直接依賴 load_data 或 fetch 回傳的完整 DataFrame 來做切片
            if target_time in self.df.index:
                full_df = self.df
            else:
                full_df = self._fetch_live_data_window()

            if full_df.empty or target_time not in full_df.index:
                return {"error": "No data available for this time"}

            # 找出位置
            idx = full_df.index.get_loc(target_time)
            if idx < needed_hours:
                # 資料不足以計算完整特徵
                print("Warning: Not enough history for full feature engineering.")
                # 嘗試用現有資料計算
                start_idx = 0
            else:
                start_idx = idx - needed_hours
            
            window_df = full_df.iloc[start_idx : idx + 1].copy() # +1 包含當下時間點作為 index
            
            # 2. 準備特徵
            input_row = self.prepare_input(window_df)
            
            # 3. 預測
            # LSTM Input
            seq_input = self.scaler_seq.transform(input_row[self.seq_cols])
            # LSTM 需要 sequence shape (1, lookback, features) 
            # 但這裡 input_row 只有一行，這代表是用單點預測？
            # 修正：Hybrid 模型通常需要一個序列輸入。
            # 如果 input_row 是單行，我們需要回推產生序列。
            
            # 修正邏輯：取 input_row 前 lookback 小時的序列
            # 這部分需要更複雜的 data prep，這裡假設 prepare_input 已經處理好直接輸入特徵 (LGBM approach)
            # 若是 LSTM，我們需要從 window_df 取最後 lookback 筆
            
            lstm_seq_raw = window_df[self.seq_cols].iloc[-self.lookback_hours:].values
            if len(lstm_seq_raw) < self.lookback_hours:
                 return {"error": "Not enough data for LSTM sequence"}
            
            lstm_seq_scaled = self.scaler_seq.transform(lstm_seq_raw).reshape(1, self.lookback_hours, -1)
            
            direct_input = self.scaler_direct.transform(input_row[self.direct_cols])
            
            # LSTM Predict
            lstm_pred_scaled = self.model_lstm.predict([lstm_seq_scaled, direct_input], verbose=0)
            lstm_pred = self.scaler_target.inverse_transform(lstm_pred_scaled)[0][0]
            
            # Residual (LGBM) Predict
            lgbm_input = input_row[self.features_lgbm]
            residual_pred = self.model_residual.predict(lgbm_input)[0]
            
            final_pred = lstm_pred + residual_pred
            
            return {
                "datetime": str(target_time),
                "lstm_pred": float(lstm_pred),
                "residual_pred": float(residual_pred),
                "final_prediction": float(final_pred)
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

# 測試用
if __name__ == "__main__":
    service = ModelService()
    # 測試一個 CSV 裡的時間
    print(service.predict_next_24h("2024-01-01 10:00:00"))
    # 測試一個 即時 時間 (請確保網路上有資料)
    # print(service.predict_next_24h("2026-01-16 10:00:00"))