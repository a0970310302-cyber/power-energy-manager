# model_service.py
import pandas as pd
import numpy as np
import joblib
import requests
import os
import re
import warnings
import json

# ==========================================
# 🚑 [設定] 抑制警告 & 相容性設定
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # 忽略日期解析警告

from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras

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

LIVE_DATA_URL = "https://getpantry.cloud/apiv1/pantry/6e282296-e38a-454b-9895-a86d12a82731/basket/new"
HISTORY_PANTRY_ID = "6a2e85f5-4af4-4efd-bb9f-c5604fe8475e" 
LOOKBACK_HOURS = 168

# ==========================================
# 🔧 [新增] 輔助工具：自動計算籃子名稱
# ==========================================
def get_basket_name(dt: datetime):
    """
    根據日期自動計算 Basket 名稱，例如: 2026-01-06 -> '2026-q1'
    """
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-q{quarter}"

# ==========================================
# 🛠️ 特徵工程 (保持不變)
# ==========================================
def get_taiwan_holidays():
    return ["2024-01-01", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", 
            "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-28", "2024-04-04", 
            "2024-04-05", "2024-05-01", "2024-06-10", "2024-09-17", "2024-10-10",
            # 建議補上 2025, 2026 的假期，或是讓它自動化，目前先維持原樣
            "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31"]

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
# 📥 資料獲取邏輯
# ==========================================
def find_data_list(data_dict):
    target_key = "listAMIBase15MinData"
    if target_key in data_dict:
        return data_dict[target_key], None
    for key, value in data_dict.items():
        date_match = re.match(r"^\d{4}-\d{2}-\d{2}$", str(key))
        current_date = key if date_match else None
        if isinstance(value, dict):
            found, sub_date = find_data_list(value)
            if found: return found, (sub_date if sub_date else current_date)
        if isinstance(value, list) and len(value) > 0 and current_date:
            if isinstance(value[0], dict) and ("power" in value[0] or "power_kW" in value[0]):
                return value, current_date
    return None, None

def process_raw_data_to_df(target_list, date_context):
    if not target_list:
        return pd.DataFrame()

    df = pd.DataFrame(target_list)
    
    if 'power' in df.columns:
        df = df.rename(columns={'power': 'power_kW'})
    
    try:
        if 'full_timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['full_timestamp'], errors='coerce')
        elif 'date' in df.columns and 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'].astype(str) + " " + df['time'].astype(str), errors='coerce')
        elif 'time' in df.columns:
            if date_context:
                df['timestamp'] = pd.to_datetime(f"{date_context} " + df['time'], errors='coerce')
            else:
                df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
        else:
            return pd.DataFrame() 
            
    except Exception as e:
        print(f"⚠️ 時間解析失敗: {e}")
        return pd.DataFrame()

    if 'timestamp' not in df.columns or 'power_kW' not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    df['power_kW'] = pd.to_numeric(df['power_kW'], errors='coerce')
    
    if 'isMissingData' in df.columns:
        df.loc[df['isMissingData'] == 1, 'power_kW'] = np.nan
        df.loc[df['isMissingData'] == '1', 'power_kW'] = np.nan
    
    df['power_kW'] = df['power_kW'].replace(0, np.nan)
    df['power_kW'] = df['power_kW'].replace(0.0, np.nan)
    df['power_kW'] = df['power_kW'].ffill().bfill()
    
    if 'temperature' not in df.columns:
        df['temperature'] = 25.0
        df['humidity'] = 70.0
        
    return df[['power_kW', 'temperature', 'humidity']]

def fetch_live_data():
    try:
        response = requests.get(LIVE_DATA_URL, timeout=5)
        data_json = response.json()
        
        if data_json.get('status') != 1:
            print(f"⚠️ [Live] API Status: {data_json.get('status')} (暫無即時資料)")
            return None
            
        raw_data = data_json['data']
        if isinstance(raw_data, list) and len(raw_data) > 0:
            first_item = raw_data[0]
            if isinstance(first_item, dict) and ("power" in first_item or "power_kW" in first_item):
                print(f"✅ [Live] 取得散裝資料 {len(raw_data)} 筆")
                return process_raw_data_to_df(raw_data, None)

        target_list = []
        date_context = None
        if isinstance(raw_data, list) and len(raw_data) > 0:
            target_list, date_context = find_data_list(raw_data[0])
        elif isinstance(raw_data, dict):
            target_list, date_context = find_data_list(raw_data)
        
        if target_list:
            print(f"✅ [Live] 解包成功 (Date: {date_context})")
            return process_raw_data_to_df(target_list, date_context)
            
        return None
    except:
        return None

def fetch_recent_history_gap():
    """
    [修正] 自動判斷要抓取的歷史籃子 (跨季度支援)
    """
    now = datetime.now()
    
    # 1. 取得本季籃子 (例如 2026-q1)
    current_basket = get_basket_name(now)
    
    # 2. 取得上一季籃子 (例如 2025-q4)
    # 邏輯：取得本月1號的前一天，就是上個月底，用那一天來算上一季
    last_month_date = now.replace(day=1) - timedelta(days=1)
    prev_basket = get_basket_name(last_month_date)
    
    # 3. 建立目標清單 (去重排序)
    target_baskets = sorted(list({prev_basket, current_basket}))
    
    all_gap_dfs = []
    
    print(f"⏳ [Gap] 正在補齊歷史資料，目標籃子: {target_baskets} ...")
    
    for basket in target_baskets:
        url = f"https://getpantry.cloud/apiv1/pantry/{HISTORY_PANTRY_ID}/basket/{basket}"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if "data" in data and isinstance(data["data"], list):
                    raw_items = data["data"]
                    print(f"   📦 [Gap] {basket} 下載成功: {len(raw_items)} items")
                    
                    if len(raw_items) > 0 and isinstance(raw_items[0], dict) and ("power" in raw_items[0] or "power_kW" in raw_items[0]):
                         # print("   🔍 [Gap] 偵測到散裝格式，使用內建日期欄位解析...")
                         df = process_raw_data_to_df(raw_items, None)
                         if not df.empty:
                            all_gap_dfs.append(df)
                    else:
                        for item in raw_items:
                            target_list, date_context = find_data_list(item)
                            if target_list:
                                sub_df = process_raw_data_to_df(target_list, date_context)
                                if not sub_df.empty:
                                    all_gap_dfs.append(sub_df)
        except Exception as e:
            print(f"   ⚠️ [Gap Error] {basket}: {e}")
    
    if not all_gap_dfs:
        print("   ⚠️ [Gap] 未能補入任何有效資料")
        return pd.DataFrame()
        
    try:
        full_gap_df = pd.concat(all_gap_dfs)
        full_gap_df = full_gap_df.sort_index()
        full_gap_df = full_gap_df[~full_gap_df.index.duplicated(keep='first')]
        print(f"   ✅ [Gap] 補洞完成！共 {len(full_gap_df)} 筆 (範圍: {full_gap_df.index.min()} ~ {full_gap_df.index.max()})")
        return full_gap_df
    except:
        return pd.DataFrame()

# ==========================================
# 💾 [自動歸檔] 核心功能 (包含滑動視窗)
# ==========================================
def auto_archive_live_data(live_df):
    """
    自動將最新的 Live Data 歸檔到對應的歷史 Pantry Basket
    包含：自動判斷季度、滑動視窗(保留約2000筆)、防重複檢查
    """
    if live_df is None or live_df.empty:
        return

    try:
        # 1. 取出 Live Data 最新的一筆資料
        latest_record = live_df.iloc[-1].copy()
        
        # 🔥 [修正] 動態決定目標籃子
        dynamic_basket = get_basket_name(latest_record.name)
        history_url = f"https://getpantry.cloud/apiv1/pantry/{HISTORY_PANTRY_ID}/basket/{dynamic_basket}"
        
        new_data_payload = {
            "date": latest_record.name.strftime("%Y-%m-%d"),
            "time": latest_record.name.strftime("%H:%M:%S"),
            "power_kW": float(latest_record['power_kW']),
            "temperature": float(latest_record['temperature']),
            "humidity": float(latest_record['humidity'])
        }

        print(f"💾 [Archive] 準備歸檔到 [{dynamic_basket}]: {new_data_payload['date']} {new_data_payload['time']}")

        # 2. 下載目前的歷史資料
        headers = {'Content-Type': 'application/json'}
        r = requests.get(history_url, timeout=10)
        
        current_history = []
        if r.status_code == 200:
            data = r.json()
            if "data" in data and isinstance(data["data"], list):
                current_history = data["data"]
            elif isinstance(data, list):
                current_history = data
            elif isinstance(data, dict):
                current_history = [data]
        
        # 3. 重複檢查 (防呆)
        if current_history:
            last_item = current_history[-1]
            last_date = last_item.get('date', '')
            last_time = last_item.get('time', '')
            
            # 只比對到分鐘 (前5碼) 以容忍秒數差異
            new_time_short = new_data_payload['time'][:5]
            last_time_short = str(last_time)[:5]
            
            if last_date == new_data_payload['date'] and last_time_short == new_time_short:
                print("   ⚠️ [Archive] 資料已存在 (時間重複)，跳過存檔。")
                return

        # 4. 附加新資料
        current_history.append(new_data_payload)
        
        # 5. 滑動視窗保護 (避免 Basket 爆滿)
        MAX_RECORDS = 2000
        if len(current_history) > MAX_RECORDS:
            cut_count = len(current_history) - MAX_RECORDS
            current_history = current_history[-MAX_RECORDS:]
            print(f"   ✂️ [Archive] 觸發滑動視窗，移除舊資料 {cut_count} 筆")

        # 6. POST 回去
        payload_to_send = {"data": current_history}
        post_r = requests.post(history_url, json=payload_to_send, headers=headers, timeout=10)
        
        if post_r.status_code == 200:
            print(f"   ✅ [Archive] 歸檔成功！目前歷史筆數: {len(current_history)}")
        else:
            print(f"   ❌ [Archive] 上傳失敗: {post_r.text}")

    except Exception as e:
        print(f"   ❌ [Archive Error] 歸檔過程發生錯誤: {e}")

# ==========================================
# 🧠 主預測流程
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
        
        # 2. 準備三份數據
        print("📥 正在整合三方數據源...")
        
        # (A) 靜態 CSV
        hist_df = pd.read_csv(MODEL_FILES['history_data'])
        hist_df['datetime'] = pd.to_datetime(hist_df['datetime'])
        hist_df = hist_df.set_index('datetime').sort_index()
        if 'power' in hist_df.columns: hist_df = hist_df.rename(columns={'power': 'power_kW'})
        print(f"   📄 [CSV] 靜態資料: 到 {hist_df.index.max()}")
        
        # (B) 雲端補洞 (已支援跨季度)
        gap_df = fetch_recent_history_gap()
        
        # (C) 即時 Live (允許失敗)
        live_df = fetch_live_data()
        if live_df is None: 
            print("   ⚠️ [Live] 暫無即時資料，使用歷史推估")
            live_df = pd.DataFrame()
        
        # 3. 大合併
        dfs_to_concat = [df for df in [hist_df, gap_df, live_df] if not df.empty]
        if not dfs_to_concat: return None, None

        combined_df = pd.concat(dfs_to_concat)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
        combined_df['power'] = combined_df['power_kW']
        
        print(f"🎉 [Total] 整合完畢！最新時間: {combined_df.index.max()}")

        # 4. 預測
        buffer_size = 2000
        df_ready = combined_df.iloc[-buffer_size:].copy()
        
        if pd.isna(df_ready.iloc[-1]['power']) or df_ready.iloc[-1]['power'] == 0:
             valid_idx = df_ready['power'].last_valid_index()
             if valid_idx:
                 df_ready = df_ready.loc[:valid_idx]
        
        last_time = df_ready.index[-1]
        future_dates = [last_time + timedelta(hours=i+1) for i in range(24)]
        future_df = pd.DataFrame(index=future_dates, columns=df_ready.columns)
        
        future_df['temperature'] = df_ready['temperature'].iloc[-1]
        future_df['humidity'] = df_ready['humidity'].iloc[-1]
        
        full_context = pd.concat([df_ready, future_df])
        
        df_lgbm = add_lgbm_features(full_context)
        df_lstm = add_lstm_features(full_context)
        
        target_feat_lgbm = df_lgbm.iloc[-24:]
        target_feat_lstm = df_lstm.iloc[-24:]
        
        lgbm_feature_names = resources['lgbm'].feature_name()
        X_lgbm = target_feat_lgbm[lgbm_feature_names]
        pred_lgbm = resources['lgbm'].predict(X_lgbm)
        
        current_idx = -25
        seq_cols = ["power", "temperature", "humidity", "hour_sin", "hour_cos", "is_weekend"]
        dir_cols = ["lag_24h", "lag_168h", "temperature", "humidity", "hour_sin", "hour_cos", "week_sin", "week_cos", "is_weekend", "temp_squared", "rolling_mean_24h_safe", "rolling_std_24h_safe", "rolling_mean_168h", "rolling_std_168h"]
        
        seq_data = df_lstm[seq_cols].iloc[current_idx-LOOKBACK_HOURS+1 : current_idx+1]
        dir_data = df_lstm[dir_cols].iloc[current_idx+1 : current_idx+2]
        
        X_seq = resources['scaler_seq'].transform(seq_data).reshape(1, LOOKBACK_HOURS, -1)
        X_dir = resources['scaler_dir'].transform(dir_data)
        
        pred_lstm_scaled = resources['lstm'].predict([X_seq, X_dir], verbose=0)
        pred_lstm = resources['scaler_target'].inverse_transform(pred_lstm_scaled).flatten()
        
        pred_final = (pred_lgbm * resources['weights']['w_lgbm']) + (pred_lstm * resources['weights']['w_lstm'])
        
        result_df = pd.DataFrame({
            "時間": future_dates,
            "預測值": pred_final,
            "LGBM": pred_lgbm,
            "LSTM": pred_lstm
        }).set_index("時間")
        
        # =========== 🔥 執行自動歸檔 ===========
        if live_df is not None and not live_df.empty:
            print("🚀 [System] 啟動背景歸檔程序...")
            auto_archive_live_data(live_df) # 不用參數，自動判斷
        # =====================================
        
        return result_df, combined_df
        
    except Exception as e:
        print(f"❌ [Model Service Error]: {e}")
        return None, None