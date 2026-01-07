# app_utils.py
import requests
import time
import pandas as pd
import numpy as np
import os
import re
import json
import joblib
from datetime import datetime, timedelta

# ==========================================
# ⚙️ 全域設定與常數 (補齊所有頁面需要的變數)
# ==========================================
POWER_PANTRY_ID = "6a2e85f5-4af4-4efd-bb9f-c5604fe8475e"
TARGET_YEARS = [2023, 2024, 2025, 2026]
CSV_FILE_PATH = "final_training_data_with_humidity.csv"

# 1. 模型檔案路徑 (page_analysis.py 需要)
MODEL_FILES = {
    "lgbm": "lgbm_model.pkl",
    "lstm": "lstm_model.keras",
    "scaler_seq": "scaler_seq.pkl",
    "scaler_dir": "scaler_dir.pkl",
    "scaler_target": "scaler_target.pkl",
    "weights": "ensemble_weights.pkl",
    "history_data": "final_training_data_with_humidity.csv"
}

# 2. 時間電價費率表 (page_analysis.py 需要)
# 這裡定義了 夏月/非夏月 的 尖峰/離峰 價格與時段，供分析圖表參考
TOU_RATES_DATA = {
    "summer": {
        "dates": "6/1 ~ 9/30",
        "peak_price": 6.0,      # 尖峰電價 (假設值)
        "off_peak_price": 1.8,  # 離峰電價
        "peak_hours": [16, 17, 18, 19, 20, 21] # 16:00~22:00
    },
    "non_summer": {
        "dates": "10/1 ~ 5/31",
        "peak_price": 5.0,
        "off_peak_price": 1.7,
        "peak_hours": [15, 16, 17, 18, 19, 20] # 15:00~21:00
    }
}

# ==========================================
# 🎨 Lottie 動畫載入工具 (app.py 需要)
# ==========================================
def load_lottiefile(filepath: str):
    """
    載入本地 Lottie JSON 檔案
    """
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 找不到 Lottie 檔案: {filepath}")
        return None
    except Exception as e:
        print(f"⚠️ Lottie 載入錯誤: {e}")
        return None

def load_lottieurl(url: str):
    """
    載入網路 Lottie 動畫 URL
    """
    try:
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# ==========================================
# 📥 資料載入邏輯 (所有頁面共用 - 離線版)
# ==========================================
def load_data():
    """
    離線模式：直接讀取本地 CSV 檔案
    """
    # print("📂 [App Utils] 正在讀取本地歷史資料 (離線模式)...") # 減少 log 雜訊
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ 錯誤：找不到檔案 {CSV_FILE_PATH}")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        # --- 時間解析 ---
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            print("⚠️ CSV 中找不到時間欄位 (datetime 或 timestamp)")
            return pd.DataFrame()

        df = df.dropna(subset=['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # --- 欄位名稱標準化 ---
        if 'power' in df.columns:
            df = df.rename(columns={'power': 'power_kW'})
            
        if 'power_kW' in df.columns:
            df['power_kW'] = pd.to_numeric(df['power_kW'], errors='coerce')
        
        # --- 資料清洗 ---
        if 'isMissingData' in df.columns:
            # 處理各種可能的缺失標記
            df.loc[df['isMissingData'] == 1, 'power_kW'] = np.nan
            df.loc[df['isMissingData'] == '1', 'power_kW'] = np.nan
            
        df['power_kW'] = df['power_kW'].ffill().bfill()
        
        if 'temperature' not in df.columns:
            df['temperature'] = 25.0
        if 'humidity' not in df.columns:
            df['humidity'] = 70.0
            
        # print(f"✅ [App Utils] 資料載入成功！") 
        return df[['power_kW', 'temperature', 'humidity']]
        
    except Exception as e:
        print(f"❌ 讀取 CSV 時發生錯誤: {e}")
        return pd.DataFrame()

# ==========================================
# 🧠 模型載入工具 (page_analysis.py 需要)
# ==========================================
def load_model(path):
    """
    載入 .pkl 模型檔案
    """
    try:
        if not os.path.exists(path):
            print(f"⚠️ 找不到模型檔案: {path}")
            return None
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"❌ 無法載入模型 {path}: {e}")
        return None

# ==========================================
# 📊 關鍵指標計算 (page_home.py 需要)
# ==========================================
def get_core_kpis(df):
    """
    計算首頁顯示的關鍵指標：今日用電、目前負載、昨日對比
    """
    if df is None or df.empty:
        return {
            "current_load": 0,
            "today_usage": 0,
            "yesterday_usage": 0,
            "delta_percent": 0,
            "last_updated": "N/A"
        }
    
    latest_time = df.index[-1]
    
    # 1. 目前負載 (kW)
    current_load = df['power_kW'].iloc[-1]
    
    # 2. 今日累積用電 (kWh)
    today_start = latest_time.replace(hour=0, minute=0, second=0, microsecond=0)
    today_df = df[df.index >= today_start]
    today_usage = today_df['power_kW'].sum() * 0.25 # 假設每15分鐘一筆，轉為kWh
    
    # 3. 昨日同期累積用電 (kWh)
    yesterday_start = today_start - timedelta(days=1)
    yesterday_end = latest_time - timedelta(days=1)
    yesterday_df = df[(df.index >= yesterday_start) & (df.index <= yesterday_end)]
    yesterday_usage = yesterday_df['power_kW'].sum() * 0.25
    
    # 4. 差異百分比
    if yesterday_usage > 0:
        delta_percent = ((today_usage - yesterday_usage) / yesterday_usage) * 100
    else:
        delta_percent = 0
        
    return {
        "current_load": round(current_load, 3),
        "today_usage": round(today_usage, 2),
        "yesterday_usage": round(yesterday_usage, 2),
        "delta_percent": round(delta_percent, 1),
        "last_updated": latest_time.strftime("%Y-%m-%d %H:%M")
    }

# ==========================================
# ⚡ 電費分析邏輯 (page_home.py, page_dashboard.py 需要)
# ==========================================
def analyze_pricing_plans(df):
    if df is None or df.empty:
        return None
        
    df = df.copy()
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'month' not in df.columns:
        df['month'] = df.index.month
    
    # 引用上方定義的 TOU_RATES_DATA 來保持一致性
    summer_peak_price = TOU_RATES_DATA['summer']['peak_price']
    summer_off_price = TOU_RATES_DATA['summer']['off_peak_price']
    non_summer_peak_price = TOU_RATES_DATA['non_summer']['peak_price']
    non_summer_off_price = TOU_RATES_DATA['non_summer']['off_peak_price']

    # 1. 累進費率估算 (簡易版)
    def calculate_progressive_cost(row):
        is_summer = 6 <= row.name.month <= 9
        # 假設費率
        rate = 4.5 if is_summer else 3.5
        return row['power_kW'] * rate

    # 2. 時間電價估算 (TOU)
    def calculate_tou_cost(row):
        month = row.name.month
        hour = row.name.hour
        is_summer = 6 <= month <= 9
        
        is_peak = False
        if is_summer:
            # 使用 TOU_RATES_DATA 定義的時段 (16~22)
            if hour in TOU_RATES_DATA['summer']['peak_hours']: 
                is_peak = True
        else:
            # 使用 TOU_RATES_DATA 定義的時段 (15~21)
            if hour in TOU_RATES_DATA['non_summer']['peak_hours']: 
                is_peak = True
            
        if is_summer:
            rate = summer_peak_price if is_peak else summer_off_price
        else:
            rate = non_summer_peak_price if is_peak else non_summer_off_price
            
        return row['power_kW'] * rate

    df['cost_progressive'] = df.apply(calculate_progressive_cost, axis=1)
    df['cost_tou'] = df.apply(calculate_tou_cost, axis=1)
    
    return df