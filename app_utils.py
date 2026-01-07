# app_utils.py
import requests
import time
import pandas as pd
import numpy as np
import os
import re
import json
from datetime import datetime, timedelta

# ==========================================
# ⚙️ 設定 (離線模式)
# ==========================================
POWER_PANTRY_ID = "6a2e85f5-4af4-4efd-bb9f-c5604fe8475e"
TARGET_YEARS = [2023, 2024, 2025, 2026]
CSV_FILE_PATH = "final_training_data_with_humidity.csv"

# ==========================================
# 🎨 Lottie 動畫載入工具
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
# 📥 資料載入邏輯 (離線版)
# ==========================================
def load_data():
    """
    離線模式：直接讀取本地 CSV 檔案
    """
    print("📂 [App Utils] 正在讀取本地歷史資料 (離線模式)...")
    
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
            df.loc[df['isMissingData'] == 1, 'power_kW'] = np.nan
            df.loc[df['isMissingData'] == '1', 'power_kW'] = np.nan
            
        df['power_kW'] = df['power_kW'].ffill().bfill()
        
        if 'temperature' not in df.columns:
            df['temperature'] = 25.0
        if 'humidity' not in df.columns:
            df['humidity'] = 70.0
            
        print(f"✅ [App Utils] 資料載入成功！範圍: {df.index.min()} ~ {df.index.max()}")
        return df[['power_kW', 'temperature', 'humidity']]
        
    except Exception as e:
        print(f"❌ 讀取 CSV 時發生錯誤: {e}")
        return pd.DataFrame()

# ==========================================
# 📊 關鍵指標計算 (KPIs) - [補回]
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
    
    # 取得最新一筆資料的時間
    latest_time = df.index[-1]
    
    # 1. 目前負載 (kW)
    current_load = df['power_kW'].iloc[-1]
    
    # 2. 今日累積用電 (kWh)
    # 定義「今日」的範圍 (從當天 00:00 到最新時間)
    today_start = latest_time.replace(hour=0, minute=0, second=0, microsecond=0)
    today_df = df[df.index >= today_start]
    # 計算方式：功率(kW) * 時間(0.25小時, 因為是15分一筆)
    today_usage = today_df['power_kW'].sum() * 0.25
    
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
# ⚡ 電費分析邏輯
# ==========================================
def analyze_pricing_plans(df):
    if df is None or df.empty:
        return None
        
    df = df.copy()
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'month' not in df.columns:
        df['month'] = df.index.month
    
    # 1. 累進費率估算
    def calculate_progressive_cost(row):
        is_summer = 6 <= row.name.month <= 9
        rate = 4.5 if is_summer else 3.5
        return row['power_kW'] * rate

    # 2. 時間電價估算 (TOU)
    def calculate_tou_cost(row):
        month = row.name.month
        hour = row.name.hour
        is_summer = 6 <= month <= 9
        
        is_peak = False
        if is_summer:
            if 16 <= hour < 22: is_peak = True
        else:
            if 15 <= hour < 21: is_peak = True
            
        peak_rate = 6.0 if is_summer else 5.0
        off_peak_rate = 1.8 if is_summer else 1.7
        
        rate = peak_rate if is_peak else off_peak_rate
        return row['power_kW'] * rate

    df['cost_progressive'] = df.apply(calculate_progressive_cost, axis=1)
    df['cost_tou'] = df.apply(calculate_tou_cost, axis=1)
    
    return df