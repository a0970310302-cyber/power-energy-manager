# app_utils.py
import requests
import time
import pandas as pd
import numpy as np
import os
import re

# ==========================================
# ⚙️ 設定 (離線模式)
# ==========================================
# 雖然離線模式下不連網，但保留變數定義避免其他模組引用報錯
POWER_PANTRY_ID = "6a2e85f5-4af4-4efd-bb9f-c5604fe8475e"
TARGET_YEARS = [2023, 2024, 2025, 2026]
CSV_FILE_PATH = "final_training_data_with_humidity.csv"

def load_lottieurl(url: str):
    """
    載入 Lottie 動畫 (如果 Lottie 伺服器正常則可運作，若失敗回傳 None)
    """
    try:
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def load_data():
    """
    [修改版] 離線模式：直接讀取本地 CSV 檔案，不進行網路請求
    """
    print("📂 [App Utils] 正在讀取本地歷史資料 (離線模式)...")
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ 錯誤：找不到檔案 {CSV_FILE_PATH}")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        
        # --- 時間解析 ---
        # 嘗試解析 CSV 常見的時間欄位名稱
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
        # UI 介面通常預期欄位名稱為 'power_kW'，但 CSV 可能是 'power'
        if 'power' in df.columns:
            df = df.rename(columns={'power': 'power_kW'})
            
        # 確保數據為數值型態
        if 'power_kW' in df.columns:
            df['power_kW'] = pd.to_numeric(df['power_kW'], errors='coerce')
        
        # --- 簡單資料清洗 ---
        # 處理標記為缺失的數據
        if 'isMissingData' in df.columns:
            # 將字串 '1' 或數值 1 視為缺失
            df.loc[df['isMissingData'] == 1, 'power_kW'] = np.nan
            df.loc[df['isMissingData'] == '1', 'power_kW'] = np.nan
            
        # 補值 (與 model_service 保持一致)
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

def analyze_pricing_plans(df):
    """
    電費分析邏輯 (保持不變)
    """
    if df is None or df.empty:
        return None
        
    df = df.copy()
    # 確保有 hour 和 month 欄位
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    
    # 費率設定 (台電 2024 參考費率)
    
    # 1. 累進費率 (以 330度, 500度, 700度 為級距簡化估算)
    # 非夏月 (10月-5月) / 夏月 (6月-9月)
    def calculate_progressive_cost(row):
        is_summer = 6 <= row.name.month <= 9
        # 這裡僅做單一小時的估算 (假設每小時都在最低級距，實際應以月總量計算，此為示意)
        # 簡化：平均每度電 3.5 元 (非夏月) / 4.5 元 (夏月)
        rate = 4.5 if is_summer else 3.5
        return row['power_kW'] * rate

    # 2. 時間電價 (兩段式)
    # 尖峰：夏月 16:00-22:00, 非夏月 15:00-21:00 (簡化示意)
    # 離峰：其他時間
    def calculate_tou_cost(row):
        month = row.name.month
        hour = row.name.hour
        is_summer = 6 <= month <= 9
        
        is_peak = False
        if is_summer:
            if 16 <= hour < 22: is_peak = True
        else:
            if 15 <= hour < 21: is_peak = True
            
        # 費率 (參考)
        peak_rate = 6.0 if is_summer else 5.0
        off_peak_rate = 1.8 if is_summer else 1.7
        
        rate = peak_rate if is_peak else off_peak_rate
        return row['power_kW'] * rate

    df['cost_progressive'] = df.apply(calculate_progressive_cost, axis=1)
    df['cost_tou'] = df.apply(calculate_tou_cost, axis=1)
    
    return df