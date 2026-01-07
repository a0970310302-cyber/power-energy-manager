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
# ⚙️ 全域設定與常數
# ==========================================
POWER_PANTRY_ID = "6a2e85f5-4af4-4efd-bb9f-c5604fe8475e"
TARGET_YEARS = [2023, 2024, 2025, 2026]
CSV_FILE_PATH = "final_training_data_with_humidity.csv"

# 1. 模型檔案路徑
MODEL_FILES = {
    "lgbm": "lgbm_model.pkl",
    "lstm": "lstm_model.keras",
    "scaler_seq": "scaler_seq.pkl",
    "scaler_dir": "scaler_dir.pkl",
    "scaler_target": "scaler_target.pkl",
    "weights": "ensemble_weights.pkl",
    "history_data": "final_training_data_with_humidity.csv"
}

# 2. 時間電價費率表 (Time-of-Use Rates)
TOU_RATES_DATA = {
    "summer": {
        "dates": "6/1 ~ 9/30",
        "peak_price": 6.0,
        "off_peak_price": 1.8,
        "peak_hours": [16, 17, 18, 19, 20, 21]
    },
    "non_summer": {
        "dates": "10/1 ~ 5/31",
        "peak_price": 5.0,
        "off_peak_price": 1.7,
        "peak_hours": [15, 16, 17, 18, 19, 20]
    }
}

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
        # print(f"⚠️ 找不到 Lottie 檔案: {filepath}") # 可註解掉以減少噴錯
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
        
        # 補齊環境參數 (若無則給預設值)
        if 'temperature' not in df.columns:
            df['temperature'] = 25.0
        if 'humidity' not in df.columns:
            df['humidity'] = 70.0
            
        return df[['power_kW', 'temperature', 'humidity']]
        
    except Exception as e:
        print(f"❌ 讀取 CSV 時發生錯誤: {e}")
        return pd.DataFrame()

# ==========================================
# 🧠 模型載入工具
# ==========================================
def load_model(path=None):
    """
    載入 .pkl 模型檔案。如果不指定 path，則預設載入 LGBM 模型。
    """
    if path is None:
        path = MODEL_FILES.get("lgbm", "lgbm_model.pkl")

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
# 📊 關鍵指標計算 (KPIs)
# ==========================================
def get_core_kpis(df):
    """
    計算首頁、儀表板、分析頁面所需的「所有」關鍵指標
    """
    # 預設回傳字典 (防止 KeyError)
    default_kpis = {
        "status_data_available": False,
        "current_load": 0,
        "kwh_today_so_far": 0,
        "kwh_this_month_so_far": 0,
        "weekly_delta_percent": 0,
        "kwh_last_7_days": 0,
        "last_updated": "N/A"
    }

    if df is None or df.empty:
        return default_kpis
    
    try:
        latest_time = df.index[-1]
        
        # 1. 目前負載 (kW)
        current_load = df['power_kW'].iloc[-1]
        
        # 2. 今日累積用電 (kWh)
        today_start = latest_time.replace(hour=0, minute=0, second=0, microsecond=0)
        today_df = df[df.index >= today_start]
        today_usage = today_df['power_kW'].sum() * 0.25
        
        # 3. 本月累積用電 (kWh)
        month_start = latest_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_df = df[df.index >= month_start]
        kwh_this_month_so_far = month_df['power_kW'].sum() * 0.25

        # 4. 過去 7 天趨勢 (Analysis 頁面用)
        seven_days_ago = latest_time - timedelta(days=7)
        fourteen_days_ago = latest_time - timedelta(days=14)
        
        usage_last_7d = df[(df.index > seven_days_ago) & (df.index <= latest_time)]['power_kW'].sum() * 0.25
        usage_prev_7d = df[(df.index > fourteen_days_ago) & (df.index <= seven_days_ago)]['power_kW'].sum() * 0.25
        
        if usage_prev_7d > 0:
            weekly_delta = ((usage_last_7d - usage_prev_7d) / usage_prev_7d) * 100
        else:
            weekly_delta = 0

        return {
            "status_data_available": True,
            "current_load": round(current_load, 3),
            "kwh_today_so_far": round(today_usage, 2),
            "kwh_this_month_so_far": round(kwh_this_month_so_far, 2),
            "weekly_delta_percent": round(weekly_delta, 1),
            "kwh_last_7_days": round(usage_last_7d, 2),
            "last_updated": latest_time.strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        print(f"⚠️ KPI 計算錯誤: {e}")
        return default_kpis

# ==========================================
# ⚡ 電費分析邏輯 (核心計算引擎)
# ==========================================
def analyze_pricing_plans(df):
    """
    [底層引擎] 接收一個 DataFrame，逐筆計算出 '累進制' 與 '時間電價' 的成本。
    """
    if df is None or df.empty:
        return None
        
    df = df.copy()
    
    # 費率常數提取
    summer_peak = TOU_RATES_DATA['summer']['peak_price']
    summer_off = TOU_RATES_DATA['summer']['off_peak_price']
    non_summer_peak = TOU_RATES_DATA['non_summer']['peak_price']
    non_summer_off = TOU_RATES_DATA['non_summer']['off_peak_price']
    summer_hours = TOU_RATES_DATA['summer']['peak_hours']
    non_summer_hours = TOU_RATES_DATA['non_summer']['peak_hours']

    # 1. 累進費率估算 (Simplified Progressive)
    # 註：這裡做的是簡化版逐筆估算，實務上累進是看總量，但為了與 TOU 比較趨勢，這裡假設基礎費率
    def calculate_progressive_cost(row):
        month = row.name.month
        is_summer = 6 <= month <= 9
        rate = 4.5 if is_summer else 3.5  # 平均費率假設
        return row['power_kW'] * 0.25 * rate # kW -> kWh -> $

    # 2. 時間電價估算 (TOU) - 精確計算
    def calculate_tou_cost(row):
        month = row.name.month
        hour = row.name.hour
        is_summer = 6 <= month <= 9
        
        is_peak = False
        if is_summer:
            if hour in summer_hours: is_peak = True
        else:
            if hour in non_summer_hours: is_peak = True
            
        if is_summer:
            rate = summer_peak if is_peak else summer_off
        else:
            rate = non_summer_peak if is_peak else non_summer_off
            
        return row['power_kW'] * 0.25 * rate # kW -> kWh -> $

    df['cost_progressive'] = df.apply(calculate_progressive_cost, axis=1)
    df['cost_tou'] = df.apply(calculate_tou_cost, axis=1)
    
    # 用於分析頁的分類 (Peak/Off-Peak)
    df['tou_category'] = 'off_peak'
    
    # 標記尖離峰 (向量化加速)
    is_summer_mask = (df.index.month >= 6) & (df.index.month <= 9)
    df.loc[is_summer_mask & df.index.hour.isin(summer_hours), 'tou_category'] = 'peak'
    df.loc[~is_summer_mask & df.index.hour.isin(non_summer_hours), 'tou_category'] = 'peak'
    
    # 增加一個 kwh 欄位方便後續加總
    df['kwh'] = df['power_kW'] * 0.25
    
    return df

# ==========================================
# 💰 全能計費報告 (High-Level API)
# ==========================================
def get_billing_report(df, budget=3000):
    """
    【全能計費中心】
    輸入歷史數據，自動鎖定「本月」，同時計算兩種費率與預估狀態。
    供 Dashboard, Home, Analysis 三個頁面共用。
    """
    default_report = {
        "period": "N/A",
        "current_bill": 0,
        "potential_tou_bill": 0,
        "predicted_bill": 0,
        "budget": budget,
        "status": "safe",
        "usage_percent": 0.0,
        "savings": 0,
        "recommendation_msg": "資料不足，無法分析"
    }

    if df is None or df.empty:
        return default_report

    try:
        # 1. 鎖定本月數據 (This Month So Far)
        latest_time = df.index[-1]
        month_start = latest_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # 為了安全起見，如果這個月才剛開始 (ex: 1號)，我們往回抓一點避免空值，或至少抓到最後一筆
        df_this_month = df[df.index >= month_start]
        
        if df_this_month.empty:
            return default_report

        # 2. 呼叫底層引擎計算詳細成本
        df_analyzed = analyze_pricing_plans(df_this_month)
        
        # 3. 統計目前累積金額 (Actual So Far)
        # 注意：這裡加上一個基礎費修正 (Base Charge)，假設累進制底度較高
        # 為了讓 Dashboard 顯示的錢比較有感，我們對累進制做一個分段計算修正
        total_kwh = df_analyzed['kwh'].sum()
        
        # [累進制] 分段計費邏輯 (更精準的估算)
        # 夏月/非夏月判斷
        is_summer_now = 6 <= latest_time.month <= 9
        
        def calc_prog_bill(kwh, is_summer):
            # 簡易兩段式模擬：500度以下 / 500度以上
            rate1 = 3.52 if is_summer else 2.89 # 較低級距
            rate2 = 4.80 if is_summer else 3.94 # 較高級距
            
            if kwh <= 300:
                return kwh * rate1
            else:
                return 300 * rate1 + (kwh - 300) * rate2
        
        current_bill_prog = calc_prog_bill(total_kwh, is_summer_now)
        current_bill_tou = df_analyzed['cost_tou'].sum() # TOU 直接加總即可
        
        # 4. 月底預測 (Projection)
        # 計算本月進度比例：目前是第幾天 / 本月總天數
        # 例如 1月10日，進度約 10/31。 預測值 = 目前值 / 進度
        days_in_month = 31 # 簡易假設
        if latest_time.month == 2: days_in_month = 28
        elif latest_time.month in [4, 6, 9, 11]: days_in_month = 30
        
        current_day = latest_time.day
        progress_ratio = max(current_day / days_in_month, 0.05) # 避免除以 0
        
        # 預估月底帳單 (Projected Bill)
        projected_bill = current_bill_prog / progress_ratio
        
        # 5. 狀態判定
        status = "safe"
        if projected_bill > budget:
            status = "danger"
        elif projected_bill > budget * 0.9:
            status = "warning"
            
        usage_percent = min(projected_bill / budget, 1.0)
        
        # 6. 節費建議 (Savings & Insight)
        # 比較：若整個月都用 TOU 會省多少？
        projected_tou = current_bill_tou / progress_ratio
        savings = projected_bill - projected_tou
        
        recommendation = ""
        if savings > 150:
            recommendation = f"建議切換時間電價，本月預計可省 ${int(savings):,} 元"
        elif savings < -100:
             recommendation = f"目前累進費率最優，切換反而會貴 ${int(abs(savings)):,} 元"
        else:
            recommendation = "目前方案合適，無須更動"

        return {
            "period": f"{month_start.strftime('%Y-%m-%d')} ~ {latest_time.strftime('%Y-%m-%d')}",
            "current_bill": int(current_bill_prog),       # 給 Dashboard (實用性)
            "potential_tou_bill": int(current_bill_tou),  # 給 Analysis (獨特性)
            "predicted_bill": int(projected_bill),        # 給 Dashboard 進度條
            "budget": budget,
            "status": status,                             # 給 Home/Dashboard 燈號
            "usage_percent": usage_percent,
            "savings": int(savings),                      # 給 Home 通知
            "recommendation_msg": recommendation
        }

    except Exception as e:
        print(f"⚠️ Billing Report Error: {e}")
        return default_report