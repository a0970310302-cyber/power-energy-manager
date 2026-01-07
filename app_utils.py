# app_utils.py
import requests
import time
import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta

# ==========================================
# ⚙️ 全域設定與常數
# ==========================================
# 演示用放大倍率 (讓 Demo 接近真實家庭 400~500度/雙月 的水準)
DESIGN_PEAK_LOAD_KW = 20.0 

CSV_FILE_PATH = "final_training_data_with_humidity.csv"
MODEL_FILES = {
    "lgbm": "lgbm_model.pkl",
    "lstm": "lstm_model.keras",
    "scaler_seq": "scaler_seq.pkl",
    "scaler_dir": "scaler_dir.pkl",
    "scaler_target": "scaler_target.pkl",
    "weights": "ensemble_weights.pkl",
    "history_data": "final_training_data_with_humidity.csv"
}

# ==========================================
# 📅 歷史費率資料庫 (Rate History DB)
# ==========================================
# 依據台電歷年公告：2022/7, 2023/4, 2024/4 三次調整
RATES_DB = {
    # --- 2022年 (7月前) ---
    "2022_H1": {
        "progressive": {
            "summer": [1.63, 2.38, 3.52, 4.80, 5.66, 6.41],
            "non_summer": [1.63, 2.10, 2.89, 3.94, 4.60, 5.03]
        },
        "tou": {
            "summer": {"peak": 4.44, "off": 1.80},
            "non_summer": {"peak": 4.23, "off": 1.73}
        },
        "tou_peak_hours_type": "old" # 舊制：白天是尖峰
    },
    # --- 2022年 (7月後，1000度以上調漲) ---
    "2022_H2": {
        "progressive": {
            "summer": [1.63, 2.38, 3.52, 4.80, 5.66, 6.99], # >1001度調漲
            "non_summer": [1.63, 2.10, 2.89, 3.94, 4.60, 5.48]
        },
        "tou": {
            "summer": {"peak": 4.44, "off": 1.80},
            "non_summer": {"peak": 4.23, "off": 1.73}
        },
        "tou_peak_hours_type": "old"
    },
    # --- 2023年 (4月後，700度以上調漲，TOU時段改變) ---
    "2023": {
        "progressive": {
            "summer": [1.63, 2.38, 3.52, 4.80, 5.83, 7.69], # 700+調漲
            "non_summer": [1.63, 2.10, 2.89, 3.94, 4.74, 6.03]
        },
        "tou": {
            "summer": {"peak": 4.71, "off": 1.96}, # 簡易型二段式
            "non_summer": {"peak": 4.48, "off": 1.89} # 非夏月其實全日單一價(簡易型特例)，但此處模擬住商型標準
        },
        "tou_peak_hours_type": "new" # 新制：太陽光電併網，尖峰移至傍晚
    },
    # --- 2024年~2025年 (4月後，全面調漲) ---
    "2024": {
        "progressive": {
            "summer": [1.68, 2.45, 3.70, 5.04, 6.24, 8.46],
            "non_summer": [1.68, 2.16, 3.03, 4.14, 5.07, 6.63]
        },
        "tou": {
            "summer": {"peak": 5.01, "off": 1.96}, # 依據您提供的113年檔案
            "non_summer": {"peak": 4.78, "off": 1.89}
        },
        "tou_peak_hours_type": "new"
    }
}

def get_rate_config(date_obj):
    """根據日期自動選擇正確的歷史費率"""
    d = pd.to_datetime(date_obj)
    
    if d < datetime(2022, 7, 1):
        return RATES_DB["2022_H1"]
    elif d < datetime(2023, 4, 1):
        return RATES_DB["2022_H2"]
    elif d < datetime(2024, 4, 1):
        return RATES_DB["2023"]
    else:
        return RATES_DB["2024"] # 2024, 2025 使用最新

# ==========================================
# 📥 資料載入 (維持不變)
# ==========================================
def load_data():
    if not os.path.exists(CSV_FILE_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        if 'datetime' in df.columns: df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else: return pd.DataFrame()

        df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
        if 'power' in df.columns: df = df.rename(columns={'power': 'power_kW'})
        if 'power_kW' in df.columns: df['power_kW'] = pd.to_numeric(df['power_kW'], errors='coerce')
        
        # [反歸一化]
        if df['power_kW'].max() < 1.0:
            df['power_kW'] = df['power_kW'] * DESIGN_PEAK_LOAD_KW
            
        df['power_kW'] = df['power_kW'].ffill().bfill()
        if 'temperature' not in df.columns: df['temperature'] = 25.0
        if 'humidity' not in df.columns: df['humidity'] = 70.0
        return df[['power_kW', 'temperature', 'humidity']]
    except:
        return pd.DataFrame()

def load_lottiefile(filepath):
    try:
        with open(filepath, "r", encoding='utf-8') as f: return json.load(f)
    except: return None

# ==========================================
# 🧮 核心計費演算法 (支援歷史回溯)
# ==========================================
def calculate_tiered_bill(total_kwh, days_count, is_summer, rate_config=None):
    """
    計算累進電費，支援年份切換。
    """
    if rate_config is None:
        rate_config = RATES_DB["2024"] # 預設最新

    rates = rate_config["progressive"]["summer"] if is_summer else rate_config["progressive"]["non_summer"]
    
    # 判斷雙月 (超過45天視為雙月，級距 x2)
    is_bimonthly = days_count > 45
    m = 2 if is_bimonthly else 1
    
    tiers = [120, 330, 500, 700, 1000]
    tiers = [t * m for t in tiers]
    
    remaining = total_kwh
    bill = 0
    
    # 逐級計算
    for i, limit in enumerate(tiers):
        if i == 0:
            usage = min(remaining, limit)
        else:
            usage = min(remaining, limit - tiers[i-1])
            
        bill += usage * rates[i]
        remaining -= usage
        if remaining <= 0: break
            
    if remaining > 0: # 超過1000度部分
        bill += remaining * rates[5]

    return int(bill)

def analyze_pricing_plans(df):
    """
    [智慧分析] 逐筆判斷該時間點應用的費率 (2022 vs 2024)
    """
    if df is None or df.empty: return None, None
    df = df.copy()
    
    # 時間間隔
    time_factor = 0.25
    if len(df) > 1:
        time_factor = (df.index[1] - df.index[0]).total_seconds() / 3600.0
    
    df['kwh'] = df['power_kW'] * time_factor
    
    # --- 逐筆 TOU 計算 (最精確的方法) ---
    def calc_row_tou(row):
        ts = row.name
        rc = get_rate_config(ts) # 取得該時間點的費率設定
        
        m = ts.month
        h = ts.hour
        is_summer = 6 <= m <= 9
        
        # 判斷尖峰 (根據新舊制自動切換)
        is_peak = False
        if rc["tou_peak_hours_type"] == "new":
            # 新制 (2023後): 下午傍晚是尖峰
            # 簡易型二段式：夏月 09:00-24:00 尖峰 (週一~五)
            # 非夏月 06:00-11:00, 14:00-24:00 (視為尖峰，簡化模擬)
            if ts.dayofweek < 5: # 平日
                if is_summer:
                    if 9 <= h < 24: is_peak = True
                else:
                    if (6 <= h < 11) or (14 <= h < 24): is_peak = True
        else:
            # 舊制 (2022前): 白天是尖峰 (約 07:30 ~ 22:30)
            if ts.dayofweek < 5:
                if 7 <= h < 23: is_peak = True
                
        prices = rc["tou"]["summer"] if is_summer else rc["tou"]["non_summer"]
        rate = prices["peak"] if is_peak else prices["off"]
        
        return row['kwh'] * rate, 'peak' if is_peak else 'off_peak'

    # 應用計算
    tou_results = df.apply(calc_row_tou, axis=1)
    df['cost_tou'] = tou_results.apply(lambda x: x[0])
    df['tou_category'] = tou_results.apply(lambda x: x[1])
    
    # --- 累進費率計算 (需先加總 KWh，但因為費率會變，我們拆分時段計算) ---
    # 簡單做法：取出資料的「中間點日期」來決定用哪個年度的累進費率表
    # (一般帳單不會跨越太久，這樣誤差極小)
    mid_date = df.index[len(df)//2]
    rate_config_period = get_rate_config(mid_date)
    
    total_kwh = df['kwh'].sum()
    days = (df.index.max() - df.index.min()).days + 1
    # 判斷主要季節
    summer_hours = df.index.month.isin([6,7,8,9]).sum()
    is_summer_mode = summer_hours > (len(df)/2)
    
    total_prog_cost = calculate_tiered_bill(total_kwh, days, is_summer_mode, rate_config_period)
    
    results = {
        "cost_progressive": total_prog_cost,
        "cost_tou": int(df['cost_tou'].sum())
    }
    return results, df

# ==========================================
# 📊 統一計費報告
# ==========================================
def get_billing_report(df, budget=3000):
    default = {"period": "N/A", "current_bill": 0, "predicted_bill": 0, "budget": budget, "status": "safe", "usage_percent": 0.0, "savings": 0}
    if df is None or df.empty: return default
    
    latest_time = df.index[-1]
    month_start = latest_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    df_period = df[df.index >= month_start]
    
    if df_period.empty: return default
    
    res, _ = analyze_pricing_plans(df_period)
    current_bill = res['cost_progressive']
    current_tou = res['cost_tou']
    
    days_in_month = 30
    progress = max(latest_time.day / days_in_month, 0.05)
    pred_bill = current_bill / progress
    pred_tou = current_tou / progress
    
    savings = pred_bill - pred_tou
    status = "safe"
    if pred_bill > budget: status = "danger"
    elif pred_bill > budget * 0.9: status = "warning"
    
    return {
        "period": f"{month_start.strftime('%Y-%m-%d')} ~ {latest_time.strftime('%Y-%m-%d')}",
        "current_bill": int(current_bill),
        "predicted_bill": int(pred_bill),
        "budget": budget,
        "status": status,
        "usage_percent": min(pred_bill/budget, 1.0),
        "savings": int(savings)
    }

def get_core_kpis(df):
    """
    維持原本 KPI 計算邏輯，僅需確保回傳欄位完整
    """
    default_kpis = {
        "status_data_available": False, "current_load": 0, "kwh_today_so_far": 0,
        "kwh_this_month_so_far": 0, "weekly_delta_percent": 0, "kwh_last_7_days": 0,
        "last_updated": "N/A"
    }
    if df is None or df.empty: return default_kpis
    
    try:
        time_factor = 0.25
        if len(df) > 1:
            time_factor = (df.index[1] - df.index[0]).total_seconds() / 3600.0
        
        latest_time = df.index[-1]
        current_load = df['power_kW'].iloc[-1]
        
        today_start = latest_time.replace(hour=0, minute=0, second=0, microsecond=0)
        today_usage = df[df.index >= today_start]['power_kW'].sum() * time_factor
        
        month_start = latest_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        kwh_this_month = df[df.index >= month_start]['power_kW'].sum() * time_factor

        seven_days_ago = latest_time - timedelta(days=7)
        fourteen_days_ago = latest_time - timedelta(days=14)
        
        usage_last_7d = df[(df.index > seven_days_ago) & (df.index <= latest_time)]['power_kW'].sum() * time_factor
        usage_prev_7d = df[(df.index > fourteen_days_ago) & (df.index <= seven_days_ago)]['power_kW'].sum() * time_factor
        
        weekly_delta = 0
        if usage_prev_7d > 0:
            weekly_delta = ((usage_last_7d - usage_prev_7d) / usage_prev_7d) * 100

        return {
            "status_data_available": True,
            "current_load": round(current_load, 3),
            "kwh_today_so_far": round(today_usage, 2),
            "kwh_this_month_so_far": round(kwh_this_month, 2),
            "weekly_delta_percent": round(weekly_delta, 1),
            "kwh_last_7_days": round(usage_last_7d, 2),
            "last_updated": latest_time.strftime("%Y-%m-%d %H:%M")
        }
    except:
        return default_kpis