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
# 演示用放大倍率 (維持 20.0 以符合真實家庭負載)
DESIGN_PEAK_LOAD_KW = 20.0 

CSV_FILE_PATH = "final_training_data_with_humidity.csv"

# [關鍵修改] 更新為新版混合模型的檔案路徑
MODEL_FILES = {
    "config": "hybrid_residual.pkl",    # 總指揮官 (含 Scalers)
    "lgbm": "lgbm_residual.pkl",        # 殘差修正模型
    "lstm": "lstm_hybrid.keras",        # 序列預測模型
    "history_data": "final_training_data_with_humidity.csv"
}

# ==========================================
# 📅 歷史費率資料庫 (Rate History DB)
# ==========================================
RATES_DB = {
    "2022_H1": {
        "progressive": {
            "summer": [1.63, 2.38, 3.52, 4.80, 5.66, 6.41],
            "non_summer": [1.63, 2.10, 2.89, 3.94, 4.60, 5.03]
        },
        "tou": {"summer": {"peak": 4.44, "off": 1.80}, "non_summer": {"peak": 4.23, "off": 1.73}},
        "tou_peak_hours_type": "old"
    },
    "2022_H2": {
        "progressive": {
            "summer": [1.63, 2.38, 3.52, 4.80, 5.66, 6.99],
            "non_summer": [1.63, 2.10, 2.89, 3.94, 4.60, 5.48]
        },
        "tou": {"summer": {"peak": 4.44, "off": 1.80}, "non_summer": {"peak": 4.23, "off": 1.73}},
        "tou_peak_hours_type": "old"
    },
    "2023": {
        "progressive": {
            "summer": [1.63, 2.38, 3.52, 4.80, 5.83, 7.69],
            "non_summer": [1.63, 2.10, 2.89, 3.94, 4.74, 6.03]
        },
        "tou": {"summer": {"peak": 4.71, "off": 1.96}, "non_summer": {"peak": 4.48, "off": 1.89}},
        "tou_peak_hours_type": "new"
    },
    "2024": {
        "progressive": {
            "summer": [1.68, 2.45, 3.70, 5.04, 6.24, 8.46],
            "non_summer": [1.68, 2.16, 3.03, 4.14, 5.07, 6.63]
        },
        "tou": {"summer": {"peak": 5.01, "off": 1.96}, "non_summer": {"peak": 4.78, "off": 1.89}},
        "tou_peak_hours_type": "new"
    },
    "2025": {
        "progressive": {
            "summer": [1.78, 2.55, 3.80, 5.14, 6.44, 8.86],
            "non_summer": [1.78, 2.26, 3.13, 4.24, 5.27, 7.03]
        },
        "tou": {"summer": {"peak": 5.11, "off": 2.06}, "non_summer": {"peak": 4.88, "off": 1.99}},
        "tou_peak_hours_type": "new"
    }
}

def get_rate_config(date_obj):
    d = pd.to_datetime(date_obj)
    if d < datetime(2022, 7, 1): return RATES_DB["2022_H1"]
    elif d < datetime(2023, 4, 1): return RATES_DB["2022_H2"]
    elif d < datetime(2024, 4, 1): return RATES_DB["2023"]
    elif d < datetime(2025, 10, 1): return RATES_DB["2024"]
    else: return RATES_DB["2025"]

# ==========================================
# 📥 資料載入
# ==========================================
def load_data():
    if not os.path.exists(CSV_FILE_PATH): return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        if 'datetime' in df.columns: df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else: return pd.DataFrame()

        df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
        if 'power' in df.columns: df = df.rename(columns={'power': 'power_kW'})
        if 'power_kW' in df.columns: df['power_kW'] = pd.to_numeric(df['power_kW'], errors='coerce')
        
        # [Reality Booster] 放大倍率
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
# 🧠 模型載入工具
# ==========================================
def load_model(path=None):
    # 預設載入 config (因為現在它是核心)
    if path is None: path = MODEL_FILES.get("config", "hybrid_residual.pkl")
    try:
        if not os.path.exists(path): return None
        return joblib.load(path)
    except: return None

# ==========================================
# 🧮 核心計費演算法
# ==========================================
def calculate_tiered_bill(total_kwh, days_count, is_summer, rate_config=None):
    if rate_config is None: rate_config = RATES_DB["2024"]
    rates = rate_config["progressive"]["summer"] if is_summer else rate_config["progressive"]["non_summer"]
    
    is_bimonthly = days_count > 45
    m = 2 if is_bimonthly else 1
    tiers = [t * m for t in [120, 330, 500, 700, 1000]]
    
    remaining = total_kwh
    bill = 0
    
    for i, limit in enumerate(tiers):
        if i == 0: usage = min(remaining, limit)
        else: usage = min(remaining, limit - tiers[i-1])
        bill += usage * rates[i]
        remaining -= usage
        if remaining <= 0: break
            
    if remaining > 0: bill += remaining * rates[5]
    return int(bill)

def analyze_pricing_plans(df):
    if df is None or df.empty: return None, None
    df = df.copy()
    
    time_factor = 0.25
    if len(df) > 1:
        time_factor = (df.index[1] - df.index[0]).total_seconds() / 3600.0
    
    df['kwh'] = df['power_kW'] * time_factor
    
    def calc_row_tou(row):
        ts = row.name
        rc = get_rate_config(ts)
        m, h = ts.month, ts.hour
        is_summer = 6 <= m <= 9
        
        is_peak = False
        if rc["tou_peak_hours_type"] == "new":
            if ts.dayofweek < 5:
                if is_summer:
                    if 9 <= h < 24: is_peak = True
                else:
                    if (6 <= h < 11) or (14 <= h < 24): is_peak = True
        else:
            if ts.dayofweek < 5 and (7 <= h < 23): is_peak = True
                
        prices = rc["tou"]["summer"] if is_summer else rc["tou"]["non_summer"]
        rate = prices["peak"] if is_peak else prices["off"]
        return row['kwh'] * rate, 'peak' if is_peak else 'off_peak'

    tou_results = df.apply(calc_row_tou, axis=1)
    df['cost_tou'] = tou_results.apply(lambda x: x[0])
    df['tou_category'] = tou_results.apply(lambda x: x[1])
    
    mid_date = df.index[len(df)//2]
    rate_config_period = get_rate_config(mid_date)
    
    total_kwh = df['kwh'].sum()
    days = (df.index.max() - df.index.min()).days + 1
    is_summer_mode = df.index.month.isin([6,7,8,9]).sum() > (len(df)/2)
    
    total_prog_cost = calculate_tiered_bill(total_kwh, days, is_summer_mode, rate_config_period)
    
    return {"cost_progressive": total_prog_cost, "cost_tou": int(df['cost_tou'].sum())}, df

def get_billing_report(df, budget=3000):
    default = {"period": "N/A", "current_bill": 0, "predicted_bill": 0, "potential_tou_bill":0, "budget": budget, "status": "safe", "usage_percent": 0.0, "savings": 0, "recommendation_msg": "N/A"}
    if df is None or df.empty: return default
    
    latest_time = df.index[-1]
    month_start = latest_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    df_period = df[df.index >= month_start]
    
    if df_period.empty: return default
    
    res, _ = analyze_pricing_plans(df_period)
    current_bill = res['cost_progressive']
    current_tou = res['cost_tou']
    
    progress = max(latest_time.day / 30, 0.05)
    pred_bill = current_bill / progress
    
    savings = pred_bill - (current_tou / progress)
    status = "safe"
    if pred_bill > budget: status = "danger"
    elif pred_bill > budget * 0.9: status = "warning"
    
    recommendation = ""
    if savings > 150: recommendation = f"建議切換時間電價，本月預計可省 ${int(savings):,} 元"
    elif savings < -100: recommendation = f"累進費率目前最優，切換反而貴 ${int(abs(savings)):,} 元"
    else: recommendation = "目前方案合適"

    return {
        "period": f"{month_start.strftime('%Y-%m-%d')} ~ {latest_time.strftime('%Y-%m-%d')}",
        "current_bill": int(current_bill),
        "predicted_bill": int(pred_bill),
        "potential_tou_bill": int(current_tou),
        "budget": budget,
        "status": status,
        "usage_percent": min(pred_bill/budget, 1.0),
        "savings": int(savings),
        "recommendation_msg": recommendation
    }

def get_core_kpis(df):
    default_kpis = {
        "status_data_available": False, "current_load": 0, "kwh_today_so_far": 0,
        "kwh_this_month_so_far": 0, "weekly_delta_percent": 0, "kwh_last_7_days": 0,
        "last_updated": "N/A"
    }
    if df is None or df.empty: return default_kpis
    try:
        time_factor = 0.25
        if len(df) > 1: time_factor = (df.index[1] - df.index[0]).total_seconds() / 3600.0
        
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
        if usage_prev_7d > 0: weekly_delta = ((usage_last_7d - usage_prev_7d) / usage_prev_7d) * 100

        return {
            "status_data_available": True,
            "current_load": round(current_load, 3),
            "kwh_today_so_far": round(today_usage, 2),
            "kwh_this_month_so_far": round(kwh_this_month, 2),
            "weekly_delta_percent": round(weekly_delta, 1),
            "kwh_last_7_days": round(usage_last_7d, 2),
            "last_updated": latest_time.strftime("%Y-%m-%d %H:%M")
        }
    except: return default_kpis