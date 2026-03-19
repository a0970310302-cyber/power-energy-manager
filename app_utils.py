# app_utils.py
import requests
import time
import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta
import calendar
import streamlit as st

# ==========================================
# ⚙️ 全域設定與常數
# ==========================================
# 根據真實帳單校正後的倍率
DESIGN_PEAK_LOAD_KW = 3.6

CSV_FILE_PATH = "final_training_data_with_humidity.csv"

MODEL_FILES = {
    "config": "hybrid_residual.pkl",    # 總指揮官 (含 Scalers)
    "lgbm": "lgbm_residual.pkl",        # 殘差修正模型
    "lstm": "lstm_hybrid.keras",        # 序列預測模型
    "history_data": "final_training_data_with_humidity.csv"
}

def get_current_bill_cycle(current_date=None):
    """
    計算當前日期所屬的帳單週期 (奇數月結算制)
    解決 12月 跨年導致月份 13 的錯誤
    """
    if current_date is None:
        current_date = datetime.now()
    
    year = current_date.year
    month = current_date.month
    
    if month == 1:
        start_year = year - 1
        start_month = 12
        end_year = year
        end_month = 1
    elif month == 12:
        start_year = year
        start_month = 12
        end_year = year + 1  # 跨年
        end_month = 1
    elif month % 2 == 0:
        start_year = year
        start_month = month
        end_year = year
        end_month = month + 1
    else:
        start_year = year
        start_month = month - 1
        end_year = year
        end_month = month
        
    start_date = datetime(start_year, start_month, 1)
    last_day = calendar.monthrange(end_year, end_month)[1]
    end_date = datetime(end_year, end_month, last_day, 23, 59, 59)
    
    return start_date, end_date

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
# 📥 資料載入 (已統一資料源)
# ==========================================
@st.cache_data(ttl=3600) # 🌟 核心修改：快取 1 小時，避免重複讀取磁碟與重複處理資料邏輯
def load_data():
    """
    從本地 CSV 讀取並清洗資料。
    使用 st.cache_data 確保全域只有一份處理好的 DataFrame。
    """
    if not os.path.exists(CSV_FILE_PATH): 
        return pd.DataFrame()
    try:
        # 只在第一次執行時會讀取磁碟
        df = pd.read_csv(CSV_FILE_PATH)
        
        # 統一時間 index
        if 'datetime' in df.columns: 
            df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns: 
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else: 
            return pd.DataFrame()

        df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
        
        # 雙向欄位綁定與數值轉換
        if 'power' in df.columns and 'power_kW' not in df.columns:
            df['power_kW'] = df['power']
        elif 'power_kW' in df.columns and 'power' not in df.columns:
            df['power'] = df['power_kW']
            
        df['power'] = pd.to_numeric(df['power'], errors='coerce')
        df['power'] = df['power'].ffill().bfill()
        df['power_kW'] = df['power']
        
        # 氣象特徵填補
        if 'temperature' not in df.columns: df['temperature'] = 25.0
        if 'humidity' not in df.columns: df['humidity'] = 70.0
        df['temperature'] = df['temperature'].ffill().bfill()
        df['humidity'] = df['humidity'].ffill().bfill()
        
        print("✅ [Cache Miss] 成功從 CSV 讀取並處理資料")
        return df
    except Exception as e:
        print(f"❌ Error in load_data: {e}")
        return pd.DataFrame()
    
@st.cache_data
def load_lottiefile(filepath):
    try:
        with open(filepath, "r", encoding='utf-8') as f: return json.load(f)
    except: return None

# ==========================================
# 🧠 模型載入工具
# ==========================================
def load_model(path=None):
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
    
    time_factor = 1
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
    
    return {"cost_progressive": total_prog_cost, "cost_tou": int(df['cost_tou'].sum()), "total_kwh": total_kwh}, df

    # 🌟 將括號內增加一個 current_time=None 參數
def get_billing_report(df, budget=1000, current_time=None):
    default = {"period": "N/A", "current_bill": 0, "predicted_bill": 0, "potential_tou_bill":0, "budget": budget, "status": "safe", "usage_percent": 0.0, "savings": 0, "recommendation_msg": "N/A"}
    if df is None or df.empty: return default
    
    # 🌟 修改 1：如果沒有指定時間，才使用資料的最後一筆
    if current_time is None:
        current_time = df.index[-1]
    
    # 🌟 修改 2：改用 current_time 來尋找帳單週期
    cycle_start, cycle_end = get_current_bill_cycle(current_time)
    
    df_period = df[(df.index >= cycle_start) & (df.index <= cycle_end)]
    
    if df_period.empty: return default
    
    res, _ = analyze_pricing_plans(df_period)
    total_bill_projected = res['cost_progressive'] 
    total_tou_projected = res['cost_tou']
    total_kwh_projected = res.get('total_kwh', 0) # 🌟 取得剛剛回傳的總預估度數
    
    # 🌟 新增：台電級距推算邏輯
    days_count = (df_period.index.max() - df_period.index.min()).days + 1
    is_bimonthly = days_count > 45
    m = 2 if is_bimonthly else 1
    tiers = [t * m for t in [120, 330, 500, 700, 1000]]
    
    current_tier = 1
    next_tier_kwh = tiers[0]
    for i, limit in enumerate(tiers):
        if total_kwh_projected <= limit:
            current_tier = i + 1
            next_tier_kwh = limit
            break
    else:
        current_tier = 6 # 超過最高級距
        next_tier_kwh = None
    
    kwh_to_next_tier = (next_tier_kwh - total_kwh_projected) if next_tier_kwh else 0
    total_bill_projected = res['cost_progressive'] 
    total_tou_projected = res['cost_tou']
    
    # 🌟 修改 3：把原本的 datetime.now() 改成 current_time，確保時間比較基準一致
    df_actual = df_period[df_period.index <= current_time]
    
    if not df_actual.empty:
        res_actual, _ = analyze_pricing_plans(df_actual)
        current_bill = res_actual['cost_progressive']
    else:
        current_bill = 0

    pred_bill = int(total_bill_projected)
    savings = pred_bill - int(total_tou_projected)
    
    status = "safe"
    if pred_bill > budget: status = "danger"
    elif pred_bill > budget * 0.9: status = "warning"
    
    recommendation = ""
    if savings > 30: 
        recommendation = f"建議切換時間電價，本期預計可省 ${int(savings):,} 元"
    elif savings < -30: 
        recommendation = f"累進費率目前最優，切換反而貴 ${int(abs(savings)):,} 元"
    else: 
        recommendation = "目前方案合適，無顯著價差"


    return {
        "period": f"{cycle_start.strftime('%Y-%m-%d')} ~ {cycle_end.strftime('%Y-%m-%d')}",
        "current_bill": int(current_bill),
        "predicted_bill": int(pred_bill),
        "potential_tou_bill": int(total_tou_projected),
        "budget": budget,
        "status": status,
        "usage_percent": min(pred_bill/budget, 1.0) if budget > 0 else 0,
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
        time_factor = 1
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
        if usage_prev_7d > 0.1: 
            weekly_delta = ((usage_last_7d - usage_prev_7d) / usage_prev_7d) * 100

        return {
            "period": f"{cycle_start.strftime('%Y-%m-%d')} ~ {cycle_end.strftime('%Y-%m-%d')}",
            "current_bill": int(current_bill),
            "predicted_bill": int(pred_bill),
            "potential_tou_bill": int(total_tou_projected),
            "budget": budget,
            "status": status,
            "usage_percent": min(pred_bill/budget, 1.0) if budget > 0 else 0,
            "savings": int(savings),
            "recommendation_msg": recommendation,
            "current_tier": current_tier,             # 新增欄位
            "total_kwh": total_kwh_projected,         # 新增欄位
            "kwh_to_next_tier": kwh_to_next_tier      # 新增欄位
        }

    except: return default_kpis
