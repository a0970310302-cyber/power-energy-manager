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
# 📅 台灣電價歷史資料庫 (Time Machine Rate DB)
# 根據台電歷年公告整理 (111~114年)
# ==========================================
RATE_DATABASE = [
    {
        "id": "period_1_frozen",
        "name": "凍漲時期 (111-112年)",
        "start": "2020-01-01", 
        "end": "2024-03-31",
        # 累進費率 (非夏月/夏月)
        "prog_rates": {
            "non_summer": [1.63, 2.10, 2.89, 3.94, 4.60, 6.03],
            "summer":     [1.63, 2.38, 3.52, 4.80, 5.83, 7.69]
        },
        # 時間電價 (簡易二段式)
        "tou_rates": {
            "summer":     {"peak": 4.44, "off": 1.80}, # 推估值
            "non_summer": {"peak": 4.23, "off": 1.73}
        }
    },
    {
        "id": "period_2_hike_113",
        "name": "第一次調漲 (113年4月起)",
        "start": "2024-04-01",
        "end": "2025-09-30",
        "prog_rates": {
            # 113.04.01 實施
            "non_summer": [1.68, 2.16, 3.03, 4.14, 5.07, 6.63],
            "summer":     [1.68, 2.45, 3.70, 5.04, 6.24, 8.46]
        },
        "tou_rates": {
            "summer":     {"peak": 5.01, "off": 1.96}, 
            "non_summer": {"peak": 4.78, "off": 1.89}
        }
    },
    {
        "id": "period_3_hike_114",
        "name": "第二次調漲 (114年10月起)",
        "start": "2025-10-01",
        "end": "2099-12-31",
        "prog_rates": {
            # 114.10.01 實施 (User PDF)
            "non_summer": [1.78, 2.26, 3.13, 4.24, 5.27, 7.03],
            "summer":     [1.78, 2.55, 3.80, 5.14, 6.44, 8.86]
        },
        "tou_rates": {
            # 114.10.01 簡易型二段式
            "summer":     {"peak": 5.16, "off": 2.06}, 
            "non_summer": {"peak": 4.93, "off": 1.99}
        }
    }
]

# 定義尖峰時段 (簡易型二段式)
# 夏月: 09:00~24:00 (Peak)
# 非夏: 06:00~11:00, 14:00~24:00 (Peak)
TOU_PEAK_HOURS = {
    "summer": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "non_summer": [6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
}

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
# 🧮 核心計費演算法 (支援歷史費率切換)
# ==========================================
def get_rate_config(target_date):
    """根據日期找出當時的費率設定"""
    target_str = target_date.strftime("%Y-%m-%d")
    for period in RATE_DATABASE:
        if period["start"] <= target_str <= period["end"]:
            return period
    return RATE_DATABASE[-1] # 預設回傳最新

def calculate_tiered_bill(total_kwh, days_count, is_summer, rate_config):
    """
    計算累進電費 (需傳入當下的 rate_config)
    """
    # 判斷雙月 (超過45天視為雙月，級距 x2)
    is_bimonthly = days_count > 45
    m = 2 if is_bimonthly else 1
    
    # 根據季節選費率
    rates = rate_config["prog_rates"]["summer"] if is_summer else rate_config["prog_rates"]["non_summer"]
    
    # 級距固定 (120, 330, 500, 700, 1000)
    tiers = [120, 330, 500, 700, 1000]
    tiers = [t * m for t in tiers]
    
    remaining = total_kwh
    bill = 0
    
    # 簡化迴圈計算
    # Tier 1
    usage = min(remaining, tiers[0])
    bill += usage * rates[0]
    remaining -= usage
    
    # Tier 2 ~ 5
    for i in range(4):
        if remaining <= 0: break
        width = tiers[i+1] - tiers[i]
        usage = min(remaining, width)
        bill += usage * rates[i+1]
        remaining -= usage
        
    # Tier 6 (>1000)
    if remaining > 0:
         bill += remaining * rates[5]

    return int(bill)

def analyze_pricing_plans(df):
    """
    [時光機引擎] 自動根據資料的年份，切換成該年份的費率來計算。
    """
    if df is None or df.empty: return None, None
    df = df.copy()
    
    # 自動計算時間間隔
    time_factor = 0.25
    if len(df) > 1:
        time_factor = (df.index[1] - df.index[0]).total_seconds() / 3600.0
    
    df['kwh'] = df['power_kW'] * time_factor
    
    total_prog_cost = 0
    total_tou_cost = 0
    
    # 用來畫圖的分類
    df['tou_category'] = 'off_peak' 
    
    # --- [關鍵] 依據費率時期切分資料 ---
    for period in RATE_DATABASE:
        # 找出屬於此時期的資料
        mask = (df.index >= period["start"]) & (df.index <= period["end"])
        if not mask.any():
            continue
            
        sub_df = df.loc[mask].copy()
        
        # 1. 計算該區段的 TOU (逐筆算)
        tou_rates = period["tou_rates"]
        
        def calc_tou_row(row):
            m = row.name.month
            h = row.name.hour
            is_summer = 6 <= m <= 9
            
            is_peak = False
            if is_summer:
                if h in TOU_PEAK_HOURS['summer']: is_peak = True
            else:
                if h in TOU_PEAK_HOURS['non_summer']: is_peak = True
                
            price = tou_rates["summer"]["peak" if is_peak else "off"] if is_summer else tou_rates["non_summer"]["peak" if is_peak else "off"]
            return row['kwh'] * price

        # 計算此區段總 TOU
        segment_tou = sub_df.apply(calc_tou_row, axis=1).sum()
        total_tou_cost += segment_tou
        
        # 2. 計算該區段的 累進費率 (總量算)
        seg_kwh = sub_df['kwh'].sum()
        seg_days = (sub_df.index.max() - sub_df.index.min()).days + 1
        # 簡單判定季節 (取眾數)
        is_summer_mode = 6 <= sub_df.index.month.mode()[0] <= 9
        
        seg_prog = calculate_tiered_bill(seg_kwh, seg_days, is_summer_mode, period)
        total_prog_cost += seg_prog
        
        # 3. 標記 TOU 類別 (給畫圖用) - 簡化處理
        # 這裡我們只做簡單標記，不影響金額計算
        is_summer_mask = (sub_df.index.month >= 6) & (sub_df.index.month <= 9)
        sub_df.loc[is_summer_mask & sub_df.index.hour.isin(TOU_PEAK_HOURS['summer']), 'tou_category'] = 'peak'
        sub_df.loc[~is_summer_mask & sub_df.index.hour.isin(TOU_PEAK_HOURS['non_summer']), 'tou_category'] = 'peak'
        df.loc[mask, 'tou_category'] = sub_df['tou_category']

    # 為了相容 UI 顯示，我們把計算出的總金額平均攤回去 (雖不精確但足夠繪圖)
    df['cost_tou'] = 0 # 實際上我們只看總和
    df['cost_progressive'] = 0 
    
    results = {
        "cost_progressive": int(total_prog_cost),
        "cost_tou": int(total_tou_cost)
    }
    return results, df

# ==========================================
# 📊 統一計費報告 (Dashboard 用)
# ==========================================
def get_billing_report(df, budget=3000):
    default = {"period": "N/A", "current_bill": 0, "predicted_bill": 0, "budget": budget, "status": "safe", "usage_percent": 0.0, "savings": 0}
    if df is None or df.empty: return default
    
    latest_time = df.index[-1]
    
    # 鎖定本月/本期
    month_start = latest_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    df_period = df[df.index >= month_start]
    
    if df_period.empty: return default
    
    # 取得「當下」適用的費率 (Dashboard 永遠看最新的)
    current_rate_config = get_rate_config(latest_time)
    
    # 計算本期目前費用
    total_kwh = df_period['power_kW'].sum() * 0.25 # 假設 15min
    days_so_far = (latest_time - month_start).days + 1
    is_summer = 6 <= latest_time.month <= 9
    
    current_bill = calculate_tiered_bill(total_kwh, days_so_far, is_summer, current_rate_config)
    
    # 預測月底
    days_in_month = 30
    progress = max(latest_time.day / days_in_month, 0.05)
    pred_bill = current_bill / progress
    
    # 簡易 TOU 估算 (用於比較)
    # 這裡簡單假設 TOU 平均單價 (因為 Dashboard 不需要精確 TOU)
    tou_avg_price = 3.5 
    pred_tou = (total_kwh / progress) * tou_avg_price
    
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
        "savings": int(savings),
        "rate_name": current_rate_config['name'] # 讓前端知道現在是用哪個費率
    }

def get_core_kpis(df):
    # 維持原本邏輯，確保相容性
    # ... (使用上一版提供的 get_core_kpis 即可)
    default_kpis = {
        "status_data_available": False, "current_load": 0, "kwh_today_so_far": 0, "kwh_this_month_so_far": 0,
        "weekly_delta_percent": 0, "kwh_last_7_days": 0, "last_updated": "N/A"
    }
    if df is None or df.empty: return default_kpis
    try:
        latest_time = df.index[-1]
        current_load = df['power_kW'].iloc[-1]
        time_factor = 0.25 # 簡化
        today_start = latest_time.replace(hour=0, minute=0, second=0, microsecond=0)
        today_usage = df[df.index >= today_start]['power_kW'].sum() * time_factor
        month_start = latest_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        kwh_this_month = df[df.index >= month_start]['power_kW'].sum() * time_factor
        
        # 過去7天
        seven_days_ago = latest_time - timedelta(days=7)
        usage_last_7d = df[df.index > seven_days_ago]['power_kW'].sum() * time_factor
        
        return {
            "status_data_available": True,
            "current_load": round(current_load, 3),
            "kwh_today_so_far": round(today_usage, 2),
            "kwh_this_month_so_far": round(kwh_this_month, 2),
            "weekly_delta_percent": 0, # 簡化
            "kwh_last_7_days": round(usage_last_7d, 2),
            "last_updated": latest_time.strftime("%Y-%m-%d %H:%M")
        }
    except:
        return default_kpis