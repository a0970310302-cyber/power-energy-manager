import streamlit as st
import pandas as pd
import joblib
import os
import json
import time
from datetime import datetime, timedelta

# 確保 data_loader.py 和 model_trainer.py 在同一個資料夾
try:
    from data_loader import load_all_history_data
except ImportError:
    st.error("錯誤：找不到 data_loader.py。")
    # 在這裡我們不能 st.stop()，所以回傳一個函式讓主程式處理
    def load_all_history_data():
        return pd.DataFrame()

# --- 1. Lottie 動畫載入函式 ---
@st.cache_data
def load_lottiefile(filepath: str):
    """
    輔助函式，用於從本地 JSON 檔案載入 Lottie 動畫。
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # st.error(f"錯誤：找不到 Lottie 動畫檔案 '{filepath}'。") # 主程式會處理
        return None
    except Exception as e:
        # st.error(f"載入本地 Lottie 檔案時發生錯誤：{e}") # 主程式會處理
        return None

# --- 2. 核心快取功能 (Caching) ---
@st.cache_resource
def load_model(model_path="model.pkl"):
    """
    載入訓練好的 AI 模型。
    """
    if not os.path.exists(model_path):
        # st.error(f"錯誤：找不到模型檔案 '{model_path}'。") # 主程式會處理
        return None
    try:
        time.sleep(2) # 模擬模型載入
        model = joblib.load(model_path)
        return model
    except Exception as e:
        # st.error(f"載入模型時發生錯誤：{e}") # 主程式會處理
        return None

@st.cache_data
def load_data():
    """
    載入並清理所有歷史數據。
    """
    try:
        time.sleep(1) # 模擬數據載入
        df_history = load_all_history_data()
        if df_history.empty:
            # st.warning("警告：未載入任何歷史資料。") # 主程式會處理
            return pd.DataFrame()
        return df_history
    except Exception as e:
        # st.error(f"載入歷史資料時發生錯誤：{e}") # 主程式會處理
        return pd.DataFrame()

# --- 3. 電價計算邏輯 (共用) ---
PROGRESSIVE_RATES = [
    (120, 1.68, 1.68), (210, 2.45, 2.16), (170, 3.70, 3.03),
    (200, 5.04, 4.14), (300, 6.24, 5.07), (float('inf'), 8.46, 6.63)
]
TOU_RATES_DATA = {
    'basic_fee_monthly': 75.0, 'surcharge_kwh_threshold': 2000.0, 'surcharge_rate_per_kwh': 0.99,
    'rates': {'summer': {'peak': 4.71, 'off_peak': 1.85}, 'nonsummer': {'peak': 4.48, 'off_peak': 1.78}}
}

def calculate_progressive_cost(total_kwh_month, is_summer):
    """
    輔助函式：計算單月的「累進電價」總電費
    """
    cost = 0
    kwh_remaining = total_kwh_month
    rate_index = 1 if is_summer else 2
    for (bracket_kwh, *rates) in PROGRESSIVE_RATES:
        rate = rates[rate_index - 1]
        if kwh_remaining <= 0: break
        kwh_in_bracket = min(kwh_remaining, bracket_kwh)
        cost += kwh_in_bracket * rate
        kwh_remaining -= kwh_in_bracket
    return cost

def get_tou_details(timestamp):
    """
    輔助函式：根據時間戳記返回TOU類別和費率
    """
    is_summer = (timestamp.month >= 6) and (timestamp.month <= 9)
    is_weekend = timestamp.dayofweek >= 5
    hour = timestamp.hour
    category = 'off_peak'
    if not is_weekend:
        if is_summer:
            if 9 <= hour < 24: category = 'peak'
        else:
            if (6 <= hour < 11) or (14 <= hour < 24): category = 'peak'
    season = 'summer' if is_summer else 'nonsummer'
    rate = TOU_RATES_DATA['rates'][season][category]
    return category, rate, is_summer

@st.cache_data
def analyze_pricing_plans(df_period):
    """
    核心分析函式：比較「累進電價」與「TOU時間電價」
    """
    df_analysis = df_period.copy()
    # 1. TOU 成本
    tou_details = df_analysis.index.map(get_tou_details)
    df_analysis['tou_category'] = [cat for cat, rate, season in tou_details]
    df_analysis['tou_rate'] = [rate for cat, rate, season in tou_details]
    df_analysis['is_summer'] = [season for cat, rate, season in tou_details]
    df_analysis['kwh'] = df_analysis['power_kW'] * 0.25
    df_analysis['tou_flow_cost'] = df_analysis['kwh'] * df_analysis['tou_rate']
    monthly_tou = df_analysis.resample('MS').agg(kwh=('kwh', 'sum'), flow_cost=('tou_flow_cost', 'sum'))
    monthly_tou['basic_fee'] = TOU_RATES_DATA['basic_fee_monthly']
    threshold = TOU_RATES_DATA['surcharge_kwh_threshold']
    surcharge_rate = TOU_RATES_DATA['surcharge_rate_per_kwh']
    monthly_tou['surcharge'] = monthly_tou['kwh'].apply(lambda x: max(0, x - threshold) * surcharge_rate)
    monthly_tou['total_cost'] = monthly_tou['flow_cost'] + monthly_tou['basic_fee'] + monthly_tou['surcharge']
    total_cost_tou = monthly_tou['total_cost'].sum()
    
    # 2. 累進電價 成本
    monthly_prog = df_analysis.resample('MS').agg(kwh=('kwh', 'sum'))
    monthly_prog['is_summer'] = (monthly_prog.index.month >= 6) & (monthly_prog.index.month <= 9)
    monthly_prog['total_cost'] = monthly_prog.apply(lambda row: calculate_progressive_cost(row['kwh'], row['is_summer']), axis=1)
    total_cost_progressive = monthly_prog['total_cost'].sum()
    
    # 3. 結果
    results = {'total_kwh': df_analysis['kwh'].sum(), 'cost_progressive': total_cost_progressive, 'cost_tou': total_cost_tou}
    return results, df_analysis

# --- 4. 核心 KPI 計算函式 ---
def get_core_kpis(df_history):
    """
    計算所有頁面共用的核心 KPI
    """
    # 初始化
    kpis = {
        'projected_cost': 0, 'kwh_this_month_so_far': 0, 'kwh_last_7_days': 0,
        'kwh_previous_7_days': 0, 'weekly_delta_percent': 0, 'status_data_available': False,
        'peak_kwh': 0, 'off_peak_kwh': 0, 'PRICE_PER_KWH_AVG': 3.5,
        'kwh_today_so_far': 0, 'cost_today_so_far': 0, 'latest_data': None
    }
    
    if df_history.empty:
        return kpis # 返回初始值

    try:
        # --- 預估電費 (累進) ---
        kwh_last_30d = df_history.last('30D')['power_kW'].sum() * 0.25
        today = df_history.index.max()
        is_summer_now = (today.month >= 6) & (today.month <= 9)
        kpis['projected_cost'] = calculate_progressive_cost(kwh_last_30d, is_summer_now)
        if kwh_last_30d > 0:
            kpis['PRICE_PER_KWH_AVG'] = kpis['projected_cost'] / kwh_last_30d
        
        # --- 今日數據 ---
        today_start = df_history.index.max().normalize()
        df_today = df_history.loc[today_start:]
        kpis['kwh_today_so_far'] = (df_today['power_kW'].sum() * 0.25)
        kpis['cost_today_so_far'] = kpis['kwh_today_so_far'] * kpis['PRICE_PER_KWH_AVG']

        # --- 本月累積 ---
        today_date = df_history.index.max().date()
        start_of_month = today_date.replace(day=1)
        if start_of_month < df_history.index.min().date():
            start_of_month = df_history.index.min().date()
        df_this_month = df_history.loc[start_of_month:]
        kpis['kwh_this_month_so_far'] = (df_this_month['power_kW'].sum() * 0.25)

        # --- 用電狀態 (週) ---
        df_last_7d = df_history.last('7D')
        kpis['kwh_last_7_days'] = (df_last_7d['power_kW'].sum() * 0.25)
        start_of_prev_7d = (df_last_7d.index.min() - timedelta(days=7))
        end_of_prev_7d = df_last_7d.index.min()
        
        if start_of_prev_7d >= df_history.index.min():
            df_prev_7d = df_history.loc[start_of_prev_7d:end_of_prev_7d]
            kpis['kwh_previous_7_days'] = (df_prev_7d['power_kW'].sum() * 0.25)
            if kpis['kwh_previous_7_days'] > 0: 
                kpis['weekly_delta_percent'] = ((kpis['kwh_last_7_days'] - kpis['kwh_previous_7_days']) / kpis['kwh_previous_7_days']) * 100
            kpis['status_data_available'] = True
        
        # --- 尖峰/離峰 (TOU) ---
        df_last_30d = df_history.last('30D').copy()
        tou_details_30d = df_last_30d.index.map(get_tou_details)
        df_last_30d['tou_category'] = [cat for cat, rate, season in tou_details_30d] # 【⭐ 修正點】
        df_last_30d['kwh'] = df_last_30d['power_kW'] * 0.25
        kpis['peak_kwh'] = df_last_30d[df_last_30d['tou_category'] == 'peak']['kwh'].sum()
        kpis['off_peak_kwh'] = df_last_30d[df_last_30d['tou_category'] == 'off_peak']['kwh'].sum()

        # --- 最新數據 ---
        kpis['latest_data'] = df_history.iloc[-1]

        return kpis

    except Exception as e:
        st.error(f"核心 KPI 計算錯誤: {e}")
        return kpis # 返回初始值