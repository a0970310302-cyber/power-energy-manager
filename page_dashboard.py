# page_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np 
import time

# 匯入共用函式
from app_utils import load_data, get_core_kpis, get_billing_report, get_current_bill_cycle
from model_service import load_resources_and_predict

def show_dashboard_page():
    """
    顯示「用電儀表板」的內容
    """
    # --- 1. 資料獲取 ---
    if "current_data" in st.session_state and st.session_state.current_data is not None:
        df_history = st.session_state.current_data
        data_source_msg = "🟢 即時數據 (Live Data)"
    else:
        df_history = load_data()
        data_source_msg = "🟠 歷史存檔 (Offline Data)"
    
    if df_history is None or df_history.empty:
        st.warning("儀表板無資料可顯示。")
        return

    # 計算基礎用電指標
    kpis = get_core_kpis(df_history)

    st.title("💡 智慧電能管家")
    st.caption(f"{data_source_msg} | Hybrid AI 運算引擎：Online") 

    if not kpis['status_data_available']:
        st.warning("資料量不足，部分指標可能無法計算。")
    
    # 🌟 嘗試讀取離線預測快取 (全域共用)
    try:
        pred_cache = pd.read_csv("prediction_cache.csv")
        pred_cache['datetime'] = pd.to_datetime(pred_cache['datetime'])
        pred_cache = pred_cache.set_index('datetime')
        st.session_state.prediction_result = pred_cache
        data_ready = True
        
        # 將 AI 預測的欄位重新命名以符合 app_utils 的計算格式
        pred_for_bill = pred_cache.copy()
        pred_for_bill['power_kW'] = pred_for_bill['預測值']
        future_pred = pred_for_bill[pred_for_bill.index > df_history.index[-1]]
        df_combined = pd.concat([df_history, future_pred[['power_kW']]])
        
    except FileNotFoundError:
        st.session_state.prediction_result = None
        data_ready = False
        df_combined = df_history 

    # ==========================================
    # 區塊 1: 帳單預算監控
    # ==========================================
    st.header("💰 帳單預算監控")

    # 明確告訴計費程式，真正的「現在」是純歷史資料的最後一筆時間
    true_current_time = df_history.index[-1]
    report = get_billing_report(df_combined, current_time=true_current_time)
    
    st.info(f"📅 **本期帳單週期： {report['period']}**")
    
    c1, c2 = st.columns(2)
    c1.metric("💸 目前累積電費 (已知)", f"NT$ {report['current_bill']:,}", delta="已定案")
    
    delta_val = report['predicted_bill'] - report['budget']
    delta_msg = f"超支 {delta_val:,} 元" if delta_val > 0 else f"省下 {abs(delta_val):,} 元"
    
    c2.metric("🔮 AI 預估結算 (本期)", f"NT$ {report['predicted_bill']:,}", 
              delta=delta_msg, delta_color="inverse")

    usage_percent = report['usage_percent']
    st.write(f"**預算消耗進度 (目標：NT$ {report['budget']:,})**")
    
    bar_caption = f"✅ 狀態良好：目前預測佔預算 {usage_percent*100:.1f}%"
    if usage_percent > 1.0 or report['status'] == "danger":
        bar_caption = f"⚠️ 警告：預測即將超支！目前預測佔預算 {usage_percent*100:.1f}%"
    
    st.progress(min(usage_percent, 1.0))
    st.caption(bar_caption)
    
    st.divider()

    # ==========================================
    # 區塊 2: 即時用電狀態
    # ==========================================
    st.subheader("⚡ 即時用電狀態")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("今日累積用電", f"{kpis['kwh_today_so_far']:.2f} kWh")
    k2.metric("當前功率", f"{kpis['current_load']:.3f} kW")
    k3.metric("近 7 天累積", f"{kpis['kwh_last_7_days']:.1f} kWh")
    k4.metric("本期累積用量", f"{kpis['kwh_this_month_so_far']:.1f} kWh")

    st.divider()
    
    # ==========================================
    # 區塊 3: 滾動預測趨勢圖 (動態範圍與效能監測)
    # ==========================================
    st.subheader("📈 預測趨勢與效能評估")
    
    latest_time = df_history.index[-1]
    
    # --- 互動選單 ---
    col_sel, col_status = st.columns([2, 1], vertical_alignment="bottom")
    with col_sel:
        view_option = st.selectbox(
            "選擇預測顯示範圍：",
            [
                "48 小時 (短期)", 
                "7 天 (168小時)", 
                "1 個月 (約30天)",     # 🌟 新增 1 個月
                "2 個月 (約60天)",     # 🌟 新增 2 個月
                "本期結算日 (直到截止)" # 依你的要求，放在這兩個的下方
            ],
            index=0
        )
    
    # 映射選單到目標步數
    cycle_start, cycle_end = get_current_bill_cycle(latest_time)
    hours_to_end = int((cycle_end - latest_time).total_seconds() // 3600)
    
    step_map = {
        "48 小時 (短期)": 48,
        "7 天 (168小時)": 168,
        "1 個月 (約30天)": 720,        # 🌟 30天 * 24小時 = 720
        "2 個月 (約60天)": 1440,       # 🌟 60天 * 24小時 = 1440
        "本期結算日 (直到截止)": hours_to_end
    }
    view_steps = step_map[view_option]

    with col_status:
        # 🌟 直接使用前面判定過的 data_ready 狀態
        if data_ready:
            st.success("⚡ AI 預測數據已就緒 (Offline Inference)", icon="✅")
        else:
            st.warning("🔄 系統正在背景進行首次推論，請稍後重整。", icon="⏳")

    # --- 3.3 圖表繪製 ---
    tab_chart, tab_data = st.tabs(["趨勢圖表", "詳細歷史數據"])
    
    with tab_chart:
        df_hist_plot = df_history[(df_history.index >= cycle_start)].copy()
        plot_data = []
        
        if not df_hist_plot.empty:
            while not df_hist_plot.empty and (df_hist_plot.iloc[-1]['power_kW'] <= 0):
                df_hist_plot = df_hist_plot.iloc[:-1]
                
            h_data = df_hist_plot[['power_kW']].reset_index()
            h_data.columns = ['time', 'value']
            h_data['type'] = '歷史實績 (Actual)'
            plot_data.append(h_data)

        if st.session_state.get("prediction_result") is not None:
            pred_res = st.session_state.prediction_result.copy()

            display_end = latest_time + timedelta(hours=view_steps)
            pred_res = pred_res[(pred_res.index > latest_time) & (pred_res.index <= display_end)]
            
            p_data = pred_res[['預測值']].reset_index()
            p_data.columns = ['time', 'value']
            p_data['type'] = 'AI 預測 (Forecast)'
            
            if not df_hist_plot.empty:
                last_hist_point = h_data.iloc[[-1]].copy()
                last_hist_point['type'] = 'AI 預測 (Forecast)'
                p_data = pd.concat([last_hist_point, p_data]).reset_index(drop=True)
            
            plot_data.append(p_data)

        if plot_data:
            df_final_chart = pd.concat(plot_data)
            
            color_map = {
                '歷史實績 (Actual)': '#00CC96', 
                'AI 預測 (Forecast)': '#EF553B' 
            }
            
            fig = px.line(df_final_chart, x='time', y='value', color='type',
                          color_discrete_map=color_map,
                          title=f"電力滾動趨勢 ({view_option})",
                          template="plotly_dark")
            
            fig.update_traces(mode='lines+markers', marker=dict(size=4)) 
            
            fig.update_layout(
                xaxis_title="時間",
                yaxis_title="功率 (kW)",
                legend_title="數據類型",
                hovermode="x unified"
            )
            
            fig.update_xaxes(range=[latest_time - timedelta(days=3), latest_time + timedelta(hours=view_steps)])
            
            fig.add_vline(x=latest_time.timestamp() * 1000, line_width=1, line_dash="solid", line_color="white")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("尚無足夠數據繪製趨勢圖。")

    with tab_data:
        st.dataframe(df_history.tail(100), use_container_width=True)
