# page_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np 
import time  # 🌟 用於計算運算耗時

# 匯入共用函式
from app_utils import load_data, get_core_kpis, get_billing_report, get_current_bill_cycle
from model_service import load_resources_and_predict

def show_dashboard_page():
    """
    顯示「用電儀表板」的內容
    """
    # --- 1. 資料獲取 ---
    # 優先從 session_state 獲取數據以維持快取一致性
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

    # ==========================================
    # 區塊 1: 帳單預算監控
    # ==========================================
    st.header("💰 帳單預算監控")
    report = get_billing_report(df_history)
    
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
    
    # --- 3.1 參數計算 ---
    latest_time = df_history.index[-1]
    cycle_start, cycle_end = get_current_bill_cycle(latest_time)
    hours_to_end = int((cycle_end - latest_time).total_seconds() // 3600)
    
    # --- 3.2 互動選單與對齊優化 ---
    # 使用 vertical_alignment="bottom" 確保下拉選單與右側按鈕水平對齊
    col_sel, col_status = st.columns([2, 1], vertical_alignment="bottom")
    
    with col_sel:
        view_option = st.selectbox(
            "選擇預測顯示範圍：",
            ["48 小時 (短期)", "7 天 (168小時)", "本期結算日 (直到截止)"],
            index=0,
            help="選擇不同的時間跨度，查看 AI 模擬的電力走勢。"
        )
    
    # 映射選單到目標步數
    step_map = {
        "48 小時 (短期)": 48,
        "7 天 (168小時)": 168,
        "本期結算日 (直到截止)": hours_to_end
    }
    target_steps = step_map[view_option]
    
    # 檢查 session 中已存在的預測數據長度
    current_pred_df = st.session_state.get("prediction_result", None)
    current_len = len(current_pred_df) if current_pred_df is not None else 0

    with col_status:
        # 情況 A：數據不足，顯示執行按鈕
        if current_len < target_steps:
            if st.button(f"🚀 執行深度預測 ({target_steps}h)", use_container_width=True, type="primary"):
                start_time = time.time()  # 開始計時
                with st.spinner(f"AI 核心運算中..."):
                    new_pred, _ = load_resources_and_predict(steps=target_steps)
                    st.session_state.prediction_result = new_pred
                
                end_time = time.time()  # 結束計時
                duration = end_time - start_time
                st.session_state.last_calc_time = duration # 存入 session 以便顯示
                st.rerun()
        # 情況 B：數據充足，顯示成功狀態
        else:
            st.success("數據已就緒", icon="✅")

    # 顯示上一次運算的耗時資訊（若存在）
    if "last_calc_time" in st.session_state:
        st.caption(f"⏱️ 最近一次 AI 深度運算耗時：{st.session_state.last_calc_time:.2f} 秒")

    # --- 3.3 圖表繪製 ---
    tab_chart, tab_data = st.tabs(["趨勢圖表", "詳細歷史數據"])
    
    with tab_chart:
        # A. 準備歷史數據 (從帳單週期開始到現在)
        df_hist_plot = df_history[(df_history.index >= cycle_start)].copy()
        plot_data = []
        
        if not df_hist_plot.empty:
            # 清理末尾可能的異常 0 值 (若有)
            while not df_hist_plot.empty and (df_hist_plot.iloc[-1]['power_kW'] <= 0):
                df_hist_plot = df_hist_plot.iloc[:-1]
                
            h_data = df_hist_plot[['power_kW']].reset_index()
            h_data.columns = ['time', 'value']
            h_data['type'] = '歷史實績 (Actual)'
            plot_data.append(h_data)

        # B. 準備預測數據並處理「連線縫合」
        if st.session_state.get("prediction_result") is not None:
            pred_res = st.session_state.prediction_result.copy()
            
            # 根據使用者選取的範圍過濾顯示數據
            display_end = latest_time + timedelta(hours=target_steps)
            pred_res = pred_res[pred_res.index <= display_end]
            
            p_data = pred_res[['預測值']].reset_index()
            p_data.columns = ['time', 'value']
            p_data['type'] = 'AI 預測 (Forecast)'
            
            # 🌟 縫合邏輯：抓取歷史最後一點作為預測線的起點，消除斷點
            if not df_hist_plot.empty:
                last_hist_point = h_data.iloc[[-1]].copy()
                last_hist_point['type'] = 'AI 預測 (Forecast)'
                p_data = pd.concat([last_hist_point, p_data]).reset_index(drop=True)
            
            plot_data.append(p_data)

        # C. 渲染 Plotly 圖表
        if plot_data:
            df_final_chart = pd.concat(plot_data)
            
            color_map = {
                '歷史實績 (Actual)': '#00CC96', # 翠綠
                'AI 預測 (Forecast)': '#EF553B' # 鮮紅
            }
            
            fig = px.line(df_final_chart, x='time', y='value', color='type',
                          color_discrete_map=color_map,
                          title=f"電力滾動趨勢 ({view_option})",
                          template="plotly_dark")
            
            fig.update_traces(mode='lines+markers', marker=dict(size=4)) # 讓預測與歷史都顯示點
            
            fig.update_layout(
                xaxis_title="時間",
                yaxis_title="功率 (kW)",
                legend_title="數據類型",
                hovermode="x unified"
            )
            
            # 強制鎖定 X 軸範圍，確保選單切換時縮放正確
            fig.update_xaxes(range=[latest_time - timedelta(days=3), latest_time + timedelta(hours=target_steps)])
            
            # 加入「現在時間」垂直線
            fig.add_vline(x=latest_time.timestamp() * 1000, line_width=1, line_dash="solid", line_color="white")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("尚無足夠數據繪製趨勢圖。")

    with tab_data:
        st.dataframe(df_history.tail(100), use_container_width=True)