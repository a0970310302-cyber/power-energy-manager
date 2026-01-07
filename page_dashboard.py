# page_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np 

# 匯入共用函式 (含新增的 get_current_bill_cycle)
from app_utils import load_data, get_core_kpis, get_billing_report, get_current_bill_cycle

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

    kpis = get_core_kpis(df_history)

    st.title("💡 家庭智慧電管家")
    st.caption(f"{data_source_msg} | AI 滾動修正模組：Online") 

    if not kpis['status_data_available']:
        st.warning("資料量不足，部分指標可能無法計算。")

    # ==========================================
    # 區塊 1: 帳單監控
    # ==========================================
    st.header("💰 帳單預算監控")
    report = get_billing_report(df_history)
    
    st.info(f"📅 **本期帳單週期： {report['period']}**")
    
    c1, c2 = st.columns(2)
    c1.metric("💸 目前累積電費 (已知)", f"NT$ {report['current_bill']:,}", delta="已定案")
    
    delta_val = report['predicted_bill'] - report['budget']
    delta_msg = f"超支 {delta_val:,} 元" if delta_val > 0 else f"省下 {abs(delta_val):,} 元"
    delta_color = "inverse"
    
    c2.metric("🔮 AI 預估結算 (本期)", f"NT$ {report['predicted_bill']:,}", 
              delta=delta_msg, delta_color=delta_color)

    usage_percent = report['usage_percent']
    st.write(f"**預算消耗進度 (目標：NT$ {report['budget']:,})**")
    
    bar_caption = f"✅ 狀態良好：目前預測佔預算 {usage_percent*100:.1f}%"
    if usage_percent > 1.0 or report['status'] == "danger":
        bar_caption = f"⚠️ 警告：預測即將超支！目前預測佔預算 {usage_percent*100:.1f}%"
    
    st.progress(min(usage_percent, 1.0))
    st.caption(bar_caption)
    
    st.divider()

    # ==========================================
    # 區塊 2: 即時用電
    # ==========================================
    st.subheader("⚡ 即時用電狀態")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("今日累積用電", f"{kpis['kwh_today_so_far']:.2f} kWh")
    k2.metric("當前功率", f"{kpis['current_load']:.3f} kW")
    k3.metric("近 7 天累積", f"{kpis['kwh_last_7_days']:.1f} kWh")
    k4.metric("本期累積用量", f"{kpis['kwh_this_month_so_far']:.1f} kWh")

    st.divider()

    # ==========================================
    # 區塊 3: 滾動預測趨勢圖 (修正為雙月全景模式)
    # ==========================================
    st.subheader("📈 雙月滾動式修正趨勢 (全週期監控)")
    
    tab1, tab2 = st.tabs(["預測 vs 真實", "詳細歷史數據"])
    
    with tab1:
        # [核心修正] 取得當期帳單的起訖日
        latest_time = df_history.index[-1]
        cycle_start, cycle_end = get_current_bill_cycle(latest_time)
        
        # 1. 準備歷史資料 (鎖定本週期)
        df_hist_plot = df_history[(df_history.index >= cycle_start) & (df_history.index <= cycle_end)].copy()
        
        # 清理無效值
        if not df_hist_plot.empty:
            while not df_hist_plot.empty and (df_hist_plot.iloc[-1]['power_kW'] <= 0):
                df_hist_plot = df_hist_plot.iloc[:-1]

        df_hist_plot = df_hist_plot[['power_kW']].reset_index()
        df_hist_plot.columns = ['time', 'value']
        df_hist_plot['type'] = '真實數據 (Actual)'
        
        # 2. 準備預測資料
        df_pred_plot = pd.DataFrame()
        if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
            pred_res = st.session_state.prediction_result.copy()
            
            # 篩選出本週期的預測值 (不顯示下個週期的)
            pred_res = pred_res[(pred_res.index >= cycle_start) & (pred_res.index <= cycle_end)]
            
            if not df_hist_plot.empty and not pred_res.empty:
                last_hist_point = pd.DataFrame({
                    'time': [df_hist_plot.iloc[-1]['time']], 
                    'value': [df_hist_plot.iloc[-1]['value']],
                    'type': ['AI 預測 (Forecast)'] 
                })
                future_pred = pred_res[['預測值']].reset_index()
                future_pred.columns = ['time', 'value']
                future_pred['type'] = 'AI 預測 (Forecast)'
                
                df_pred_plot = pd.concat([last_hist_point, future_pred])
            elif not pred_res.empty:
                df_pred_plot = pred_res[['預測值']].reset_index()
                df_pred_plot.columns = ['time', 'value']
                df_pred_plot['type'] = 'AI 預測 (Forecast)'

        # 3. 繪圖
        if not df_hist_plot.empty or not df_pred_plot.empty:
            df_chart = pd.concat([df_hist_plot, df_pred_plot])
            
            fig = px.line(df_chart, x='time', y='value', color='type', 
                          color_discrete_map={'真實數據 (Actual)': '#00CC96', 'AI 預測 (Forecast)': '#EF553B'},
                          line_dash='type',
                          line_dash_map={'真實數據 (Actual)': 'solid', 'AI 預測 (Forecast)': 'dash'},
                          title=f"帳單週期全程監控 ({cycle_start.strftime('%m/%d')} ~ {cycle_end.strftime('%m/%d')})",
                          template="plotly_dark")
            
            # 鎖定 X 軸範圍 (這是實現「全景圖」的關鍵)
            fig.update_xaxes(range=[cycle_start, cycle_end])
            
            # 標示目前時間點
            last_real_time = df_hist_plot['time'].iloc[-1] if not df_hist_plot.empty else datetime.now()
            fig.add_vline(x=last_real_time.timestamp() * 1000, line_width=1, line_dash="dot", line_color="white")
            
            # 如果還有預測數據，標註 "Now"
            max_val = df_chart['value'].max() if not df_chart.empty else 1
            fig.add_annotation(x=last_real_time.timestamp() * 1000, y=max_val, 
                               text="Now", showarrow=True, arrowhead=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            if not df_pred_plot.empty:
                 st.info(f"ℹ️ AI 已推算至本期結算日 ({cycle_end.strftime('%m/%d')})，橘色虛線為預測走勢。")
        else:
            st.info("尚無本期數據。")
        
        with st.expander("ℹ️ 技術原理：Hybrid Model"):
            st.write("""
            本系統結合 **LightGBM** 與 **LSTM**。
            上方橘色虛線即為兩種模型加權後的最終預測結果。
            """)

    with tab2:
        st.dataframe(df_history.tail(100))