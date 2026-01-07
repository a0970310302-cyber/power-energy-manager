# page_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np 

# 匯入共用函式
from app_utils import load_data, get_core_kpis, get_billing_report

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
    # 區塊 1: 帳單監控 (改用統一報表)
    # ==========================================
    st.header("💰 帳單預算監控")
    
    # [關鍵修改] 呼叫全能計費中心
    # 這裡的數據來源跟首頁是同一個大腦，所以不會有矛盾
    report = get_billing_report(df_history)
    
    st.info(f"📅 **本期帳單週期： {report['period']}**")
    
    c1, c2 = st.columns(2)
    c1.metric("💸 目前累積電費 (已知)", f"NT$ {report['current_bill']:,}", delta="已定案")
    
    # 預測結算
    delta_val = report['predicted_bill'] - report['budget']
    delta_msg = f"超支 {delta_val:,} 元" if delta_val > 0 else f"省下 {abs(delta_val):,} 元"
    delta_color = "inverse" # 紅色代表增加(超支)不好，綠色代表減少(省錢)好
    
    c2.metric("🔮 AI 預估結算 (本期)", f"NT$ {report['predicted_bill']:,}", 
              delta=delta_msg, delta_color=delta_color)

    # 進度條
    usage_percent = report['usage_percent']
    st.write(f"**預算消耗進度 (目標：NT$ {report['budget']:,})**")
    
    if usage_percent > 1.0 or report['status'] == "danger":
        bar_caption = f"⚠️ 警告：預測即將超支！目前預測佔預算 {usage_percent*100:.1f}%"
        bar_color = "red" # Streamlit progress 不支援直接改色，但可用 emoji 提示
    else:
        bar_caption = f"✅ 狀態良好：目前預測佔預算 {usage_percent*100:.1f}%"
    
    st.progress(usage_percent)
    st.caption(bar_caption)
    
    st.divider()

    # ==========================================
    # 區塊 2: 即時用電 (維持 KPI 邏輯)
    # ==========================================
    st.subheader("⚡ 即時用電狀態")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("今日累積用電", f"{kpis['kwh_today_so_far']:.2f} kWh")
    
    # 計算瞬間功率變化
    current_kw = kpis['current_load']
    instant_delta = 0
    # 這裡可以做簡單比較，或者省略
    
    k2.metric("當前功率", f"{current_kw:.3f} kW")
    k3.metric("近 7 天累積", f"{kpis['kwh_last_7_days']:.1f} kWh")
    k4.metric("本期累積用量", f"{kpis['kwh_this_month_so_far']:.1f} kWh")

    st.divider()

    # ==========================================
    # 區塊 3: 滾動預測趨勢圖 (真實數據版)
    # ==========================================
    st.subheader("📈 雙月滾動式修正趨勢")
    
    tab1, tab2 = st.tabs(["預測 vs 真實", "詳細歷史數據"])
    
    with tab1:
        df_hist_plot = df_history.last('60D').copy()
        
        # 過濾掉尾端無效值
        if not df_hist_plot.empty:
            while not df_hist_plot.empty and (df_hist_plot.iloc[-1]['power_kW'] <= 0 or pd.isna(df_hist_plot.iloc[-1]['power_kW'])):
                df_hist_plot = df_hist_plot.iloc[:-1]

        df_hist_plot = df_hist_plot[['power_kW']].reset_index()
        df_hist_plot.columns = ['time', 'value']
        df_hist_plot['type'] = '真實數據 (Actual)'
        
        # 2. 準備預測資料 (從 Session State 拿)
        df_pred_plot = pd.DataFrame()
        if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
            pred_res = st.session_state.prediction_result.copy()
            
            # [視覺優化] 縫合線段
            if not df_hist_plot.empty:
                last_hist_point = pd.DataFrame({
                    'time': [df_hist_plot.iloc[-1]['time']], 
                    'value': [df_hist_plot.iloc[-1]['value']],
                    'type': ['AI 預測 (Forecast)'] 
                })
                future_pred = pred_res[['預測值']].reset_index()
                future_pred.columns = ['time', 'value']
                future_pred['type'] = 'AI 預測 (Forecast)'
                
                df_pred_plot = pd.concat([last_hist_point, future_pred])
            else:
                df_pred_plot = pred_res[['預測值']].reset_index()
                df_pred_plot.columns = ['time', 'value']
                df_pred_plot['type'] = 'AI 預測 (Forecast)'

        # 3. 繪圖
        if not df_pred_plot.empty:
            df_chart = pd.concat([df_hist_plot, df_pred_plot])
            
            last_real_time = df_hist_plot['time'].iloc[-1] if not df_hist_plot.empty else datetime.now()

            fig = px.line(df_chart, x='time', y='value', color='type', 
                          color_discrete_map={'真實數據 (Actual)': '#00CC96', 'AI 預測 (Forecast)': '#EF553B'},
                          line_dash='type',
                          line_dash_map={'真實數據 (Actual)': 'solid', 'AI 預測 (Forecast)': 'dash'},
                          title=f"負載預測 (最後更新: {last_real_time.strftime('%H:%M')})",
                          template="plotly_dark")
            
            fig.add_vline(x=last_real_time.timestamp() * 1000, line_width=1, line_dash="dot", line_color="white")
            fig.add_annotation(x=last_real_time.timestamp() * 1000, y=df_chart['value'].max(), 
                               text="Now", showarrow=True, arrowhead=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            if (datetime.now() - last_real_time).total_seconds() > 3600:
                 st.info(f"ℹ️ 系統備註：目前 **{last_real_time.strftime('%H:%M')}** 之後的數據由 AI 預測模型即時填補。")
        else:
            st.info("無法顯示預測圖表 (無預測數據)。")
        
        with st.expander("ℹ️ 技術原理：Hybrid Model"):
            st.write("""
            本系統結合 **LightGBM** 與 **LSTM**。
            上方橘色虛線即為兩種模型加權後的最終預測結果。
            """)

    with tab2:
        st.dataframe(df_history.tail(100))