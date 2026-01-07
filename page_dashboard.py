# page_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np 

# 匯入共用函式
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
    # 區塊 3: 滾動預測趨勢圖 (升級版：分段式信心區間)
    # ==========================================
    st.subheader("📈 雙月滾動式修正趨勢 (全週期監控)")
    
    tab1, tab2 = st.tabs(["預測 vs 真實", "詳細歷史數據"])
    
    with tab1:
        # 1. 取得當期帳單週期
        latest_time = df_history.index[-1]
        cycle_start, cycle_end = get_current_bill_cycle(latest_time)
        
        # 2. 準備歷史資料 (鎖定本週期)
        df_hist_plot = df_history[(df_history.index >= cycle_start) & (df_history.index <= cycle_end)].copy()
        
        # 清理無效值
        if not df_hist_plot.empty:
            while not df_hist_plot.empty and (df_hist_plot.iloc[-1]['power_kW'] <= 0):
                df_hist_plot = df_hist_plot.iloc[:-1]

        # 轉為繪圖格式
        plot_data = []
        
        # A. 加入歷史數據
        if not df_hist_plot.empty:
            hist_data = df_hist_plot[['power_kW']].reset_index()
            hist_data.columns = ['time', 'value']
            hist_data['type'] = '歷史實績 (Actual)'
            plot_data.append(hist_data)
            
            # 取得最後一個歷史點，作為預測線的起點 (確保線條連續)
            last_hist_point = hist_data.iloc[[-1]].copy()
        else:
            last_hist_point = None

        # B. 準備預測資料 (進行分流：短期 vs 長期)
        if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
            pred_res = st.session_state.prediction_result.copy()
            # 篩選本週期
            pred_res = pred_res[(pred_res.index >= cycle_start) & (pred_res.index <= cycle_end)]
            
            if not pred_res.empty:
                # 定義短期界線：未來 48 小時
                short_term_end = latest_time + timedelta(hours=48)
                
                # --- B1. 短期預測 (高信心區) ---
                pred_short = pred_res[pred_res.index <= short_term_end]
                if not pred_short.empty:
                    short_data = pred_short[['預測值']].reset_index()
                    short_data.columns = ['time', 'value']
                    short_data['type'] = 'AI 短期預測 (48h)'
                    
                    # 縫合歷史與短期
                    if last_hist_point is not None:
                        # 將上一段的終點，改成這一段的型別，加到這一段的開頭
                        connector = last_hist_point.copy()
                        connector['type'] = 'AI 短期預測 (48h)'
                        short_data = pd.concat([connector, short_data])
                    
                    plot_data.append(short_data)
                    # 更新接點
                    last_short_point = short_data.iloc[[-1]].copy()
                else:
                    last_short_point = last_hist_point

                # --- B2. 長期推估 (趨勢參考區) ---
                pred_long = pred_res[pred_res.index > short_term_end]
                if not pred_long.empty:
                    long_data = pred_long[['預測值']].reset_index()
                    long_data.columns = ['time', 'value']
                    long_data['type'] = '長期趨勢推估 (Trend)'
                    
                    # 縫合短期與長期
                    if last_short_point is not None:
                        connector = last_short_point.copy()
                        connector['type'] = '長期趨勢推估 (Trend)'
                        long_data = pd.concat([connector, long_data])
                        
                    plot_data.append(long_data)

        # 3. 繪圖
        if plot_data:
            df_chart = pd.concat(plot_data)
            
            # 定義顏色與線條樣式
            color_map = {
                '歷史實績 (Actual)': '#00CC96',       # 綠色
                'AI 短期預測 (48h)': '#EF553B',       # 深紅色
                '長期趨勢推估 (Trend)': '#FFA15A'     # 橘黃色 (較柔和)
            }
            dash_map = {
                '歷史實績 (Actual)': 'solid',
                'AI 短期預測 (48h)': 'dot',           # 點線 (強調預測性質)
                '長期趨勢推估 (Trend)': 'dash'        # 虛線 (強調不確定性)
            }

            fig = px.line(df_chart, x='time', y='value', color='type', 
                          color_discrete_map=color_map,
                          line_dash='type',
                          line_dash_map=dash_map,
                          title=f"帳單週期全程監控 ({cycle_start.strftime('%m/%d')} ~ {cycle_end.strftime('%m/%d')})",
                          template="plotly_dark")
            
            # 強制鎖定 X 軸範圍 (實現雙月全景)
            fig.update_xaxes(range=[cycle_start, cycle_end])
            
            # 標示 "Now"
            fig.add_vline(x=latest_time.timestamp() * 1000, line_width=1, line_dash="solid", line_color="white")
            
            # 在圖表上方加入標註
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            ℹ️ **圖表說明**：
            * **綠線**：已發生的真實用電。
            * **紅點線**：AI 針對未來 48 小時的高精度預測。
            * **橘虛線**：依據您的用電慣性與歷史氣溫，推估至結算日 ({cycle_end.strftime('%m/%d')}) 的參考走勢。
            """)
            
        else:
            st.info("尚無本期數據。")
        
        with st.expander("ℹ️ 技術原理：Hybrid Model"):
            st.write("""
            本系統結合 **LightGBM** 與 **LSTM**。
            短期預測採用即時特徵運算，長期推估則引入 **WeatherSimulator** 進行氣候模擬。
            """)

    with tab2:
        st.dataframe(df_history.tail(100))