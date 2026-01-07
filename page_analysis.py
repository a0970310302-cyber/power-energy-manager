# page_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# 從 app_utils 匯入我們需要的函式 (包含新的 get_billing_report)
from app_utils import (
    load_model, load_data, get_core_kpis, 
    analyze_pricing_plans, get_billing_report
)

def show_analysis_page():
    """
    顯示「AI 決策分析室」的內容
    核心價值：展示「獨特性 (滾動預測)」與「技術深度」
    """
    # --- 1. 確保資料已載入 ---
    # 優先從 Session State 拿 (減少 IO)，沒有才讀檔
    if "current_data" in st.session_state and st.session_state.current_data is not None:
        df_history = st.session_state.current_data
    else:
        df_history = load_data()
    
    if df_history is None or df_history.empty:
        st.error("❌ 無法載入歷史數據，請先至首頁初始化系統。")
        return

    # 計算基礎 KPI
    kpis = get_core_kpis(df_history)

    # --- 頁面標題 ---
    st.title("🔬 AI 決策分析室")
    st.caption(f"🟢 AI 核心：Online | 最後更新：{kpis['last_updated']}")

    # --- 分頁導航 ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 滾動式預測趨勢",  
        "💰 電價方案模擬",
        "⚠️ 異常耗電偵測",
        "🎯 節能目標管理"
    ])

    # ==========================================
    # Tab 1: 滾動式預測趨勢 (真實 AI 數據版)
    # ==========================================
    with tab1:
        st.subheader("📈 雙月滾動式修正預測")
        st.markdown("""
        此圖表展示系統如何結合 **歷史數據 (實線)** 與 **AI 預測 (虛線)**。
        系統每日凌晨自動將昨天的「預測值」校正為「真實值」，消除累積誤差。
        """)
        
        # 1. 準備歷史數據：過去 7 天 (實線/真實)
        last_timestamp = df_history.index.max()
        start_history = last_timestamp - timedelta(days=7)
        
        df_actual = df_history.loc[start_history:].copy()
        df_actual = df_actual[['power_kW']].reset_index()
        df_actual.columns = ['time', 'value'] 
        df_actual['Type'] = '真實數據 (Actual)'
        
        # 2. 準備預測數據：從 Session State 獲取 (不再造假！)
        df_forecast_plot = pd.DataFrame()
        
        if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
            pred_res = st.session_state.prediction_result.copy()
            
            # 格式化預測數據
            df_forecast = pred_res[['預測值']].reset_index()
            df_forecast.columns = ['time', 'value']
            df_forecast['Type'] = 'AI 預測 (Forecast)'
            
            # 【視覺優化】縫合手術：把歷史數據的最後一點加到預測數據的第一點
            # 這樣畫出來的線才不會中間斷掉
            if not df_actual.empty:
                last_point = df_actual.iloc[[-1]].copy()
                last_point['Type'] = 'AI 預測 (Forecast)' # 改名以便與預測線連在一起
                df_forecast_plot = pd.concat([last_point, df_forecast])
            else:
                df_forecast_plot = df_forecast
        else:
            st.warning("⚠️ 目前沒有預測數據，請回到側邊欄點擊「更新即時數據」。")

        # 3. 合併並繪圖
        if not df_actual.empty:
            df_chart = pd.concat([df_actual, df_forecast_plot])
            
            # 使用 Plotly 繪製
            fig = px.line(df_chart, x='time', y='value', color='Type',
                          line_dash='Type', 
                          line_dash_map={'真實數據 (Actual)': 'solid', 'AI 預測 (Forecast)': 'dash'},
                          color_discrete_map={'真實數據 (Actual)': '#00CC96', 'AI 預測 (Forecast)': '#EF553B'},
                          template="plotly_dark")
            
            # 標記 "Now" 的時間點
            fig.add_vline(x=last_timestamp.timestamp() * 1000, line_width=2, line_dash="dot", line_color="white")
            fig.add_annotation(x=last_timestamp.timestamp() * 1000, y=df_chart['value'].max(), 
                               text="Now (修正點)", showarrow=True, arrowhead=1, ax=40, ay=0)
            
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=20, b=20),
                height=450,
                xaxis_title="時間",
                yaxis_title="功率 (kW)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示模型權重 (如果有)
            if "weights" in st.session_state: # 假設你有存這個，或者只是純文字說明
                pass 
                
        with st.expander("ℹ️ 技術解密：混合模型架構 (Hybrid Model)", expanded=False):
            st.markdown("""
            本系統採用 **Ensemble Learning** 技術，不再是隨機生成的假曲線：
            * **LightGBM**：負責捕捉氣候特徵 (溫度、濕度) 與假日效應。
            * **LSTM (深度學習)**：負責記憶長短期的用電慣性 (Sequence Memory)。
            * **最終輸出**：上述兩者加權平均後的最佳解。
            """)

    # ==========================================
    # Tab 2: 電價方案模擬 (整合 get_billing_report)
    # ==========================================
    with tab2:
        st.subheader("💰 AI 電價分析器")
        
        # 1. 先顯示本月目前的 AI 建議 (Unified Logic)
        st.info("📊 **本月即時分析** (基於目前累積用量與預測)")
        report = get_billing_report(df_history) # 呼叫全能計費中心
        
        c1, c2, c3 = st.columns(3)
        c1.metric("累進制 (現況)", f"${report['current_bill']:,}")
        c2.metric("時間電價 (試算)", f"${report['potential_tou_bill']:,}")
        
        savings = report['savings']
        if savings > 0:
            c3.metric("最佳方案", "時間電價", f"省 ${savings:,}", delta_color="inverse")
            st.success(f"💡 **AI 建議**：{report['recommendation_msg']}")
        else:
            c3.metric("最佳方案", "累進制", f"省 ${abs(savings):,}", delta_color="inverse")
            st.info(f"💡 **AI 建議**：{report['recommendation_msg']}")

        st.divider()
        
        # 2. 歷史回測工具 (保留原本功能，讓使用者自己選日期)
        st.markdown("#### 🕰️ 歷史帳單回測")
        st.caption("您可以自訂日期範圍，分析過去若使用不同費率的差異。")
        
        col_date1, col_date2 = st.columns(2)
        min_date = df_history.index.min().date()
        max_date = df_history.index.max().date()
        # 預設選上個月
        default_start = max(min_date, max_date - timedelta(days=29))
        
        with col_date1:
            start_date = st.date_input("開始日期", value=default_start, min_value=min_date, max_value=max_date)
        with col_date2:
            end_date = st.date_input("結束日期", value=max_date, min_value=start_date, max_value=max_date)
            
        if st.button("🚀 開始回測", use_container_width=True):
            analysis_df = df_history.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]
            
            if analysis_df.empty:
                st.error("選取範圍無資料。")
            else:
                with st.spinner("AI 正在精算每一度電的成本..."):
                    # 這裡呼叫底層引擎，因為是自訂範圍
                    results = analyze_pricing_plans(analysis_df)
                    cost_prog = results['cost_progressive'].sum()
                    cost_tou = results['cost_tou'].sum()
                    diff = cost_prog - cost_tou
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("區間累進費用", f"${int(cost_prog):,}")
                    r2.metric("區間時間電價", f"${int(cost_tou):,}")
                    r3.metric("潛在價差", f"${int(diff):,}", delta="正值代表時間電價較省" if diff>0 else "負值代表累進較省")

                    # 畫派餅圖
                    st.markdown("#### 📊 用電時段分佈")
                    df_dist = results.groupby('tou_category')['kwh'].sum().reset_index()
                    fig_pie = px.pie(df_dist, names='tou_category', values='kwh', 
                                     color='tou_category',
                                     color_discrete_map={'peak':'#FF6B6B', 'off_peak':'#00CC96'},
                                     template="plotly_dark",
                                     title="尖峰 vs 離峰 用電佔比")
                    st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # Tab 3: 異常耗電偵測
    # ==========================================
    with tab3:
        st.subheader("⚠️ AI 用電異常分析")
        st.markdown("利用統計模型偵測歷史數據中的**異常高耗電**事件。")
        
        if st.button("🔍 掃描異常事件"):
            with st.spinner("正在掃描歷史數據..."):
                df_anom = df_history.copy()
                window = 96 * 7 # 一週
                df_anom['mean'] = df_anom['power_kW'].rolling(window=window, min_periods=1).mean()
                df_anom['std'] = df_anom['power_kW'].rolling(window=window, min_periods=1).std()
                df_anom['threshold'] = df_anom['mean'] + 2.5 * df_anom['std']
                
                anomalies = df_anom[df_anom['power_kW'] > df_anom['threshold']]
                
                if anomalies.empty:
                    st.success("✅ 檢測完畢，未發現顯著異常。")
                else:
                    st.warning(f"⚠️ 偵測到 {len(anomalies)} 筆異常高耗電紀錄！")
                    st.dataframe(anomalies[['power_kW', 'mean', 'threshold']].style.format("{:.2f}"))
                    
                    fig_anom = px.scatter(anomalies.reset_index(), x='timestamp', y='power_kW', 
                                          title="異常點時間分佈",
                                          color_discrete_sequence=['red'])
                    st.plotly_chart(fig_anom, use_container_width=True)

    # ==========================================
    # Tab 4: 節能目標管理 (整合 Unified Logic)
    # ==========================================
    with tab4:
        st.subheader("🎯 節能目標管理")
        
        # 使用統一報告中的「預測帳單」
        report = get_billing_report(df_history)
        current_proj_cost = report['predicted_bill']
        
        target = st.number_input("設定本月電費目標 (元)", value=3000, step=100)
        
        col_t1, col_t2 = st.columns(2)
        col_t1.metric("本月目標", f"${target:,}")
        
        delta = target - current_proj_cost
        if delta >= 0:
             col_t2.metric("AI 預測結算", f"${current_proj_cost:,}", delta=f"安全 (剩餘 ${delta:,})")
             st.success("🎉 目前控制良好，請繼續保持！")
             st.progress(min(current_proj_cost / target, 1.0))
        else:
             col_t2.metric("AI 預測結算", f"${current_proj_cost:,}", delta=f"超支 ${abs(delta):,}", delta_color="inverse")
             st.error(f"⚠️ 警告：依目前趨勢，月底將超支 {abs(delta):,} 元！")
             st.progress(1.0) # 全滿紅條
             
             st.markdown("**💡 AI 建議行動：**")
             st.markdown("- [ ] 檢查冷氣溫度是否過低 (建議 26~28°C)")
             st.markdown("- [ ] 離峰時間再使用高耗電家電 (洗衣機、烘衣機)")