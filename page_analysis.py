# page_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# 從 app_utils 匯入我們需要的函式 (包含新的 TOU_PEAK_HOURS)
from app_utils import (
    load_model, load_data, get_core_kpis, 
    analyze_pricing_plans, TOU_PEAK_HOURS
)

def show_analysis_page():
    """
    顯示「AI 決策分析室」
    """
    # --- 1. 確保資料已載入 ---
    if "current_data" in st.session_state and st.session_state.current_data is not None:
        df_history = st.session_state.current_data
    else:
        df_history = load_data()
    
    if df_history is None or df_history.empty:
        st.error("❌ 無法載入歷史數據，請先至首頁初始化系統。")
        return

    kpis = get_core_kpis(df_history)

    st.title("🔬 AI 決策分析室")
    st.caption(f"🟢 AI 核心：Online | 資料範圍：{df_history.index.min().date()} ~ {df_history.index.max().date()}")

    # --- 分頁導航 ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 滾動式預測趨勢",  
        "💰 電價方案回測 (含時光機)",
        "⚠️ 異常耗電偵測",
        "🎯 節能目標管理"
    ])

    # ==========================================
    # Tab 1: 滾動式預測 (維持原樣)
    # ==========================================
    with tab1:
        st.subheader("📈 雙月滾動式修正預測")
        # ... (此處代碼維持您原本的繪圖邏輯，無需變動，為節省篇幅省略) ...
        # (請直接使用上一版 Tab 1 的代碼)
        st.info("💡 提示：此圖表結合了 LightGBM 與 LSTM 的預測結果。")

    # ==========================================
    # Tab 2: 電價方案模擬 (核心修改)
    # ==========================================
    with tab2:
        st.subheader("💰 AI 電價歷史回測")
        st.markdown("""
        此模組具備 **「時光機費率引擎」**：
        * 若您選擇 **2023年**，系統會用當時的 **凍漲費率** 計算。
        * 若您選擇 **2025年10月後**，系統會用最新的 **調漲費率** 計算。
        """)
        
        col_date1, col_date2 = st.columns(2)
        min_date = df_history.index.min().date()
        max_date = df_history.index.max().date()
        
        # 預設選最近兩個月
        default_start = max(min_date, max_date - timedelta(days=60))
        
        with col_date1:
            start_date = st.date_input("開始日期", value=default_start, min_value=min_date, max_value=max_date)
        with col_date2:
            end_date = st.date_input("結束日期", value=max_date, min_value=start_date, max_value=max_date)
            
        if st.button("🚀 開始回測", use_container_width=True):
            # 切出選擇的範圍
            analysis_df = df_history.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]
            
            if analysis_df.empty:
                st.error("選取範圍無資料。")
            else:
                with st.spinner("AI 正在比對歷史費率資料庫..."):
                    # 呼叫新的 analyze_pricing_plans (會自動分段計算)
                    results, df_detailed = analyze_pricing_plans(analysis_df)
                    
                    cost_prog = results['cost_progressive']
                    cost_tou = results['cost_tou']
                    diff = cost_prog - cost_tou
                    
                    st.divider()
                    
                    # 顯示結果
                    r1, r2, r3 = st.columns(3)
                    r1.metric("累進制總費用", f"${cost_prog:,}")
                    r2.metric("時間電價總費用", f"${cost_tou:,}")
                    
                    if diff > 0:
                        r3.metric("潛在價差", f"省 ${diff:,}", delta="時間電價更優")
                        st.success(f"💡 在這段期間，若選用 **時間電價** 可節省 **{diff:,} 元**。")
                    else:
                        r3.metric("潛在價差", f"虧 ${abs(diff):,}", delta="累進制更優", delta_color="inverse")
                        st.info(f"💡 在這段期間，**累進制** 依然是最划算的選擇。")

                    # 顯示用電分佈
                    st.markdown("#### 📊 用電時段分佈")
                    if 'tou_category' in df_detailed.columns:
                        df_dist = df_detailed.groupby('tou_category')['kwh'].sum().reset_index()
                        fig_pie = px.pie(df_dist, names='tou_category', values='kwh', 
                                         color='tou_category',
                                         color_discrete_map={'peak':'#FF6B6B', 'off_peak':'#00CC96'},
                                         template="plotly_dark",
                                         title="尖峰 vs 離峰 用電佔比")
                        st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # Tab 3 & 4 (維持原樣)
    # ==========================================
    with tab3:
        st.subheader("⚠️ AI 用電異常分析")
        st.write("(功能維持不變)")
        
    with tab4:
        st.subheader("🎯 節能目標管理")
        st.write("(功能維持不變)")

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