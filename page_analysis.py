# page_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# 從 app_utils 匯入我們需要的函式
# 注意：已移除對 TOU_RATES_DATA 的依賴，改由 analyze_pricing_plans 自動處理
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
        "🧩 用電模式特徵",  
        "💰 電價方案模擬",
        "⚠️ 異常耗電偵測",
        "🎯 節能目標管理"
    ])


        # ==========================================
    # Tab 1: 用電行為與環境特徵分析 (全新改造)
    # ==========================================
    with tab1:
        st.subheader("🧩 用電行為與環境特徵解構")
        st.markdown("""
        AI 精準預測的背後，來自於對使用者**生活作息**與**環境抗性**的深度理解。
        此區塊為您透視系統提取的關鍵特徵模式。
        """)
        
        # --- 1. 用電熱力圖 ---
        st.markdown("#### 🕒 週期作息熱力圖 (24h x 7Days)")
        st.caption("顏色越亮代表平均耗電量越高。這有助於直觀判斷您的「用電尖峰」是否與台電的昂貴時段重疊。")
        
        # 準備熱力圖資料
        df_heatmap = df_history.copy()
        df_heatmap['Hour'] = df_heatmap.index.hour
        df_heatmap['DayOfWeek'] = df_heatmap.index.dayofweek
        
        # 將星期數字轉為中文，並強制排序
        day_map = {0:'一', 1:'二', 2:'三', 3:'四', 4:'五', 5:'六', 6:'日'}
        df_heatmap['DayName'] = df_heatmap['DayOfWeek'].map(day_map)
        
        # 計算每個星期幾的每個小時的平均用電
        agg_df = df_heatmap.groupby(['DayOfWeek', 'DayName', 'Hour'])['power_kW'].mean().reset_index()
        
        fig_heat = px.density_heatmap(
            agg_df, x='Hour', y='DayName', z='power_kW',
            histfunc='avg', nbinsx=24,
            category_orders={'DayName': ['一', '二', '三', '四', '五', '六', '日']},
            color_continuous_scale="Inferno", # 使用火焰色系凸顯高耗電
            template="plotly_dark",
            labels={'Hour': '時間 (24H)', 'DayName': '星期', 'power_kW': '平均功率 (kW)'}
        )
        fig_heat.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        # --- 2. 氣溫關聯散佈圖 ---
        st.markdown("#### 🌡️ 環境溫度 vs. 耗電量 關聯度")
        st.caption("展示氣象因子對家庭負載的影響。通常在特定溫度以上，因空調啟動會出現明顯的「耗電陡升」現象。")
        
        if 'temperature' in df_history.columns:
            # 取最近一個月的資料來畫散佈圖，避免點太密集
            df_scatter = df_history.tail(24 * 30).copy()
            df_scatter['Hour'] = df_scatter.index.hour
            
            fig_scatter = px.scatter(
                df_scatter, x='temperature', y='power_kW', 
                color='Hour', # 用顏色區分白天或黑夜
                color_continuous_scale="Turbo",
                template="plotly_dark",
                opacity=0.7,
                labels={'temperature': '外部氣溫 (°C)', 'power_kW': '實際功率 (kW)', 'Hour': '發生時間'}
            )
            fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("ℹ️ 目前資料庫中尚未檢測到完整的氣溫 (temperature) 特徵，無法繪製環境關聯圖。")

    # ==========================================
    # Tab 2: 電價方案模擬 (整合多年度費率)
    # ==========================================
    with tab2:
        st.subheader("💰 AI 電價分析器 (支援 2022-2025 歷史費率)")
        
        st.info("📊 **本月即時分析** (基於目前累積用量與預測)")
        report = get_billing_report(df_history) 
        
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
        
        # 歷史回測工具
        st.markdown("#### 🕰️ 歷史帳單回測")
        st.caption("AI 會自動根據您選擇的年份，套用當年度正確的電價公式 (含尖峰時段調整)。")
        
        col_date1, col_date2 = st.columns(2)
        min_date = df_history.index.min().date()
        max_date = df_history.index.max().date()
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
                with st.spinner("AI 正在比對歷史費率資料庫..."):
                    # 呼叫新的 analyze_pricing_plans，它會自動查表
                    results, df_detailed = analyze_pricing_plans(analysis_df)
                    cost_prog = results['cost_progressive']
                    cost_tou = results['cost_tou']
                    diff = cost_prog - cost_tou
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("區間累進費用", f"${int(cost_prog):,}")
                    r2.metric("區間時間電價", f"${int(cost_tou):,}")
                    r3.metric("潛在價差", f"${int(diff):,}", delta="正值代表時間電價較省" if diff>0 else "負值代表累進較省")

                    # 顯示使用的費率版本
                    mid_date = start_date + (end_date - start_date)/2
                    year_ver = "2024~2025 (最新費率)"
                    if mid_date < datetime(2022, 7, 1).date(): year_ver = "2022H1 (凍漲舊費率)"
                    elif mid_date < datetime(2023, 4, 1).date(): year_ver = "2022H2 (大戶調漲費率)"
                    elif mid_date < datetime(2024, 4, 1).date(): year_ver = "2023 (新時段費率)"
                    elif mid_date >= datetime(2025, 10, 16).date(): year_ver = "2025 (114年新制)"
                    
                    st.caption(f"ℹ️ 計算基準：使用 {year_ver} 標準")

                    st.markdown("#### 📊 用電時段分佈")
                    df_dist = df_detailed.groupby('tou_category')['kwh'].sum().reset_index()
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
        if st.button("🔍 掃描異常事件"):
            with st.spinner("正在掃描歷史數據..."):
                df_anom = df_history.copy()
                window = 96 * 7 
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
    # Tab 4: 節能目標管理
    # ==========================================
    with tab4:
        st.subheader("🎯 節能目標管理")
        report = get_billing_report(df_history)
        current_proj_cost = report['predicted_bill']
        
        target = st.number_input("設定本月電費目標 (元)", value=1000, step=100)
        
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
             st.progress(1.0)
             
             st.markdown("**💡 AI 建議行動：**")
             st.markdown("- [ ] 檢查冷氣溫度是否過低 (建議 26~28°C)")
             st.markdown("- [ ] 離峰時間再使用高耗電家電 (洗衣機、烘衣機)")
