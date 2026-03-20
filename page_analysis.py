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
    true_current_time = df_history.index[-1]
    try:
        pred_cache = pd.read_csv("prediction_cache.csv")
        pred_cache['datetime'] = pd.to_datetime(pred_cache['datetime'])
        pred_cache = pred_cache.set_index('datetime')
        
        pred_for_bill = pred_cache.copy()
        pred_for_bill['power_kW'] = pred_for_bill['預測值']
        df_combined = pd.concat([df_history, pred_for_bill[['power_kW']]])
    except FileNotFoundError:
        df_combined = df_history

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
    # Tab 1: 用電行為與環境特徵解構 (加入自動文字解讀)
    # ==========================================
    with tab1:
        st.subheader("🧩 用電行為與環境特徵解構")
        st.markdown("AI 藉由深度學習您的**作息規律**與**環境抗性**來進行預測。以下為系統提取的關鍵特徵：")
        
        # --- 1. 準備熱力圖資料 ---
        df_heatmap = df_history.copy()
        df_heatmap['Hour'] = df_heatmap.index.hour
        df_heatmap['DayOfWeek'] = df_heatmap.index.dayofweek
        day_map = {0:'一', 1:'二', 2:'三', 3:'四', 4:'五', 5:'六', 6:'日'}
        df_heatmap['DayName'] = df_heatmap['DayOfWeek'].map(day_map)
        agg_df = df_heatmap.groupby(['DayOfWeek', 'DayName', 'Hour'])['power_kW'].mean().reset_index()
        
        # 🌟 【新增】AI 自動判讀作息
        peak_idx = agg_df['power_kW'].idxmax()
        peak_day = agg_df.loc[peak_idx, 'DayName']
        peak_hour = agg_df.loc[peak_idx, 'Hour']
        peak_val = agg_df.loc[peak_idx, 'power_kW']
        
        st.success(f"💡 **作息特徵洞察**：\n系統發現您的用電最高峰通常落在 **星期{peak_day} 的 {peak_hour}:00 左右** (平均 {peak_val:.2f} kW)。若此時段剛好是台電的高價時段，建議可嘗試將洗衣、烘衣等活動移至其他時間。")
        
        # 繪製熱力圖
        fig_heat = px.density_heatmap(
            agg_df, x='Hour', y='DayName', z='power_kW', histfunc='avg', nbinsx=24,
            category_orders={'DayName': ['一', '二', '三', '四', '五', '六', '日']},
            color_continuous_scale="Inferno", template="plotly_dark",
            labels={'Hour': '時間 (24H)', 'DayName': '星期', 'power_kW': '平均功率 (kW)'}
        )
        fig_heat.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        # --- 2. 氣溫關聯散佈圖 ---
        st.markdown("#### 🌡️ 環境溫度 vs. 耗電量 關聯度")
        
        if 'temperature' in df_history.columns:
            df_scatter = df_history.tail(24 * 30).copy()
            df_scatter['Hour'] = df_scatter.index.hour
            
            # 🌟 【新增】AI 自動判讀氣溫相關性
            corr = df_scatter['temperature'].corr(df_scatter['power_kW'])
            
            if corr > 0.6:
                msg = f"呈 **高度正相關** (相關係數 {corr:.2f})"
                adv = "這代表您的家庭耗電極度依賴冷暖氣設備。建議定期清洗濾網，或將冷氣調高 1 度，節能效果將會非常驚人！"
            elif corr > 0.3:
                msg = f"呈 **中度正相關** (相關係數 {corr:.2f})"
                adv = "氣溫變化對您的耗電有一定影響，但您仍有其他常駐耗電設備。"
            elif corr < -0.3:
                msg = f"呈 **負相關** (相關係數 {corr:.2f})"
                adv = "氣溫越低反而越耗電，系統推測您可能頻繁使用電暖器或高功率熱水設備。"
            else:
                msg = f"**關聯性極低** (相關係數 {corr:.2f})"
                adv = "您的耗電量幾乎不受氣溫影響，代表主要耗電來源可能來自照明、電腦設備或營業用機具。"


            # 🌟 修改說明文字
            st.info(f"💡 **氣候敏感度診斷**：您的用電量與外部氣溫 {msg}。{adv} \n\n*(圖中點的亮色程度代表該時段的實際耗電強度)*")
            
            # 🌟 修改畫圖邏輯：color 改為 'power_kW'
            fig_scatter = px.scatter(
                df_scatter, x='temperature', y='power_kW', color='power_kW', 
                color_continuous_scale="Turbo", template="plotly_dark", opacity=0.7,
                labels={'temperature': '外部氣溫 (°C)', 'power_kW': '實際功率 (kW)'}
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
        
        # 🌟 修改 1：文字改為「本周期」
        st.info("📊 **本周期即時分析** (基於目前累積用量與預測)")
        
        # 🌟 修改 2：改用 df_combined 與 true_current_time
        report = get_billing_report(df_combined, current_time=true_current_time) 
        
        c1, c2, c3 = st.columns(3)
        # 🌟 修改 3：把 current_bill 改為 predicted_bill，這樣才會顯示包含預測的最終總額
        c1.metric("累進制 (預測結算)", f"${report['predicted_bill']:,}")
        c2.metric("時間電價 (預測結算)", f"${report['potential_tou_bill']:,}")
        
        savings = report['savings']
        if savings > 0:
            c3.metric("最佳方案", "時間電價", f"省 ${savings:,}", delta_color="inverse")
            st.success(f"💡 **AI 建議**：{report['recommendation_msg']}")
        else:
            c3.metric("最佳方案", "累進制", f"省 ${abs(savings):,}", delta_color="inverse")
            st.info(f"💡 **AI 建議**：{report['recommendation_msg']}")

        st.divider()
        
        # 歷史回測工具
        st.markdown("#### 🕰️ 歷史帳單回顧")
        st.caption("AI 會自動根據您選擇的年份，套用當年度正確的電價公式 (含尖峰時段調整)。")
        
        col_date1, col_date2 = st.columns(2)
        min_date = df_history.index.min().date()
        max_date = df_history.index.max().date()
        default_start = max(min_date, max_date - timedelta(days=29))
        
        with col_date1:
            start_date = st.date_input("開始日期", value=default_start, min_value=min_date, max_value=max_date)
        with col_date2:
            end_date = st.date_input("結束日期", value=max_date, min_value=start_date, max_value=max_date)
            
        if st.button("🚀 開始回顧", use_container_width=True):
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
    # Tab 3: 異常耗電偵測 (優化：同時段基準法)
    # ==========================================
    with tab3:
        st.subheader("⚠️ AI 用電異常分析")
        st.markdown("系統會比對您過去 30 天內 **「同一個時間點」** 的平均用電習慣，精準抓出不尋常的耗電行為，排除日夜作息的干擾。")
        
        if st.button("🔍 掃描近期異常事件"):
            with st.spinner("正在進行時序特徵比對..."):
                # 只取最近 30 天的資料來分析，避免太久以前的習慣影響判斷
                df_anom = df_history.tail(24 * 30).copy()
                
                # 提取小時特徵
                df_anom['Hour'] = df_anom.index.hour
                
                # 🌟 核心修正：計算「每個小時」專屬的平均值與標準差
                hourly_stats = df_anom.groupby('Hour')['power_kW'].agg(['mean', 'std'])
                
                # 將計算結果合併回原表
                df_anom = df_anom.join(hourly_stats, on='Hour')
                
                # 動態門檻：該時段平均值 + 3倍標準差 (Z-score > 3 視為極端異常)
                df_anom['threshold'] = df_anom['mean'] + 3 * df_anom['std'].fillna(0)
                
                # 篩選出異常點
                anomalies = df_anom[df_anom['power_kW'] > df_anom['threshold']]
                
                if anomalies.empty:
                    st.success("✅ 檢測完畢，近期未發現任何異常耗電行為。")
                else:
                    st.warning(f"⚠️ 偵測到 {len(anomalies)} 筆異常耗電紀錄！(已排除正常日夜峰值)")
                    
                    # 整理顯示表格
                    display_df = anomalies[['power_kW', 'mean', 'threshold']].copy()
                    display_df.columns = ['實際耗電 (kW)', '該時段歷史平均 (kW)', '警報門檻 (kW)']
                    st.dataframe(display_df.style.format("{:.2f}"))
                    
                    fig_anom = px.scatter(anomalies.reset_index(), x='timestamp', y='power_kW', 
                                          title="異常點時間分佈",
                                          color_discrete_sequence=['#FF4B4B'])
                    fig_anom.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_anom, use_container_width=True)

    # ==========================================
    # Tab 4: 節能目標管理
    # ==========================================
    with tab4:
        st.subheader("🎯 節能目標管理")
        
        # 🌟 修改：改用 df_combined 與 true_current_time
        report = get_billing_report(df_combined, current_time=true_current_time)
        current_proj_cost = report['predicted_bill']

        # 🌟 擷取級距資訊
        tier = report.get('current_tier', 1)
        total_kwh = report.get('total_kwh', 0)
        to_next = report.get('kwh_to_next_tier', 0)

        # 🌟 顯示目前的用電量與級距狀態
        st.markdown(f"#### 📊 本期預估總用量：:blue[{total_kwh:.1f} 度] (落於第 {tier} 級距)")

        # 動態預警提示
        if tier < 6:
            if to_next <= 50:
                st.error(f"🚨 警告：距離第 {tier+1} 級距漲價僅剩 **{to_next:.1f} 度**！建議立即採取節能措施。")
            else:
                st.info(f"ℹ️ 距離進入下一級距尚有 {to_next:.1f} 度的安全額度。")
        else:
            st.error("🚨 已達最高級距 (第 6 級)，每度電費將以最高費率計算！")

        st.divider()
        
        target = st.number_input("設定本期電費目標 (元)", value=1000, step=100)
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
