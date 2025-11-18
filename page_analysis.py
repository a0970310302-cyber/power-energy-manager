import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# 從 app_utils 匯入我們需要的函式
from app_utils import (
    load_model, load_data, get_core_kpis, 
    analyze_pricing_plans, TOU_RATES_DATA
)

# 從 model_trainer 匯入特徵工程函式
try:
    from model_trainer import create_features
except ImportError:
    st.error("錯誤：找不到 model_trainer.py。")
    def create_features(df):
        return df # 返回一個空值

def show_analysis_page():
    """
    顯示「AI 決策分析室」的內容
    """
    # --- 載入數據並計算 KPI (為了 Tab 4) ---
    model = load_model()
    df_history = load_data()
    kpis = get_core_kpis(df_history)

    # --- AI 決策分析室頁面內容 ---
    st.header("🔬 AI 決策分析室")
    st.info("利用 AI 模型預測未來用電，並分析您的最佳電價方案。")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI 用電預測",  
        "💰 AI 電價分析器",
        "⚠️ AI 用電異常分析",
        "🎯 AI 節能建議"
        ])

    # --- AI 預測分頁 ---
    with tab1:
        st.subheader("🤖 AI 用電預測")
        
        if model is None or df_history.empty:
            st.error("模型或歷史資料載入失敗，無法進行預測。")
        else:
            default_future_date = df_history.index.max().date() + timedelta(days=1)
            future_date = st.date_input(
                "請選擇您要預測的日期：",
                value=default_future_date,
                min_value=df_history.index.min().date() + timedelta(days=1),
                max_value=df_history.index.max().date() + timedelta(days=30),
                help="AI 將根據歷史數據，預測您所選日期當天的 15 分鐘用電曲線。"
            )

            if st.button("📈 開始預測"):
                with st.spinner("AI 正在為您計算... (這可能需要幾秒鐘)"):
                    try:
                        future_timestamps = pd.date_range(start=future_date, periods=96, freq='15T')
                        df_future = pd.DataFrame(index=future_timestamps)
                        
                        lag_date = future_date - timedelta(days=1)
                        lag_data_time = future_timestamps - timedelta(days=1)
                        
                        try:
                            lag_df = df_history.loc[lag_data_time]
                            lag_df = lag_df.set_index(future_timestamps)
                            df_future['lag_1_day'] = lag_df['power_kW']
                        except KeyError:
                            st.error(f"錯誤：找不到 {lag_date.strftime('%Y-%m-%d')} 的完整歷史資料，無法產生『昨日同期』特徵。")
                            df_future['lag_1_day'] = 0  
                            st.warning("已使用 0 填充 'lag_1_day' 特徵。")
                        except Exception as e:
                            st.error(f"提取 Lag 特徵時發生未知錯誤：{e}")
                            raise  

                        df_future_with_feats = create_features(df_future)
                        FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'is_weekend', 'lag_1_day']
                        
                        missing_features = [f for f in FEATURES if f not in df_future_with_feats.columns]
                        if missing_features:
                            raise ValueError(f"即時特徵工程中缺少以下特徵：{missing_features}")

                        X_future = df_future_with_feats[FEATURES]
                        prediction = model.predict(X_future)
                        df_pred = pd.DataFrame(prediction, index=future_timestamps, columns=['預測用電 (kW)'])
                        
                        st.subheader(f"📅 {future_date.strftime('%Y-%m-%d')} 預測結果")
                        
                        total_kwh = df_pred['預測用電 (kW)'].sum() * 0.25  
                        peak_power = df_pred['預測用電 (kW)'].max()
                        peak_time = df_pred['預測用電 (kW)'].idxmax().strftime('%H:%M')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("預測總度數 (kWh)", f"{total_kwh:.2f} 度")
                        with col2:
                            st.metric("預測用電高峰", f"{peak_power:.3f} kW", f"發生在 {peak_time}")
                        
                        fig_pred = px.line(df_pred, y='預測用電 (kW)', template="plotly_dark", color_discrete_sequence=['#FF6B6B'])
                        fig_pred.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        with st.expander("📖 顯示預測的 15 分鐘原始數據"):
                            st.dataframe(df_pred.style.format("{:.3f} kW"))
                    
                    except ValueError as ve:
                        st.error(f"執行 AI 預測時發生錯誤：{ve}")
                    except Exception as e:
                        st.error(f"執行 AI 預測時發生未知錯誤：{e}")

    # --- AI 電價分析器分頁 ---
    with tab2:
        st.subheader("💰 AI 電價分析器 (依據2024/4/1電價)")
        
        if df_history.empty:
            st.warning("無歷史資料可供分析。")
        else:
            st.markdown("此功能將回測您的歷史用電數據，比較 **「累進電價」** 與 **「簡易型時間電價 (TOU)」** 的總成本。")
            
            with st.expander("點此查看電價方案詳情"):
                st.markdown("##### 方案一：累進電價 (一般住宅預設)")
                st.markdown("""
                | 每月用電度數 (kWh) | 夏月 (6-9月) | 非夏月 |
                | :--- | :---: | :---: |
                | 120 度以下 | 1.68 元 | 1.68 元 |
                | 121~330 度 | 2.45 元 | 2.16 元 |
                | 331~500 度 | 3.70 元 | 3.03 元 |
                | 501~700 度 | 5.04 元 | 4.14 元 |
                | 701~1000 度 | 6.24 元 | 5.07 元 |
                | 1001 度以上 | 8.46 元 | 6.63 元 |
                """)
                
                st.markdown("##### 方案二：簡易型時間電價 (TOU) - 二段式")
                st.markdown(f"- **基本電費：** 每月 `{TOU_RATES_DATA['basic_fee_monthly']}` 元")
                st.markdown(f"- **夏月 (6/1-9/30)**")
                st.markdown(f"  - **尖峰 (週一至五 09:00-24:00)：** `{TOU_RATES_DATA['rates']['summer']['peak']}` 元/度")
                st.markdown(f"  - **離峰 (尖峰以外 + 假日)：** `{TOU_RATES_DATA['rates']['summer']['off_peak']}` 元/度")
                st.markdown(f"- **非夏月**")
                st.markdown(f"  - **尖峰 (週一至五 06:00-11:00, 14:00-24:00)：** `{TOU_RATES_DATA['rates']['nonsummer']['peak']}` 元/度")
                st.markdown(f"  - **離峰 (尖峰以外 + 假日)：** `{TOU_RATES_DATA['rates']['nonsummer']['off_peak']}` 元/度")
                st.markdown(f"*注意：每月總用電量超過 {TOU_RATES_DATA['surcharge_kwh_threshold']} 度，超過部分每度加收 {TOU_RATES_DATA['surcharge_rate_per_kwh']} 元。*")

            st.markdown("---")
            st.markdown("##### 選擇您要分析的歷史資料範圍")
            min_date = df_history.index.min().date()
            max_date = df_history.index.max().date()
            default_start_date = max(min_date, max_date - timedelta(days=29))  

            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("分析開始日期", value=default_start_date, min_value=min_date, max_value=max_date, key="analysis_start")
            with col_date2:
                end_date = st.date_input("分析結束日期", value=max_date, min_value=start_date, max_value=max_date, key="analysis_end")
            
            analysis_df = df_history.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].copy()

            if st.button("💰 開始分析電價"):
                if analysis_df.empty:
                    st.error("選定範圍內無資料，請重新選擇日期。")
                else:
                    with st.spinner("AI 正在回測您的歷史用電..."):
                        try:
                            results, df_detailed = analyze_pricing_plans(analysis_df)
                            
                            cost_prog = results['cost_progressive']
                            cost_tou = results['cost_tou']
                            total_kwh = results['total_kwh']

                            st.subheader(f"📅 {start_date} 至 {end_date} 電價分析結果")
                            st.markdown(f"期間總用電量： **{total_kwh:,.2f} kWh**")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("方案一：累進電價 (標準)", f"{cost_prog:,.0f} 元")
                            col2.metric("方案二：簡易型時間電價 (TOU)", f"{cost_tou:,.0f} 元")
                            
                            st.divider()
                            
                            difference = cost_prog - cost_tou
                            if difference > 0:
                                best_plan = "簡易型時間電價 (TOU)"
                                savings = difference
                                st.success(f"**分析建議：:green[(๑•̀ㅂ•́)و✧]**")
                                st.success(f"在此期間，若選用 **{best_plan}**，預計可**節省 {savings:,.0f} 元**！")
                                st.info("您的用電模式可能在離峰時段佔比較高。")
                            else:
                                best_plan = "累進電價 (標準)"
                                savings = abs(difference)
                                st.warning(f"**分析建議：:red[(｡ ́︿ ̀｡)]**")
                                st.warning(f"在此期間，選用 **{best_plan}** 較為划算 (可省 {savings:,.0f} 元)。")
                                st.info(f"若要改用時間電價，建議您將尖峰用電轉移至離峰時段。")
                                
                            st.markdown("---")
                            st.subheader("TOU 用電分佈 (kWh)")
                            
                            df_kwh_dist = df_detailed.groupby('tou_category')['kwh'].sum().reset_index()
                            
                            fig_pie_kwh = px.pie(df_kwh_dist, names='tou_category', values='kwh', 
                                                title='TOU 時段用電量 (kWh) 分佈',
                                                color_discrete_map={'peak':'#FF6B6B', 'off_peak':'#4ECDC4'},
                                                template="plotly_dark")
                            st.plotly_chart(fig_pie_kwh, use_container_width=True)
                            
                            st.subheader("TOU 成本分佈 (時間電價)")
                            df_cost_dist = df_detailed.groupby('tou_category')['tou_flow_cost'].sum().reset_index()
                            
                            fig_pie_cost = px.pie(df_cost_dist, names='tou_category', values='tou_flow_cost', 
                                                title='TOU 時段電費 (元) 分佈',
                                                color_discrete_map={'peak':'#FF6B6B', 'off_peak':'#4ECDC4'},
                                                template="plotly_dark")
                            st.plotly_chart(fig_pie_cost, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"執行電價分析時發生錯誤: {e}")
                            st.error("請檢查您的資料範圍是否完整。")

    # --- 異常分析分頁 ---
    with tab3:
        st.subheader("⚠️ AI 用電異常分析")
        
        if df_history.empty:
            st.warning("無歷史資料可供分析。")
        else:
            st.markdown("此功能將分析您的完整歷史數據，找出用電量顯著高於平時的時段。")
            
            with st.spinner("AI 正在分析您的歷史數據..."):
                try:
                    df_analysis_anomaly = df_history.copy()
                    window_size = 96 * 7
                    df_analysis_anomaly['rolling_avg'] = df_analysis_anomaly['power_kW'].rolling(window=window_size, center=True, min_periods=96).mean()
                    df_analysis_anomaly['rolling_std'] = df_analysis_anomaly['power_kW'].rolling(window=window_size, center=True, min_periods=96).std()
                    df_analysis_anomaly['anomaly_threshold'] = df_analysis_anomaly['rolling_avg'] + (2 * df_analysis_anomaly['rolling_std'])
                    
                    anomalies = df_analysis_anomaly[df_analysis_anomaly['power_kW'] > df_analysis_anomaly['anomaly_threshold']]

                    if anomalies.empty:
                        st.success("🎉 分析完畢：在您的歷史數據中未發現明顯的用電異常事件。")
                    else:
                        st.warning(f"偵測到 {len(anomalies)} 筆 (15分鐘) 異常用電事件！")
                        st.markdown("---")
                        st.markdown("#### 異常用電時段 vs 歷史平均 (最近 30 天)")
                        
                        chart_data = df_analysis_anomaly.last('30D')[[
                            'power_kW', 'rolling_avg', 'anomaly_threshold'
                        ]]
                        chart_data.columns = ['實際用電', '7日平均', '異常閾值']
                        
                        fig_anomaly = px.line(chart_data, template="plotly_dark")
                        fig_anomaly.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_anomaly, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("#### 異常事件詳細列表")
                        
                        with st.expander("📖 顯示異常事件的 15 分鐘原始數據"):
                            st.dataframe(anomalies[['power_kW', 'rolling_avg', 'anomaly_threshold']])

                except Exception as e:
                    st.error(f"執行異常分析時發生錯誤：{e}")

    # --- AI 節能建議分頁 ---
    with tab4:
        st.subheader("🎯 AI 節能建議")
        
        # 這裡的 'cost_target' 會從 st.session_state 讀取
        target_cost = st.session_state.get('cost_target', 1000) 
        st.info(f"您在主頁設定的本月電費目標為： **{target_cost} 元**")
        
        if df_history.empty:
            st.warning("無歷史資料，無法進行節能建議。")
        else:
            with st.spinner("AI 正在分析您的節能潛力..."):
                try:
                    difference = kpis['projected_cost'] - target_cost
                    st.markdown("---")
                    
                    if difference > 0:
                        st.error(f"**警示：:red[(｡ ́︿ ̀｡)]**")
                        st.error(f"以您過去 30 天的用電模式估算，本月電費約為 **{kpis['projected_cost']:.0f} 元** (依累進電價計算)，將**超過**您的目標 **{difference:.0f} 元**。")
                        
                        st.markdown("#### 💡 AI 節能建議：")
                        daily_kwh_reduction_needed = (difference / kpis['PRICE_PER_KWH_AVG']) / 30
                        st.markdown(f"* 您需要**每日平均減少 {daily_kwh_reduction_needed:.2f} 度 (kWh)** 的用電量才能達標。")
                        st.markdown(f"* **建議您：**")
                        st.markdown(f"    1.  前往「**AI 電價分析器**」分頁，確認您是否使用了最划算的電價方案。")
                        st.markdown(f"    2.  前往「**AI 用Dian異常分析**」分頁，找出您的異常高耗電時段。")
                        
                    else:
                        st.success(f"**恭喜！:green[(๑•̀ㅂ•́)و✧]**")
                        st.success(f"以您過去 30 天的用電模式估算，本月電費約為 **{kpis['projected_cost']:.0f} 元** (依累進電價計算)，**低於**您的 **{target_cost} 元** 目標。")
                        st.markdown("#### 💡 AI 節能建議：")
                        st.markdown("* 您的用電習慣非常良好！")
                        st.markdown("* 可以前往「**AI 電價分析器**」分頁，看看是否有機會省下更多錢！")

                except Exception as e:
                    st.error(f"執行節能建議分析時發生錯誤：{e}")