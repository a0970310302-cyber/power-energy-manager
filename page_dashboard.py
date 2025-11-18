import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 從 app_utils 匯入我們需要的函式
from app_utils import load_data, get_core_kpis

def show_dashboard_page():
    """
    顯示「用電儀表板」的內容
    """
    # --- 載入數據並計算 KPI ---
    df_history = load_data()
    kpis = get_core_kpis(df_history)

    # --- 儀表板頁面內容 ---
    st.title("💡 智慧電能管家")
    st.header("📈 用電儀表板")

    if df_history.empty or not kpis['status_data_available']:
        st.warning("儀表板無資料可顯示，或歷史資料不足 14 天。")
    else:
        # --- 本週用電狀態 ---
        if kpis['weekly_delta_percent'] > 10: status_display = f":red[(｡ ́︿ ̀｡) 警示]"
        elif kpis['weekly_delta_percent'] < -10: status_display = ":green[(๑•̀ㅂ•́)و✧ 良好]"
        else: status_display = ":blue[(・-・) 普通]"
        st.subheader(f"您的用電狀態： {status_display}")
        
        # --- KPI 控制中心 ---
        st.markdown("### 關鍵指標 (KPI) 控制中心")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("今日累積用電", f"{kpis['kwh_today_so_far']:.2f} kWh")
        col2.metric("今日預估電費", f"{kpis['cost_today_so_far']:.0f} 元")
        col3.metric("本週累積用電 (近 7 天)", f"{kpis['kwh_last_7_days']:.2f} kWh")
        col4.metric("本月累積用電 (至今)", f"{kpis['kwh_this_month_so_far']:.1f} kWh")
        
        col5, col6 = st.columns(2)
        latest_data = kpis['latest_data']
        latest_power = latest_data['power_kW']
        yesterday_time = latest_data.name - timedelta(days=1)
        instant_delta_text, instant_delta_color, yesterday_power_display = "N/A", "off", "N/A"
        
        if yesterday_time in df_history.index:
            yesterday_data = df_history.loc[yesterday_time]
            yesterday_power = yesterday_data['power_kW']
            yesterday_power_display = f"{yesterday_power:.3f} kW"
            if yesterday_power > 0:
                instant_delta = ((latest_power - yesterday_power) / yesterday_power) * 100
                if instant_delta > 10: instant_delta_text = f"高於昨日 {instant_delta:.1f}%"; instant_delta_color = "inverse"
                elif instant_delta < -10: instant_delta_text = f"低於昨日 {abs(instant_delta):.1f}%"; instant_delta_color = "normal"
                else: instant_delta_text = f"{instant_delta:+.1f}%"; instant_delta_color = "normal"
            else: instant_delta_text = "昨日無耗電"
        else: instant_delta_text = "無昨日資料"
        
        col5.metric(label=f"最新用電功率 ({latest_data.name.strftime('%H:%M')})", value=f"{latest_power:.3f} kW")
        col6.metric(label=f"昨日同期 ({yesterday_time.strftime('%H:%M')})", value=yesterday_power_display, delta=instant_delta_text, delta_color=instant_delta_color)
        
        st.divider() 

        # --- 圖表 Tabs ---
        st.subheader("用電趨勢分析")
        tab1, tab2, tab3 = st.tabs(["📈 最近 7 天趨勢", "🍩 近 30 天尖離峰", "📊 每日歷史數據"])

        with tab1:
            st.markdown("##### 最近 7 天用電曲線")
            df_7d = df_history.last('7D')['power_kW'].reset_index()
            df_7d.columns = ['時間', '功率 (kW)']
            fig_line = px.line(df_7d, x='時間', y='功率 (kW)', template="plotly_dark")
            fig_line.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=400)
            st.plotly_chart(fig_line, use_container_width=True)
            with st.expander("📖 顯示最近 7 天的 15 分鐘原始數據"):
                st.dataframe(df_7d.set_index('時間'))

        with tab2:
            st.markdown("##### 近 30 天尖離峰佔比 (TOU)")
            if kpis['peak_kwh'] + kpis['off_peak_kwh'] > 0:
                labels = ['尖峰用電', '離峰用電']
                values = [kpis['peak_kwh'], kpis['off_peak_kwh']] 
                colors = ['#FF6B6B', '#4ECDC4'] 
                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=.4, 
                    marker=dict(colors=colors, line=dict(color='#333', width=1))
                )])
                fig_donut.update_layout(
                    template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                st.info("此圖表是基於「簡易型時間電價 (TOU)」的時段定義來劃分您的用電分佈。")
            else:
                st.info("無足夠資料可分析尖離峰佔比。")
                
        with tab3:
            st.markdown("##### 每日用電量 (kWh) 長條圖")
            df_daily_kwh = (df_history['power_kW'].resample('D').sum() * 0.25).to_frame(name="每日總度數 (kWh)")
            min_date = df_daily_kwh.index.min().date()
            max_date = df_daily_kwh.index.max().date()
            default_start_date = max(min_date, max_date - timedelta(days=30))
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("選擇日期範圍 - 開始", value=default_start_date, min_value=min_date, max_value=max_date, key="hist_start")
            with col_date2:
                end_date = st.date_input("選擇日期範圍 - 結束", value=max_date, min_value=start_date, max_value=max_date, key="hist_end")
            filtered_daily_df = df_daily_kwh.loc[start_date:end_date]
            st.markdown(f"**{start_date} 至 {end_date} 數據**")
            fig_bar = px.bar(filtered_daily_df, y='每日總度數 (kWh)', template="plotly_dark")
            fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
            with st.expander("📖 顯示每日數據表格"):
                st.dataframe(filtered_daily_df.style.format("{:.2f}"))