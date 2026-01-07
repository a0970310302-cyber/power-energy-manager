# page_home.py
import streamlit as st
from datetime import timedelta, datetime
import pandas as pd

# 匯入共用函式 (包含新的全能計費報告)
from app_utils import load_data, get_core_kpis, get_billing_report

def show_home_page():
    """
    【AI 每日晨報】風格主頁
    """
    st.title("🏠 家庭智慧電管家")
    
    # --- 0. 資料準備 ---
    # 優先讀取 Session State
    if "current_data" in st.session_state and st.session_state.current_data is not None:
        df_history = st.session_state.current_data
    else:
        df_history = load_data()

    if df_history is None or df_history.empty:
        st.warning("⚠️ 系統初始化中，等待數據接入...")
        return

    # 計算 KPI
    kpis = get_core_kpis(df_history)
    
    # [關鍵修改] 取得統一的計費報告
    report = get_billing_report(df_history)
    
    # --- 1. AI 總結語 (根據 report 狀態) ---
    welcome_msg = ""
    # 優先級 1: 預算危險
    if report['status'] == "danger":
        welcome_msg = f"🚨 **警報：預測本月將超支 {report['predicted_bill'] - report['budget']:,} 元！建議立即查看儀表板。**"
        st.error(welcome_msg, icon="🚨")
    # 優先級 2: 發現省錢機會 (Savings > 100)
    elif report['savings'] > 100:
        welcome_msg = f"💡 **早安！AI 發現若切換電價方案，本月可省下 {report['savings']:,} 元，建議查看詳情。**"
        st.info(welcome_msg, icon="💡")
    # 優先級 3: 一切正常
    else:
        welcome_msg = f"✅ **早安！目前用電狀況良好，預算控制在安全範圍內。**"
        st.success(welcome_msg, icon="✅")

    st.markdown("---")

    # --- 2. 三大決策卡片 ---
    col1, col2, col3 = st.columns(3)

    # === 卡片 1: 財務安全 (使用 report 數據) ===
    with col1:
        with st.container(border=True):
            st.markdown("#### 💰 預算監控")
            
            # 使用統一計算出的狀態
            if report['status'] == "safe":
                st.markdown("# :green[安全]")
            elif report['status'] == "warning":
                st.markdown("# :orange[警戒]")
            else:
                st.markdown("# :red[超支]")
                
            st.caption(f"預測結算 ${report['predicted_bill']:,}")
            st.progress(report['usage_percent'])
            st.markdown(f"**目標：${report['budget']:,}**")

    # === 卡片 2: 方案優化 (使用 report 數據) ===
    with col2:
        with st.container(border=True):
            st.markdown("#### 📉 方案最佳化")
            savings = report['savings']
            
            if savings > 100:
                st.markdown("# :green[建議切換]")
                st.metric("可節省", f"NT$ {savings:,}", delta="時間電價更優")
            else:
                st.markdown("# :blue[維持現狀]")
                # 如果 savings 是負的，代表累進更省
                st.metric("累進最省", "最佳方案", delta_color="off")

    # === 卡片 3: 行為診斷 (維持 KPI 邏輯) ===
    with col3:
        with st.container(border=True):
            st.markdown("#### 🩺 用電健康度")
            trend = kpis['weekly_delta_percent']
            if trend > 15:
                st.markdown("# :red[異常飆升]")
                st.metric("較上週", f"+{trend:.1f}%", delta_color="inverse")
            elif trend < -10:
                st.markdown("# :green[顯著節能]")
                st.metric("較上週", f"{trend:.1f}%", delta_color="inverse")
            else:
                st.markdown("# :blue[平穩正常]")
                st.metric("較上週", f"{trend:+.1f}%")

    st.markdown("---")

    # --- 3. 快速入口 ---
    st.subheader("🚀 快速功能")
    q1, q2, q3, q4 = st.columns(4)
    
    if q1.button("📊 詳細儀表板", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()
        
    if q2.button("🔬 未來預測圖", use_container_width=True):
        st.session_state.page = "analysis"
        st.rerun()
        
    if q3.button("🔄 立即更新數據", use_container_width=True):
        # 觸發重新載入
        st.session_state.app_ready = False
        st.rerun()
             
    if q4.button("🔔 測試 Line 通知", help="發送測試訊息到綁定的 Line 群組", use_container_width=True):
        st.toast("已發送測試警報！")