# app.py
import streamlit as st
import time
import pandas as pd
import os
import traceback 
from streamlit_lottie import st_lottie

# 匯入 UI 模組
from app_utils import load_lottiefile, load_data
from page_home import show_home_page
from page_dashboard import show_dashboard_page
from page_analysis import show_analysis_page
from page_tutorial import show_tutorial_page

# 匯入後端服務
from model_service import load_resources_and_predict

# 設定頁面資訊
st.set_page_config(layout="wide", page_title="智慧電能管家", page_icon="⚡")

# ==========================================
# 🔍 系統健康檢查 (已更新為 Hybrid 架構)
# ==========================================
def check_system_integrity():
    if not st.session_state.get("app_ready", False):
        try:
            files = os.listdir('.')
            # [關鍵修正] 更新為新版模型所需的檔案清單
            required = [
                "final_training_data_with_humidity.csv", 
                "hybrid_residual.pkl",  # 新的總指揮官 (Config)
                "lgbm_residual.pkl",    # 新的 LGBM
                "lstm_hybrid.keras"     # 新的 LSTM
            ]
            missing = [f for f in required if f not in files]
            
            if missing:
                st.error(f"⚠️ 系統錯誤：偵測到關鍵檔案遺失: {missing}")
                st.stop()
        except Exception as e:
            st.error(f"系統檢查失敗: {e}")

# ==========================================
# Session State 初始化
# ==========================================
if "app_ready" not in st.session_state:
    st.session_state.app_ready = False
if "tutorial_complete" not in st.session_state:
    st.session_state.tutorial_complete = False
if "page" not in st.session_state:
    st.session_state.page = "home"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "current_data" not in st.session_state:
    st.session_state.current_data = None

# ==========================================
# 資料載入核心
# ==========================================
def initialize_system():
    """
    執行系統初始化與數據載入
    """
    if st.session_state.app_ready:
        return True

    # 1. 背景檢查
    check_system_integrity()

    # 2. 顯示載入畫面
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        st.info("⚡ 系統啟動中，正在連接 AI 模型與雲端數據庫...")
        progress_bar = st.progress(0)
        
        try:
            progress_bar.progress(10)
            time.sleep(0.1)
            
            # --- 1. 先讀取歷史資料 ---
            # 這裡讀到的資料已經被 app_utils 放大過 (x20)
            df_history = load_data()
            
            if df_history is None or df_history.empty:
                st.error("❌ 無法讀取歷史數據，請檢查資料來源。")
                st.stop()

            progress_bar.progress(40)
            
            # --- 2. 將資料傳給模型服務 ---
            # 傳入 df_history，確保模型使用相同的資料基礎
            pred_df, curr_df = load_resources_and_predict(df_history)
            
            progress_bar.progress(90)
            
            if pred_df is None:
                st.error("❌ AI 預測失敗，請稍後再試。")
                st.stop()
                
            # 存入 Session
            st.session_state.prediction_result = pred_df
            st.session_state.current_data = curr_df
            st.session_state.app_ready = True
            
            progress_bar.progress(100)
            time.sleep(0.5) 
            
            # 清除載入畫面
            loading_placeholder.empty()
            st.rerun() 
            
        except Exception as e:
            st.error("❌ 系統發生預期外的錯誤")
            with st.expander("查看錯誤詳情 (給開發人員)"):
                st.code(traceback.format_exc())
            st.stop()

    return False

# ==========================================
# 🚀 主程式進入點
# ==========================================
def main():
    # 1. 側邊欄導航
    with st.sidebar:
        if st.session_state.page != "tutorial":
            from streamlit_lottie import st_lottie
            from app_utils import load_lottiefile
        
            loading_lottie = load_lottiefile("Intelligent_tour_guide_robot.json")
            if loading_lottie:
            # 設定較小的高度使其像一個 Logo 或狀態圖示
                st_lottie(loading_lottie, speed=1, loop=True, height=120, key="sidebar_loading")
        
            st.write("---") # 分隔線
        st.title("⚡ 功能選單")
        
        if st.button("🏠 首頁總覽", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
            
        if st.button("📈 用電儀表板", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
            
        if st.button("🧠 AI 決策分析", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()

        st.markdown("---")
        # 重新整理按鈕
        if st.button("🔄 更新即時數據"):
            st.session_state.app_ready = False
            st.rerun()
            
        st.markdown("---")
        st.caption(f"Ver 2.0.0 (Hybrid Residual) | Status: {'🟢 Online' if st.session_state.app_ready else '🟡 Loading'}")

    # 2. 系統初始化守門員
    if not initialize_system():
        st.stop() 

    # 3. 頁面路由
    if st.session_state.page == "tutorial":
        show_tutorial_page()
    elif st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "dashboard":
        show_dashboard_page()
    elif st.session_state.page == "analysis":
        show_analysis_page()
    elif st.session_state.page == "tutorial":
        show_tutorial_page()
    else:
        show_home_page()

if __name__ == "__main__":
    main()