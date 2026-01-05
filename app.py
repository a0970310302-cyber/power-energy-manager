import streamlit as st
import time
import pandas as pd
import os
import traceback # 新增這個庫來顯示完整錯誤
from streamlit_lottie import st_lottie
# 注意：暫時移除 concurrent.futures 以便除錯

# 匯入原本的 UI 模組
from app_utils import load_lottiefile
from page_home import show_home_page
from page_dashboard import show_dashboard_page
from page_analysis import show_analysis_page
from page_tutorial import show_tutorial_page

# 匯入後端服務
from model_service import load_resources_and_predict

st.set_page_config(layout="wide", page_title="智慧電能管家")

# ==========================================
# 🛠️ [除錯區塊] 檢查雲端環境檔案
# ==========================================
def debug_check_files():
    st.warning("🛠️ 進入除錯模式：檢查檔案系統...")
    try:
        files = os.listdir('.')
        st.write(f"當前工作目錄: {os.getcwd()}")
        st.write("目錄下檔案列表:", files)
        
        required = ["final_training_data_with_humidity.csv", "lgbm_model.pkl", "lstm_model.keras"]
        missing = [f for f in required if f not in files]
        
        if missing:
            st.error(f"❌ 致命錯誤：雲端環境找不到以下檔案: {missing}")
            st.stop()
        else:
            st.success("✅ 關鍵檔案檢查通過")
    except Exception as e:
        st.error(f"檢查檔案時發生錯誤: {e}")

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
# 核心修改：同步載入函式 (取代原本的 ThreadPool)
# ==========================================
def ensure_data_loaded():
    """
    修改版：不做背景執行，直接在前景執行並印出每一步，
    這樣如果卡住或報錯，畫面會直接顯示。
    """
    if st.session_state.app_ready:
        return True

    # 1. 先執行檔案檢查
    debug_check_files()

    st.info("⚡ 正在載入模型與數據 (同步除錯模式)...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("正在呼叫 load_resources_and_predict()...")
        
        # 直接呼叫，不使用 ThreadPool
        start_time = time.time()
        pred_df, curr_df = load_resources_and_predict()
        end_time = time.time()
        
        status_text.text(f"函式執行完成，耗時 {end_time - start_time:.2f} 秒")
        
        if pred_df is None:
            st.error("❌ 載入失敗：model_service 回傳了 None。請檢查 logs。")
            st.stop()
            
        st.session_state.prediction_result = pred_df
        st.session_state.current_data = curr_df
        st.session_state.app_ready = True
        
        progress_bar.progress(100)
        time.sleep(0.5) # 讓使用者看到完成
        progress_bar.empty()
        status_text.empty()
        st.rerun() # 重新整理進入主頁
        
    except Exception as e:
        # 這是最重要的部分：抓出所有錯誤並顯示
        st.error("❌ 發生嚴重錯誤！")
        st.code(traceback.format_exc()) # 印出完整的錯誤追蹤
        st.stop()

    return False