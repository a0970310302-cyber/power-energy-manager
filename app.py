import streamlit as st
import time
import pandas as pd
import os
import traceback 
from streamlit_lottie import st_lottie

# 匯入 UI 模組
from app_utils import load_lottiefile
from page_home import show_home_page
from page_dashboard import show_dashboard_page
from page_analysis import show_analysis_page
from page_tutorial import show_tutorial_page

# 匯入後端服務
from model_service import load_resources_and_predict

# 設定頁面資訊
st.set_page_config(layout="wide", page_title="智慧電能管家")

# ==========================================
# 🛠️ [除錯區塊] 檢查雲端環境檔案
# ==========================================
def debug_check_files():
    # 只有在還沒準備好時才檢查，避免畫面一直被洗版
    if not st.session_state.get("app_ready", False):
        st.warning("🛠️ 進入除錯模式：檢查檔案系統...")
        try:
            files = os.listdir('.')
            # st.write(f"當前工作目錄: {os.getcwd()}") # 註解掉以保持畫面乾淨，需要時再打開
            # st.write("目錄下檔案列表:", files)
            
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
# 同步載入函式
# ==========================================
def ensure_data_loaded():
    """
    修改版：不做背景執行，直接在前景執行並印出每一步。
    回傳 True 代表已載入完成，False 代表正在載入中。
    """
    if st.session_state.app_ready:
        return True

    # 1. 先執行檔案檢查
    debug_check_files()

    st.info("⚡ 正在載入模型與數據 (同步除錯模式)...這可能需要 10-30 秒")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("正在呼叫 load_resources_and_predict()...")
        
        # 計時開始
        start_time = time.time()
        
        # 執行核心載入 (這一步最花時間)
        pred_df, curr_df = load_resources_and_predict()
        
        end_time = time.time()
        status_text.text(f"函式執行完成，耗時 {end_time - start_time:.2f} 秒")
        
        if pred_df is None:
            st.error("❌ 載入失敗：model_service 回傳了 None。請檢查下方 Logs 或 model_service.py。")
            st.stop()
            
        st.session_state.prediction_result = pred_df
        st.session_state.current_data = curr_df
        st.session_state.app_ready = True
        
        progress_bar.progress(100)
        time.sleep(0.5) 
        st.rerun() # 重新整理以進入主頁面
        
    except Exception as e:
        st.error("❌ 發生嚴重錯誤！")
        st.code(traceback.format_exc()) # 印出完整的錯誤追蹤
        st.stop()

    return False

# ==========================================
# 🚀 [這就是缺少的] 主程式執行流程
# ==========================================
def main():
    # 1. 側邊欄導航 (Sidebar)
    with st.sidebar:
        st.title("功能選單")
        
        # 使用按鈕切換頁面
        if st.button("🏠 主頁", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
            
        if st.button("📈 用電儀表板", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
            
        if st.button("🔬 AI 決策分析室", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()

        st.markdown("---")
        if st.button("🔄 重新抓取數據"):
            st.session_state.app_ready = False
            st.rerun()

    # 2. 確保數據載入 (守門員)
    # 如果還沒載入好，程式會停在 ensure_data_loaded 裡面，不會往下跑
    if not ensure_data_loaded():
        st.stop() 

    # 3. 頁面路由 (Router)
    # 只有當數據載入完成後，才會執行到這裡
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "dashboard":
        show_dashboard_page()
    elif st.session_state.page == "analysis":
        show_analysis_page()
    elif st.session_state.page == "tutorial":
        show_tutorial_page()
    else:
        show_home_page()

# 執行主程式
if __name__ == "__main__":
    main()