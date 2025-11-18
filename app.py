import streamlit as st
import time
from streamlit_lottie import st_lottie

# 匯入我們拆分出去的檔案
from app_utils import load_lottiefile, load_model, load_data
from page_home import show_home_page
from page_dashboard import show_dashboard_page
from page_analysis import show_analysis_page
from page_tutorial import show_tutorial_page # 匯入教學頁面

# --- 0. 頁面設定 (必須是第一個 st 指令) ---
st.set_page_config(layout="wide", page_title="智慧電能管家")

# --- 1. 初始化所有 Session State 旗標 ---
if "app_ready" not in st.session_state:
    st.session_state.app_ready = False
if "tutorial_complete" not in st.session_state:
    st.session_state.tutorial_complete = False

# --- 2. 應用程式三階段邏輯 ---

# --- 階段一：開場動畫 (Loading Screen) ---
if not st.session_state.app_ready:
    


    lottie_filepath = "lottiefiles/loading_animation.json"
    lottie_json = load_lottiefile(lottie_filepath)
    
    # 使用空白推擠內容到中間
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_json:
            st_lottie(lottie_json, speed=1, width=400, height=400, key="loading_lottie")
        else:
            st.warning("動畫載入失敗...")
        
        st.subheader("💡 智慧電能管家 啟動中...")
        st.text("正在為您載入 AI 模型與歷史數據...")

    # 【⭐ 修改點 3：強制等待 3 秒 ⭐】
    # 這是解決「動畫一閃而過」的關鍵！
    # 即使模型是從快取秒開的，我們也讓動畫至少播 3 秒
    time.sleep(3)

    # 觸發快取函式
    model = load_model()
    df_history = load_data()

    # 載入成功，切換狀態
    if model is not None and not df_history.empty:
        st.session_state.app_ready = True
        st.rerun()
    else:
        st.error("啟動失敗：無法載入模型或數據。請檢查您的檔案。")
        st.stop()

# --- 階段二：教學導覽 ---
elif not st.session_state.tutorial_complete:
    show_tutorial_page()

# --- 階段三：主應用程式 ---
else:
    # 1. 側邊欄
    with st.sidebar:
        # 嘗試載入不同的 Logo 檔名 (容錯處理)
        lottie_logo = load_lottiefile("lottiefiles/intelligent_tour_guide_robot.json")
        if not lottie_logo:
             lottie_logo = load_lottiefile("lottiefiles/Intelligent_tour_guide_robot_green.json")
             
        if lottie_logo:
            st_lottie(
                lottie_logo,
                speed=1,
                loop=True,
                quality="high",
                height=150,
                key="logo_animation"
            )
        else:
            st.header("AI Power Forecast")
            
        st.header("功能選單")
        st.divider()

        # 初始化預設頁面
        if 'page' not in st.session_state:
            st.session_state.page = "🏠 主頁"
        
        current_page = st.session_state.page

        if st.button("🏠 主頁", key="nav_home", use_container_width=True, type="secondary" if current_page != "🏠 主頁" else "primary"):
            st.session_state.page = "🏠 主頁"
            st.rerun()
        
        if st.button("📈 用電儀表板", key="nav_dashboard", use_container_width=True, type="secondary" if current_page != "📈 用電儀表板" else "primary"):
            st.session_state.page = "📈 用電儀表板"
            st.rerun()

        if st.button("🔬 AI 決策分析室", key="nav_analysis", use_container_width=True, type="secondary" if current_page != "🔬 AI 決策分析室" else "primary"):
            st.session_state.page = "🔬 AI 決策分析室"
            st.rerun()

    # 2. 頁面路由
    if current_page == "📈 用電儀表板":
        show_dashboard_page()
    elif current_page == "🔬 AI 決策分析室":
        show_analysis_page()
    else: # 預設或 "🏠 主頁"
        show_home_page()