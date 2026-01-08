# app.py 完整代碼
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
# 🛠️ 視覺與初始化邏輯
# ==========================================
def initialize_system():
    # 確保 session_state 存在
    if 'page' not in st.session_state:
        st.session_state.page = "tutorial"  # 強制初次載入為導覽
    if 'tutorial_finished' not in st.session_state:
        st.session_state.tutorial_finished = False
    if 'app_ready' not in st.session_state:
        st.session_state.app_ready = False

def apply_custom_style():
    # 判斷是否為導覽模式，若是則隱藏側邊欄達成「全螢幕」
    if st.session_state.page == "tutorial":
        st.markdown("""
            <style>
                [data-testid="stSidebar"] {display: none;}
                [data-testid="stSidebarNav"] {display: none;}
                .stAppHeader {display: none;}
                .block-container {padding-top: 1rem;}
            </style>
        """, unsafe_allow_html=True)
    else:
        # 非導覽模式：側邊欄 Logo 放大
        st.markdown("""
            <style>
                .block-container {padding-top: 2rem;}
            </style>
        """, unsafe_allow_html=True)

def main():
    initialize_system()
    apply_custom_style()

    if st.session_state.page != "tutorial":
        with st.sidebar:
            loading_lottie = load_lottiefile("lottiefiles/Intelligent_tour_guide_robot.json")
            if loading_lottie:
                st_lottie(loading_lottie, speed=1, loop=True, height=250, key="sidebar_loading")
        
            st.write("---")
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
            if st.button("🔄 更新即時數據", use_container_width=True):
                st.session_state.app_ready = False
                st.rerun()
                
            st.caption(f"Ver 2.1.0 | Status: {'🟢 Online' if st.session_state.app_ready else '🟡 Loading'}")

    # 2. 路由控制
    if st.session_state.page == "tutorial":
        show_tutorial_page()
    elif st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "dashboard":
        show_dashboard_page()
    elif st.session_state.page == "analysis":
        show_analysis_page()

if __name__ == "__main__":
    main()