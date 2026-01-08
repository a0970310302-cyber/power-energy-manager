# page_tutorial.py
import streamlit as st
import time
from streamlit_lottie import st_lottie
from app_utils import load_lottiefile, load_data
from model_service import load_resources_and_predict

def show_tutorial_page():
    """
    全螢幕導覽模式 - 修正白屏問題 (先渲染 UI，最後再跑模型)
    """
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # ==========================================
    # 🎨 UI 渲染區 (先做這個，確保畫面秒開)
    # ==========================================
    
    # 視覺置中佈局
    st.write("#") # 頂部間距
    _, col2, _ = st.columns([0.5, 2, 0.5])

    with col2:
        step = st.session_state.tutorial_step
        
        # 步驟 1: 歡迎
        if step == 1:
            st.title("⚡ 歡迎進入「智慧電能管理系統」")
            lottie_logo = load_lottiefile("lottiefiles/intelligent_tour_guide_robot.json")
            if lottie_logo:
                st_lottie(lottie_logo, speed=1, loop=True, height=350, key="tutorial_hero")

            st.markdown("""
            ### 掌握電能，就像導航一樣簡單。
            您是否曾經在收到帳單時，才發現電費超出預期？
            智慧電管家結合了 **LSTM 深度學習** 與 **氣候動態模擬**，為您提前一個月預測帳單。
            """)
            st.write("#")
            if st.button("探索 AI 的運作原理 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()

        # 步驟 2: 雙週期修正
        elif step == 2:
            st.title("📊 雙月滾動式修正趨勢")
            st.markdown("""
            ### 數據不只是冷冰冰的數字，而是「活著的預報」。
            我們獨創的雙週期監控系統，將為您呈現：
            
            * **近期高精度預測**：抓取未來 48 小時的瞬間變動。
            * **遠期帳單推估**：模擬未來氣溫，預估直到結算日的總體支出。
            
            > **視覺提示：** 實線代表過去，點線代表精確未來，虛線代表長期趨勢。
            """)
            # 可以加入一個簡單的示意圖位置
            st.write("#")
            c1, c2 = st.columns(2)
            if c1.button("⬅ 回上一步", use_container_width=True):
                st.session_state.tutorial_step = 1
                st.rerun()
            if c2.button("如何優化我的電費？ ➔", use_container_width=True):
                st.session_state.tutorial_step = 3
                st.rerun()

        # 步驟 3: 決策建議
        elif step == 3:
            st.title("💡 智慧省錢決策室")
            st.markdown("""
            ### 幫您選出最適合的台電費率。
            系統會自動對比 **累進費率** 與 **時間電價** 的實際成本。
            
            當 AI 偵測到您的用電模式切換到時間電價會更便宜時，
            我們會第一時間在首頁發出「省錢建議」，直接幫您看緊荷包。
            """)
            st.write("#")
            c1, c2 = st.columns(2)
            if c1.button("⬅ 回上一步", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()
            
            # 判斷按鈕狀態
            is_ready = st.session_state.get("app_ready", False)
            
            if is_ready:
                btn_text = "一切準備就緒，進入控制台！ ➔"
                btn_type = "primary"
            else:
                btn_text = "AI 正在完成數據對齊，請稍候... ➔"
                btn_type = "secondary"

            if c2.button(btn_text, type=btn_type, use_container_width=True):
                if is_ready:
                    st.session_state.page = "home"
                    st.session_state.tutorial_finished = True
                    st.rerun()
                else:
                    st.toast("AI 仍在計算中，請再給我們幾秒鐘...")

        # 底部進度條
        st.write("---")
        st.progress(step / 3)
        st.caption(f"導覽進度：{step} / 3")

    # ==========================================
    # 🚀 背景運算區 (移到最後面！避免白屏)
    # ==========================================
    if not st.session_state.get("app_ready", False):
        # 這裡不需要 with st.empty()，因為程式已經畫完 UI 了
        # 我們直接在腳本末端執行運算，使用者只會看到右上角的 "Running"
        try:
            if "prediction_result" not in st.session_state:
                res_df, hist_df = load_resources_and_predict() 
                st.session_state.prediction_result = res_df
                st.session_state.current_data = hist_df
                st.session_state.app_ready = True
                # 這裡不呼叫 rerurn，以免使用者看一半被強制重新整理
                # 當使用者點擊按鈕時，app_ready 已經是 True 了
        except:
            pass