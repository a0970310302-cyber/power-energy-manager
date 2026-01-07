# page_tutorial.py
import streamlit as st
import time
from streamlit_lottie import st_lottie
from app_utils import load_lottiefile, load_data

# 匯入後端服務，用於後台預熱
from model_service import load_resources_and_predict

def show_tutorial_page():
    """
    【故事模式】深度導覽與後台預處理
    """
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # --- 💡 關鍵優化：後台預處理 (Background Pre-loading) ---
    # 當使用者在看導覽時，如果模型還沒跑完，我們就在背景偷偷跑
    if not st.session_state.get("app_ready", False):
        # 使用一個隱藏的容器來跑，不干擾導覽 UI
        with st.empty():
            try:
                # 這裡不使用 st.spinner 避免干擾使用者閱讀導覽
                res_df, hist_df = load_resources_and_predict(steps=1200) 
                st.session_state.prediction_result = res_df
                st.session_state.current_data = hist_df
                st.session_state.app_ready = True
            except:
                pass # 失敗了也沒關係，首頁會再檢查一次

    # 使用欄位將所有內容置中
    _, col2, _ = st.columns([1, 2, 1])

    with col2:
        step = st.session_state.tutorial_step
        
        # ==========================================
        # 步驟 1: 歡迎與核心價值
        # ==========================================
        if step == 1:
            st.title("⚡ 歡迎使用智慧電管家")
            
            lottie_logo = load_lottiefile("lottiefiles/intelligent_tour_guide_robot.json")
            if lottie_logo:
                st_lottie(lottie_logo, speed=1, loop=True, height=250, key="tutorial_v1")
            else:
                st.info("💡 正在初始化 AI 大腦...")

            st.markdown("""
            ### 這不是普通的查詢工具，而是您的「電能導航員」。
            
            我們發現，傳統電費單總是「遲到的壞消息」。
            當您收到帳單時，電費早就噴掉了。
            
            **智慧電管家將為您：**
            * **預知未來**：在結算日前一個月就告訴您會花多少錢。
            * **滾動修正**：每天根據天氣與作息，重新計算最準確的趨勢。
            * **決策優化**：告訴您該不該切換時間電價，直接幫您省錢。
            """)
            st.divider()
            if st.button("我想了解 AI 如何預測 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()

        # ==========================================
        # 步驟 2: AI 短期與長期的差異 (深度了解)
        # ==========================================
        elif step == 2:
            st.title("🧠 AI 是如何思考的？")
            
            st.markdown("""
            ### 雙模型混合架構 (Hybrid AI)
            為了給您最精準的參考，我們使用了兩套 AI 大腦同時運作：
            
            1. **短期高精度 (LSTM + LightGBM)**: 
               負責抓取您未來 **48 小時** 的用電波動。它能感覺到您下班後的習慣，甚至連您煮飯的週期都能掌握。
            
            2. **長期趨勢推估 (WeatherSimulator)**: 
               針對結算日前的遠期預測。我們引入了歷史氣候數據，模擬未來的溫度趨勢。
               
            > **💡 小撇步：** 您會在儀表板看到「紅線」與「橘線」。紅線是您的精準未來，橘線是我們的預算趨勢參考。
            """)
            
            st.divider()
            c1, c2 = st.columns(2)
            if c1.button("⬅ 回上一步", use_container_width=True):
                st.session_state.tutorial_step = 1
                st.rerun()
            if c2.button("看看如何省錢 ➔", use_container_width=True):
                st.session_state.tutorial_step = 3
                st.rerun()

        # ==========================================
        # 步驟 3: 決策室與省錢邏輯
        # ==========================================
        elif step == 3:
            st.title("💰 錢要花在刀口上")
            
            st.markdown("""
            ### 為什麼我們強調「帳單」而非「度數」？
            對大多數人來說，100 度電很抽象，但 500 元很有感。
            
            **我們的決策分析室會做三件事：**
            1. **預算警報**：當 AI 發現您月底會超支，會提早兩週發出紅字警告。
            2. **資費對比**：自動計算「累進費率」與「時間電價」哪一個更適合您。
            3. **行為診斷**：如果您本週用電突然飆升，我們會直接指出異常，不讓電費悄悄流走。
            """)
            
            st.divider()
            c1, c2 = st.columns(2)
            if c1.button("⬅ 回上一步", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()
            
            # 最後一按鈕，根據後台加載狀態給予不同文案
            btn_label = "數據準備就緒，開始體驗！ ➔" if st.session_state.get("app_ready", False) else "正在完成最後加載... ➔"
            if c2.button(btn_label, type="primary", use_container_width=True):
                st.session_state.page = "home"
                st.session_state.tutorial_finished = True # 標記已看過
                st.rerun()

        # 進度條
        st.write("")
        st.progress(step / 3)
        st.caption(f"導覽進度：{step} / 3")