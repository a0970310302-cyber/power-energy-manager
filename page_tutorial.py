# page_tutorial.py
import streamlit as st
import time
from streamlit_lottie import st_lottie
from app_utils import load_lottiefile
from model_service import load_resources_and_predict

def show_tutorial_page():
    """
    【故事模式】AI 導遊帶路 -> 全螢幕動感載入 -> 進入首頁
    """
    # 初始化導覽步驟
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # ==========================================
    # 🕵️‍♂️ 背景偷跑區 (Background Pre-fetch)
    # ==========================================
    if not st.session_state.get("app_ready", False):
        try:
            if "prediction_result" not in st.session_state:
                pass 
        except:
            pass

    # ==========================================
    # 🎬 模式切換邏輯
    # ==========================================
    step = st.session_state.tutorial_step

    # 如果進入 "loading" 模式 (使用者點了開始，但數據還沒好)
    if step == "loading":
        show_fullscreen_loading()
        return  # 阻斷後續渲染，只顯示 Loading 畫面

    # ==========================================
    # 📖 一般導覽模式 (Step 1~3)
    # ==========================================
    
    # 增加頂部留白
    st.write("#") 
    
    # 使用置中佈局：左(空)-中(內容)-右(空)
    _, col_main, _ = st.columns([0.5, 2, 0.5])

    with col_main:
        # 🤖 核心修改：將 AI 導遊機器人固定在每一頁的最上方
        # 這創造了一種「它一直在這裡陪你」的連貫感
        robot_anim = load_lottiefile("lottiefiles/Intelligent_tour_guide_robot_green.json")
        if robot_anim:
            # height=280 讓它夠大，成為畫面的主角
            st_lottie(robot_anim, speed=1, loop=True, height=280, key=f"guide_robot_{step}")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=150)

        # --- Step 1: 歡迎畫面 ---
        if step == 1:
            st.markdown("<h2 style='text-align: center;'>⚡ 歡迎啟動「智慧電能管家」</h2>", unsafe_allow_html=True)
            
            st.info("""
            **嗨！我是您的 AI 電能導航員。** 🤖
            傳統電表只能紀錄過去，但我能帶您看見未來。
            讓我花 **30 秒** 為您介紹這個系統的強大功能。
            """)
            
            st.write("#")
            if st.button("第一招：預知未來 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()

        # --- Step 2: 雙軌預測機制 ---
        elif step == 2:
            st.markdown("<h2 style='text-align: center;'>🧠 我的雙核心大腦</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            > **我不只分析歷史數據，我還模擬了未來的氣候。**
            
            為了達到最高精準度，我同時運行兩套神經網路：
            """)
            
            c1, c2 = st.columns(2)
            with c1:
                st.error("🔴 **近期高精準**", icon="🔥")
                st.caption("LSTM + LightGBM")
                st.write("針對未來 **48小時** 進行毫秒級運算，連您幾點洗澡我都知道。")
            
            with c2:
                st.warning("🟠 **遠期趨勢圖**", icon="🌤️")
                st.caption("Climate Simulator")
                st.write("模擬直到 **結算日** 的氣溫變化，幫您算出最終帳單金額。")

            st.write("#")
            btn_c1, btn_c2 = st.columns([1, 2])
            if btn_c1.button("⬅ 上一步", use_container_width=True):
                st.session_state.tutorial_step = 1
                st.rerun()
            if btn_col2 := btn_c2.button("第二招：省錢決策 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 3
                st.rerun()

        # --- Step 3: 決策與啟動 ---
        elif step == 3:
            st.markdown("<h2 style='text-align: center;'>💰 我會幫您看緊荷包</h2>", unsafe_allow_html=True)
            
            st.success("""
            **不只是看圖表，我會直接給您建議：**
            
            * **💸 預算紅燈**：當我發現月底會超支時，我會立刻發出警報。
            * **⚖️ 費率裁判**：我會自動幫您算，「累進電價」與「時間電價」哪個更便宜。
            """)
            
            st.divider()
            
            btn_c1, btn_c2 = st.columns([1, 2])
            if btn_c1.button("⬅ 上一步", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()
            
            # 啟動按鈕
            if btn_c2.button("🚀 啟動系統監控", type="primary", use_container_width=True):
                # 如果後台已經好了，直接進首頁
                if st.session_state.get("app_ready", False):
                    st.session_state.page = "home"
                    st.session_state.tutorial_finished = True
                    st.rerun()
                else:
                    # 如果還沒好，切換到 "loading" 模式 (全螢幕動圖)
                    st.session_state.tutorial_step = "loading"
                    st.rerun()

        # 底部進度條
        st.write("---")
        st.progress(step / 3)
        st.caption(f"導覽進度：{step} / 3")


def show_fullscreen_loading():
    """
    【Loading 模式】全螢幕動圖 + 左下角進度條 + 真實運算
    """
    # 1. 載入 Loading 動圖
    loading_anim = load_lottiefile("lottiefiles/loading_animation.json")
    
    # 2. 佈局
    placeholder_lottie = st.empty()
    placeholder_bar = st.empty()

    # A. 顯示全螢幕動圖 (這裡就不顯示機器人了，改顯示系統運作圖)
    with placeholder_lottie:
        _, c_center, _ = st.columns([1, 2, 1])
        with c_center:
            st.write("#")
            st.write("#")
            if loading_anim:
                # 這裡可以放 loading_animation
                st_lottie(loading_anim, height=400, key="full_loader", speed=1)
            else:
                st.spinner("系統啟動中...")

    # B. 進度條邏輯
    progress_text = "正在載入 AI 模型權重..."
    my_bar = placeholder_bar.progress(0, text=progress_text)

    # Fake progress
    for percent_complete in range(0, 40, 10):
        time.sleep(0.1)
        my_bar.progress(percent_complete, text="正在同步歷史氣象資料...")

    # Real work
    try:
        my_bar.progress(50, text="啟動 LSTM 類神經網路預測中 (這可能需要幾秒鐘)...")
        res_df, hist_df = load_resources_and_predict() 
        st.session_state.prediction_result = res_df
        st.session_state.current_data = hist_df
        st.session_state.app_ready = True
    except Exception as e:
        st.error(f"啟動失敗: {e}")
        st.stop()

    # Finish
    for percent_complete in range(60, 101, 20):
        time.sleep(0.1)
        my_bar.progress(percent_complete, text="數據視覺化渲染完成！")
    
    time.sleep(0.5)

    # C. 跳轉首頁
    st.session_state.page = "home"
    st.session_state.tutorial_finished = True
    st.rerun()