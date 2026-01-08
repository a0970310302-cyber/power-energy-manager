# page_tutorial.py
import streamlit as st
import time
from streamlit_lottie import st_lottie
from app_utils import load_lottiefile
from model_service import load_resources_and_predict

def show_tutorial_page():
    """
    【故事模式】深度導覽 -> 全螢幕動感載入 -> 進入首頁
    """
    # 初始化導覽步驟
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # ==========================================
    # 🕵️‍♂️ 背景偷跑區 (Background Pre-fetch)
    # ==========================================
    # 這是為了讓使用者在看導覽時，我們就先偷偷算。
    # 但如果使用者看太快，導致這裡還沒算完，下面會有「Loading 模式」接手。
    if not st.session_state.get("app_ready", False):
        try:
            # 檢查是否已經有結果，沒有才跑
            if "prediction_result" not in st.session_state:
                # 這裡不呼叫 load_resources_and_predict，避免卡住 UI 渲染
                # 我們把計算推遲到最後的 Loading 階段，或者依賴 OS 的快取
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
    
    # 增加頂部留白，讓視覺更舒適
    st.write("#") 
    
    # 使用置中佈局
    _, col_main, _ = st.columns([0.5, 2, 0.5])

    with col_main:
        # --- Step 1: 歡迎 ---
        if step == 1:
            st.title("⚡ 歡迎啟動「智慧電能管家」")
            
            lottie_hero = load_lottiefile("lottiefiles/intelligent_tour_guide_robot.json")
            if lottie_hero:
                st_lottie(lottie_hero, speed=1, loop=True, height=300, key="hero_anim")

            st.markdown("""
            ### 告別被電費帳單嚇到的日子。
            
            傳統的電表只能告訴你「用了多少」，而我們能告訴你「將要花多少」。
            透過 **Hybrid AI 雙核心預測技術**，我們為您打造了家庭能源的導航系統。
            """)
            
            st.write("#")
            if st.button("下一步：解密 AI 大腦 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()

        # --- Step 2: 雙軌預測機制 ---
        elif step == 2:
            st.title("🧠 為什麼我們能預知未來？")
            
            st.info("""
            **我們不只看歷史，更模擬未來氣候。**
            系統同時運行兩套神經網路模型：
            """)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔴 近期高精準")
                st.caption("LSTM + LightGBM")
                st.write("針對未來 **48小時** 的生活作息進行毫秒級運算，精準捕捉每一個家電的開啟瞬間。")
            
            with c2:
                st.markdown("#### 🟠 遠期趨勢圖")
                st.caption("Climate Simulator")
                st.write("引入歷史氣象資料庫，模擬直到 **帳單結算日** 的溫濕度變化，推算最終電費金額。")

            st.write("#")
            btn_col1, btn_col2 = st.columns([1, 2])
            if btn_col1.button("⬅ 上一步", use_container_width=True):
                st.session_state.tutorial_step = 1
                st.rerun()
            if btn_col2.button("下一步：省錢決策 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 3
                st.rerun()

        # --- Step 3: 決策與啟動 ---
        elif step == 3:
            st.title("💰 您的荷包守護者")
            
            st.markdown("""
            ### 不只是看圖表，而是給建議。
            
            我們會在儀表板上即時計算：
            * **💸 預算警示**：當 AI 預測月底即將超支時，提早變色警示。
            * **⚖️ 費率試算**：自動對比「累進電價」與「時間電價」，找出最佳方案。
            """)
            
            st.divider()
            st.markdown("##### 準備好開始了嗎？")
            
            btn_c1, btn_c2 = st.columns([1, 2])
            if btn_c1.button("⬅ 上一步", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()
            
            # 這是關鍵按鈕！
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

        # 底部進度指示器
        st.write("---")
        st.progress(step / 3)
        st.caption(f"導覽進度：{step} / 3")


def show_fullscreen_loading():
    """
    【Loading 模式】全螢幕動圖 + 左下角進度條 + 真實運算
    """
    # 1. 載入動圖
    loading_anim = load_lottiefile("lottiefiles/loading_animation.json")
    
    # 2. 佈局：使用三個容器來達成置中與左下角效果
    # 這裡利用 st.empty() 來動態更新內容
    
    placeholder_lottie = st.empty()
    placeholder_status = st.empty()
    placeholder_bar = st.empty()

    # A. 顯示全螢幕動圖 (稍微放大一點)
    with placeholder_lottie:
        _, c_center, _ = st.columns([1, 2, 1])
        with c_center:
            st.write("#")
            st.write("#")
            if loading_anim:
                st_lottie(loading_anim, height=400, key="full_loader", speed=1)
            else:
                st.spinner("系統啟動中...")

    # B. 開始執行運算 (這會卡住畫面，這是正常的)
    # 我們先畫出進度條，讓使用者知道「開始跑了」
    
    progress_text = "正在載入 AI 模型權重..."
    my_bar = placeholder_bar.progress(0, text=progress_text)

    # --- 模擬動感進度條 (Visual Fake Progress) ---
    # 因為 load_resources_and_predict 是一次性函數，我們無法取得中間進度
    # 所以我們先跑一點點進度條，讓畫面動起來
    for percent_complete in range(0, 40, 10):
        time.sleep(0.1)
        my_bar.progress(percent_complete, text="正在同步歷史氣象資料...")

    # --- 🔥 真實運算開始 ---
    try:
        my_bar.progress(50, text="啟動 LSTM 類神經網路預測中 (這可能需要幾秒鐘)...")
        
        # 呼叫核心運算 (這行執行時，畫面會凍結是正常的 Streamlit 特性)
        res_df, hist_df = load_resources_and_predict() 
        
        # 存入 Session
        st.session_state.prediction_result = res_df
        st.session_state.current_data = hist_df
        st.session_state.app_ready = True
        
    except Exception as e:
        st.error(f"啟動失敗: {e}")
        st.stop()

    # --- 運算結束，跑完剩下的進度條 ---
    for percent_complete in range(60, 101, 20):
        time.sleep(0.1)
        my_bar.progress(percent_complete, text="數據視覺化渲染完成！")
    
    time.sleep(0.5) # 停留一下讓使用者看到 100%

    # C. 跳轉首頁
    st.session_state.page = "home"
    st.session_state.tutorial_finished = True
    st.rerun()