# page_tutorial.py
import streamlit as st
import time
from streamlit_lottie import st_lottie
from app_utils import load_lottiefile
from model_service import load_resources_and_predict

def show_tutorial_page():
    """
    【故事模式】左圖右文佈局 -> 深度價值溝通 -> 全螢幕動感載入
    """
    # 初始化導覽步驟
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # ==========================================
    # 🕵️‍♂️ 背景偷跑區 (Background Pre-fetch)
    # ==========================================
    # 這是為了讓使用者在看導覽時，我們就先偷偷算。
    if not st.session_state.get("app_ready", False):
        try:
            if "prediction_result" not in st.session_state:
                # 這裡不執行重型運算，避免卡頓，留給 Loading 階段處理
                pass 
        except:
            pass

    # ==========================================
    # 🎬 模式切換邏輯
    # ==========================================
    step = st.session_state.tutorial_step

    # 如果進入 "loading" 模式
    if step == "loading":
        show_fullscreen_loading()
        return

    # ==========================================
    # 📖 一般導覽模式 (左圖右文佈局)
    # ==========================================
    
    # 頂部留白
    st.write("#")
    
    # 建立左右兩欄：左邊放機器人(1.2)，右邊放內容(2.0)，中間留寬間距
    col_robot, col_content = st.columns([1.2, 2.0], gap="large")

    # --- 左側：永遠駐守的 AI 導遊 ---
    with col_robot:
        st.write("##") # 微調垂直位置，讓機器人置中一點
        robot_anim = load_lottiefile("lottiefiles/intelligent_tour_guide_robot.json")
        if robot_anim:
            st_lottie(robot_anim, speed=1, loop=True, height=350, key=f"robot_step_{step}")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=200)

    # --- 右側：根據步驟變化的內容 ---
    with col_content:
        
        # 🟢 Step 1: 核心價值 (Value Proposition)
        if step == 1:
            st.markdown("### ⚡ 歡迎啟動「智慧電能管家」")
            st.markdown("##### —— 您的家庭能源首席財務官")
            
            st.info("""
            **「為什麼帳單總是遲到的壞消息？」**
            
            傳統電表只能紀錄過去，讓您在月底面對帳單時措手不及。
            我們不同，我們是一套 **「具有預知能力」** 的決策系統。
            """)
            
            st.markdown("""
            **我們的三大核心價值：**
            1.  🔮 **預知未來**：提前 30 天告訴您本期帳單金額。
            2.  🛡️ **預算防護**：即時監控每一度電，超支前立刻攔截。
            3.  🧠 **決策大腦**：不只給數據，更直接告訴您「怎麼省」。
            """)
            
            st.write("#")
            if st.button("下一步：解密 AI 核心技術 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()

        # 🟡 Step 2: 技術獨特性 (Uniqueness & Technology)
        elif step == 2:
            st.markdown("### 🧠 獨家 Hybrid AI 雙軌預測技術")
            st.markdown("##### —— 結合深度學習與氣候模擬的完全體")
            
            st.markdown("""
            市面上的電量 APP 大多只能顯示歷史，**我們是唯一能模擬未來的系統。**
            為了達到 95% 以上的準確度，我們同時運行兩套神經網路：
            """)
            
            # 使用 Expander 讓版面乾淨但內容豐富
            with st.expander("🔴 紅線：LSTM 短期高精準模型", expanded=True):
                st.write("""
                專注於 **未來 48 小時** 的毫秒級運算。
                它學習了您的生活作息（何時洗澡、何時煮飯），能精準捕捉每一個家電開啟的瞬間波動。
                """)
                
            with st.expander("🟠 橘線：氣候模擬推估系統", expanded=True):
                st.write("""
                專注於 **直到結算日** 的長期趨勢。
                引入歷史氣象大數據，模擬未來的氣溫變化，幫您推算出最終的帳單總金額。
                """)

            st.write("#")
            c1, c2 = st.columns([1, 2])
            if c1.button("⬅ 上一步", use_container_width=True):
                st.session_state.tutorial_step = 1
                st.rerun()
            if c2.button("下一步：省錢決策室 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 3
                st.rerun()

        # 🔵 Step 3: 決策與啟動 (Actionable Insights)
        elif step == 3:
            st.markdown("### 💰 錢要花在刀口上")
            st.markdown("##### —— 讓數據轉化為您的被動收入")
            
            st.success("""
            **我們不只畫圖表，我們直接給答案。**
            系統內建的「決策分析室」將為您全天候監控：
            """)
            
            st.markdown("""
            * **💸 費率裁判官**：
                自動平行計算「累進費率」與「時間電價」的成本差異。
                *當我們發現您換費率一年能省下 $3,000 元時，我們會主動通知您。*
                
            * **🚨 異常偵探**：
                當您的用電行為偏離常軌（例如冰箱門沒關、冷氣異常耗電），AI 會立即標示紅區警報。
            """)
            
            st.write("#")
            st.divider()
            
            c1, c2 = st.columns([1, 2])
            if c1.button("⬅ 上一步", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()
            
            # 啟動按鈕
            if c2.button("🚀 啟動系統監控", type="primary", use_container_width=True):
                if st.session_state.get("app_ready", False):
                    st.session_state.page = "home"
                    st.session_state.tutorial_finished = True
                    st.rerun()
                else:
                    st.session_state.tutorial_step = "loading"
                    st.rerun()

    # 底部進度條 (跨欄顯示)
    st.write("---")
    st.progress(step / 3)
    st.caption(f"系統導覽進度：{step} / 3")


def show_fullscreen_loading():
    """
    【Loading 模式】全螢幕動圖 + 左下角進度條 + 真實運算
    """
    loading_anim = load_lottiefile("lottiefiles/loading_animation.json")
    
    placeholder_lottie = st.empty()
    placeholder_bar = st.empty()

    # A. 全螢幕動圖 (系統運作中)
    with placeholder_lottie:
        _, c_center, _ = st.columns([1, 2, 1])
        with c_center:
            st.write("#")
            st.write("#")
            if loading_anim:
                st_lottie(loading_anim, height=400, key="full_loader", speed=1)
            else:
                st.spinner("系統啟動中...")

    # B. 進度條邏輯
    progress_text = "正在載入 AI 模型架構..."
    my_bar = placeholder_bar.progress(0, text=progress_text)

    # 模擬前段加載
    for percent_complete in range(0, 30, 5):
        time.sleep(0.05)
        my_bar.progress(percent_complete, text="正在同步歷史氣象資料庫...")

    # 真實運算
    try:
        my_bar.progress(40, text="啟動 Hybrid LSTM 雙核心預測引擎...")
        
        # 執行核心運算
        res_df, hist_df = load_resources_and_predict() 
        
        st.session_state.prediction_result = res_df
        st.session_state.current_data = hist_df
        st.session_state.app_ready = True
        
    except Exception as e:
        st.error(f"啟動失敗: {e}")
        st.stop()

    # 模擬後段渲染
    for percent_complete in range(60, 101, 10):
        time.sleep(0.05)
        my_bar.progress(percent_complete, text="數據視覺化渲染完成！")
    
    time.sleep(0.5)

    # 跳轉
    st.session_state.page = "home"
    st.session_state.tutorial_finished = True
    st.rerun()