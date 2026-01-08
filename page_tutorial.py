# page_tutorial.py
import streamlit as st
import time
import threading
from streamlit_lottie import st_lottie
from app_utils import load_lottiefile
from model_service import load_resources_and_predict

# 用於在執行緒中傳遞結果的容器
# 注意：Streamlit 的 session_state 在執行緒中不一定安全，所以我們用全域變數或閉包來處理
class BackgroundWorker:
    def __init__(self):
        self.result = None
        self.history = None
        self.is_done = False
        self.is_running = False

    def run_task(self):
        self.is_running = True
        print("🧵 [Thread] Background task started...")
        try:
            # 這裡執行耗時運算
            res_df, hist_df = load_resources_and_predict()
            self.result = res_df
            self.history = hist_df
            self.is_done = True
            print("🧵 [Thread] Background task finished!")
        except Exception as e:
            print(f"🧵 [Thread] Error: {e}")
            self.is_done = True # 即使失敗也標記完成，以免無窮等待
        finally:
            self.is_running = False

# 初始化 worker 到 session_state (確保跨頁面存活)
if 'bg_worker' not in st.session_state:
    st.session_state.bg_worker = BackgroundWorker()

def start_background_thread():
    """啟動背景執行緒 (如果還沒跑的話)"""
    worker = st.session_state.bg_worker
    # 只有在「沒做完」且「沒在跑」且「APP還沒準備好」的時候才啟動
    if not worker.is_done and not worker.is_running and not st.session_state.get("app_ready", False):
        t = threading.Thread(target=worker.run_task)
        t.start()

def show_tutorial_page():
    """
    【故事模式】背景多執行緒運算 + 前台流暢導覽
    """
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # ==========================================
    # 🚀 啟動背景引擎 (不會卡住畫面)
    # ==========================================
    start_background_thread()

    # ==========================================
    # 🎬 模式切換邏輯
    # ==========================================
    step = st.session_state.tutorial_step

    # 如果進入 "loading" 模式
    if step == "loading":
        show_fullscreen_loading()
        return

    # ==========================================
    # 📖 一般導覽模式 (Step 1~3)
    # ==========================================
    st.write("#")
    
    # 左右佈局：機器人 vs 內容
    col_robot, col_content = st.columns([1.2, 2.0], gap="large")

    with col_robot:
        st.write("##")
        robot_anim = load_lottiefile("lottiefiles/Intelligent_tour_guide_robot.json")
        if robot_anim:
            st_lottie(robot_anim, speed=1, loop=True, height=350, key=f"robot_step_{step}")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=200)

    with col_content:
        # Step 1: 歡迎
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

        # Step 2: 技術
        elif step == 2:
            st.markdown("### 🧠 獨家 Hybrid AI 雙軌預測技術")
            st.markdown("##### —— 結合深度學習與氣候模擬的完全體")
            
            st.markdown("""
            市面上的電量 APP 大多只能顯示歷史，**我們是唯一能模擬未來的系統。**
            為了達到 95% 以上的準確度，我們同時運行兩套神經網路：
            """)
            
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

        # Step 3: 決策
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
            
            # --- 判斷按鈕文字 ---
            worker = st.session_state.bg_worker
            if worker.is_done:
                btn_txt = "數據已準備就緒，進入控制台！ ➔"
                btn_help = "後台模型已載入完成，可立即使用"
            else:
                btn_txt = "🚀 啟動系統監控"
                btn_help = "點擊後將進入載入畫面"

            if c2.button(btn_txt, type="primary", use_container_width=True, help=btn_help):
                # 這裡統一都進 loading，由 loading 去判斷是否要秒過
                st.session_state.tutorial_step = "loading"
                st.rerun()

    st.write("---")
    st.progress(step / 3)
    # 顯示隱藏的後台狀態給你看 (Debug用，實際上使用者不會注意到)
    status_icon = "🟢" if st.session_state.bg_worker.is_done else "🟡" if st.session_state.bg_worker.is_running else "⚪"
    st.caption(f"系統導覽進度：{step} / 3 | 後台引擎狀態：{status_icon}")


def show_fullscreen_loading():
    """
    【Loading 模式】全螢幕動圖 + 真實運算等待
    """
    loading_anim = load_lottiefile("lottiefiles/loading_animation.json")
    
    placeholder_lottie = st.empty()
    placeholder_bar = st.empty()

    # 1. 顯示動圖
    with placeholder_lottie:
        _, c_center, _ = st.columns([1, 2, 1])
        with c_center:
            st.write("#")
            st.write("#")
            if loading_anim:
                st_lottie(loading_anim, height=400, key="full_loader", speed=1)
            else:
                st.spinner("系統啟動中...")

    # 2. 檢查或等待背景執行緒
    worker = st.session_state.bg_worker
    
    # 如果還沒開始跑 (防呆)，就現在跑 (同步阻斷式)
    if not worker.is_running and not worker.is_done and not st.session_state.get("app_ready", False):
        worker.run_task() # 這會卡住畫面直到完成

    # 如果正在跑，就等待它完成
    progress_text = "正在整合 AI 運算結果..."
    my_bar = placeholder_bar.progress(0, text=progress_text)
    
    # 進入等待迴圈
    for i in range(100):
        if worker.is_done:
            my_bar.progress(100, text="載入完成！")
            break
        
        # 模擬進度條慢慢跑 (讓使用者知道沒當機)
        # 進度條最多跑到 90%，剩下 10% 等真正做完才跑
        current_progress = min(i * 2, 90)
        my_bar.progress(current_progress, text="正在同步歷史氣象資料與 LSTM 權重...")
        time.sleep(0.1) # 每 0.1 秒檢查一次狀態

    # 3. 取出結果並存入 session
    if worker.result is not None:
        st.session_state.prediction_result = worker.result
        st.session_state.current_data = worker.history
        st.session_state.app_ready = True
    
    time.sleep(0.5)

    # 4. 跳轉首頁
    st.session_state.page = "home"
    st.session_state.tutorial_finished = True
    st.rerun()