# page_tutorial.py
import streamlit as st
import time
import threading
from streamlit_lottie import st_lottie
from app_utils import load_lottiefile
from model_service import load_resources_and_predict

# ==========================================
# 🧵 背景工作執行緒 (Background Worker)
# ==========================================
class BackgroundWorker:
    def __init__(self):
        self.result = None
        self.history = None
        self.is_done = False
        self.is_running = False

    def run_task(self):
        self.is_running = True
        try:
            # 執行耗時運算
            res_df, hist_df = load_resources_and_predict()
            self.result = res_df
            self.history = hist_df
            self.is_done = True
        except Exception as e:
            print(f"Background Task Error: {e}")
            self.is_done = True # 失敗也要標記完成以免卡死
        finally:
            self.is_running = False

# 初始化 worker
if 'bg_worker' not in st.session_state:
    st.session_state.bg_worker = BackgroundWorker()

def start_background_thread():
    """啟動背景執行緒"""
    worker = st.session_state.bg_worker
    if not worker.is_done and not worker.is_running and not st.session_state.get("app_ready", False):
        t = threading.Thread(target=worker.run_task)
        t.start()

# ==========================================
# 📖 導覽頁面主邏輯
# ==========================================
def show_tutorial_page():
    
    # 1. 一進來就啟動背景運算 (Non-blocking)
    start_background_thread()

    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # 2. 模式切換：如果是 loading 狀態，直接進入全螢幕載入函式
    if st.session_state.tutorial_step == "loading":
        show_fullscreen_loading()
        return

    # 3. 一般導覽 UI
    st.write("#")
    col_robot, col_content = st.columns([1.2, 2.0], gap="large")

    # --- 左側：AI 導遊 ---
    with col_robot:
        st.write("##")
        robot_anim = load_lottiefile("lottiefiles/Intelligent_tour_guide_robot.json")
        if robot_anim:
            st_lottie(robot_anim, speed=1, loop=True, height=350, key=f"robot_step_{st.session_state.tutorial_step}")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=200)

    # --- 右側：內容 ---
    with col_content:
        
        # Step 1: 歡迎
        if st.session_state.tutorial_step == 1:
            st.markdown("### ⚡ 歡迎啟動「智慧電能管家」")
            st.markdown("##### —— 您的家庭能源首席財務官")
            
            st.info("""
            **「為什麼帳單總是遲到的壞消息？」**
            傳統電表只能紀錄過去，讓您在月底面對帳單時措手不及。
            我們不同，我們是一套 **「具有預知能力」** 的決策系統。
            """)
            
            st.write("#")
            if st.button("下一步：解密 AI 核心技術 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()

        # Step 2: 技術 (已修正誇大文案)
        elif st.session_state.tutorial_step == 2:
            st.markdown("### 🧠 獨家 Hybrid AI 雙軌預測技術")
            st.markdown("##### —— 結合深度學習與氣候模擬的完全體")
            
            st.markdown("""
            市面上的電量 APP 大多只能顯示歷史，**我們是唯一能模擬未來的系統。**
            為了達到 95% 以上的準確度，我們同時運行兩套神經網路：
            """)
            
            with st.expander("🔴 紅線：LSTM 短期高精準模型", expanded=True):
                # [修正] 將 "毫秒級" 改為 "小時級精細運算"
                st.write("""
                專注於 **未來 48 小時** 的**小時級精細運算**。
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
        elif st.session_state.tutorial_step == 3:
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
            
            # 判斷狀態，給予使用者即時回饋
            worker = st.session_state.bg_worker
            if worker.is_done:
                btn_txt = "數據已準備就緒，進入控制台！ ➔"
            else:
                btn_txt = "🚀 啟動系統監控"

            if c2.button(btn_txt, type="primary", use_container_width=True):
                st.session_state.tutorial_step = "loading"
                st.rerun()

    st.write("---")
    st.progress(st.session_state.tutorial_step / 3 if isinstance(st.session_state.tutorial_step, int) else 1.0)
    
    # Debug 狀態顯示 (可選)
    # st.caption(f"Background Status: {'Running' if st.session_state.bg_worker.is_running else 'Done' if st.session_state.bg_worker.is_done else 'Idle'}")


def show_fullscreen_loading():
    """
    【Loading 模式】死守迴圈，直到後台運算完成
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

    # 2. 進度條初始化
    my_bar = placeholder_bar.progress(0, text="正在建立與 AI 核心的連線...")
    
    # 3. 確保背景執行緒真的有在跑 (防呆機制)
    worker = st.session_state.bg_worker
    if not worker.is_running and not worker.is_done:
        start_background_thread() # 如果意外沒跑，這裡強制啟動
        time.sleep(1) # 給它一點時間啟動

    # 4. 【關鍵】真實等待迴圈 (Real Wait Loop)
    # 我們讓進度條在 0% ~ 90% 之間反覆跑，直到 worker.is_done 變成 True
    progress = 0
    wait_cycles = 0
    
    while not worker.is_done:
        # 讓進度條有在前進的感覺，但不要到 100%
        if progress < 90:
            progress += 1
        else:
            # 如果卡在 90% 太久，稍微閃爍一下文字讓使用者知道還在活著
            pass
            
        wait_cycles += 1
        
        # 動態文案
        if wait_cycles < 20:
            status_text = f"正在載入歷史氣象資料... ({progress}%)"
        elif wait_cycles < 50:
            status_text = f"啟動 LSTM 雙核心運算引擎... ({progress}%)"
        else:
            status_text = f"正在進行最後的數據整合... ({progress}%)"
            
        my_bar.progress(progress, text=status_text)
        time.sleep(0.1) # 每 0.1 秒檢查一次
        
        # 安全機制：如果卡太久 (例如超過 60秒)，可能出錯了，強制跳出
        if wait_cycles > 600:
            st.error("連線逾時，請重新整理頁面。")
            st.stop()

    # 5. 運算完成！衝刺最後 10%
    my_bar.progress(100, text="數據視覺化渲染完成！")
    time.sleep(0.5)

    # 6. 取出結果並存入 Session
    if worker.result is not None:
        st.session_state.prediction_result = worker.result
        st.session_state.current_data = worker.history
        st.session_state.app_ready = True
    
    # 7. 跳轉首頁
    st.session_state.page = "home"
    st.session_state.tutorial_finished = True
    st.rerun()