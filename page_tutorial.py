# page_tutorial.py
import streamlit as st
import time
import threading
import traceback # [修改點 1] 引入 traceback 來獲取最完整的錯誤追蹤訊息
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
        self.error_msg = None # [修改點 2] 新增 error_msg 屬性，用來將錯誤安全傳遞給主執行緒

    def run_task(self):
        self.is_running = True
        try:
            res_df, hist_df = load_resources_and_predict()
            self.result = res_df
            self.history = hist_df
            self.is_done = True
        except Exception as e:
            # [修改點 3] 捕捉完整錯誤軌跡，不僅 print 出來，也存入 error_msg
            error_details = traceback.format_exc()
            print(f"Background Task Error:\n{error_details}")
            self.error_msg = error_details 
            self.is_done = True 
        finally:
            self.is_running = False

def init_worker():
    if 'bg_worker' not in st.session_state:
        st.session_state.bg_worker = BackgroundWorker()

def start_background_thread():
    init_worker()
    worker = st.session_state.bg_worker
    if not worker.is_done and not worker.is_running and not st.session_state.get("app_ready", False):
        t = threading.Thread(target=worker.run_task)
        t.start()

# ==========================================
# 📖 導覽頁面主邏輯
# ==========================================
def show_tutorial_page():
    init_worker()
    start_background_thread()

    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # Loading 模式攔截
    if st.session_state.tutorial_step == "loading":
        show_fullscreen_loading()
        return

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
        
        # ==========================================
        # Step 1: 核心價值
        # ==========================================
        if st.session_state.tutorial_step == 1:
            st.markdown("### ⚡ 歡迎啟動「智慧電能管家」")
            st.markdown("##### —— 兼具節能與預算的得力助手")
            
            st.info("""
            **「為什麼帳單總是遲到的壞消息？」**
            傳統電表只能紀錄過去，讓您在月底面對帳單時措手不及。
            我們不同，我們是一套 **「具有預知能力」** 的決策系統。
            """)
            
            st.markdown("""
            **我們的三大核心價值：**
            1.  🔮 **預知未來**：提前 30 天告訴您本期帳單金額。
            2.  🛡️ **主動示警**：即時監控每一度電，在超支前發出預警。
            3.  🧠 **決策大腦**：不只給數據，更直接告訴您「怎麼省」。
            """)
            
            st.write("#")
            # 🌟 核心修改：建立兩欄按鈕，左邊是跳過，右邊是下一步
            c1, c2 = st.columns([1, 2]) 
            
            if c1.button("跳過導覽", use_container_width=True):
                st.session_state.tutorial_step = "loading" # 直接進入資料載入階段
                st.rerun()

            if c2.button("下一步：解密 AI 核心技術 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()

        # ==========================================
        # Step 2: 技術與數據來源 (修正技術用語)
        # ==========================================
        elif st.session_state.tutorial_step == 2:
            st.markdown("### 🧠 獨家 Hybrid AI 架構")
            st.markdown("""
            市面上的 APP 多顯示歷史，**我們是少數能推算未來的系統。**
            我們採用 **「長短週期混合運算架構」** 來確保預測的穩定性：
            """)
            
            with st.expander("🔴 紅線：短期高敏銳度模型 (LSTM)", expanded=True):
                st.write("針對 **未來 48 小時** 進行高解析度運算。精準捕捉家電開啟的瞬間波動，反映您的真實作息。")
                
            with st.expander("🟠 橘線：長期趨勢推估系統", expanded=True):
                st.write("引入 **歷史氣候大數據**，對照過去三年的氣溫模型，推算直到 **結算日** 的最終帳單趨勢。")

            st.markdown("---")
            st.caption("📡 數據來源：本系統對接台電官方 AMI 智慧電表資料庫 (service.taipower.com.tw)，雖受限於硬體傳輸有約 1 小時延遲，但能確保數據權威性。")

            st.write("#")
            c1, c2 = st.columns([1, 2])
            if c1.button("⬅ 上一步", use_container_width=True):
                st.session_state.tutorial_step = 1
                st.rerun()
            if c2.button("下一步：我們 vs 台電官方 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 3
                st.rerun()

        # ==========================================
        # Step 3: 競品分析 (我們 vs 台電)
        # ==========================================
        elif st.session_state.tutorial_step == 3:
            st.markdown("### ⚔️ 我們與官方 App 有何不同？")
            st.markdown("##### —— 後照鏡 vs GPS 導航")
            
            st.write("這不是要取代台電 App，而是為您加上一顆**預知大腦**。")

            col_official, col_us = st.columns(2)
            
            with col_official:
                st.markdown("#### 🏛️ 台電官方 App")
                st.warning("功能：數位記帳本")
                st.markdown("""
                * ❌ **只看過去**：告訴你昨天花了多少錢。
                * ❌ **被動告知**：當你發現電費過高時，**錢已經扣掉了**。
                * ❌ **單純紀錄**：給你數據，但沒告訴你該怎麼辦。
                """)
                
            with col_us:
                st.markdown("#### ⚡ 智慧電能管家")
                st.success("功能：AI 理財顧問")
                st.markdown("""
                * ✅ **推算未來**：告訴你**月底將會花多少錢**。
                * ✅ **主動示警**：在超支發生 **兩週前** 就發出提醒。
                * ✅ **決策輔助**：直接計算「換什麼費率」最省錢。
                """)

            st.write("#")
            st.divider()
            c1, c2 = st.columns([1, 2])
            if c1.button("⬅ 上一步", use_container_width=True):
                st.session_state.tutorial_step = 2
                st.rerun()
            if c2.button("下一步：省錢決策室 ➔", type="primary", use_container_width=True):
                st.session_state.tutorial_step = 4
                st.rerun()

        # ==========================================
        # Step 4: 決策與啟動
        # ==========================================
        elif st.session_state.tutorial_step == 4:
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
                st.session_state.tutorial_step = 3
                st.rerun()
            
            init_worker()
            worker = st.session_state.bg_worker
            
            if worker.is_done:
                btn_txt = "數據已準備就緒，進入控制台！ ➔"
            else:
                btn_txt = "🚀 啟動系統監控"

            if c2.button(btn_txt, type="primary", use_container_width=True):
                st.session_state.tutorial_step = "loading"
                st.rerun()

    st.write("---")
    current_step = st.session_state.tutorial_step if isinstance(st.session_state.tutorial_step, int) else 4
    st.progress(current_step / 4)
    st.caption(f"系統導覽進度：{current_step} / 4")


def show_fullscreen_loading():
    """
    【Loading 模式】
    """
    loading_anim = load_lottiefile("lottiefiles/loading_animation.json")
    
    placeholder_lottie = st.empty()
    placeholder_bar = st.empty()

    with placeholder_lottie:
        _, c_center, _ = st.columns([1, 2, 1])
        with c_center:
            st.write("#")
            st.write("#")
            if loading_anim:
                st_lottie(loading_anim, height=400, key="full_loader", speed=1)
            else:
                st.spinner("系統啟動中...")

    my_bar = placeholder_bar.progress(0, text="正在建立與 AI 核心的連線...")
    
    init_worker()
    worker = st.session_state.bg_worker
    
    if not worker.is_running and not worker.is_done:
        start_background_thread() 
        time.sleep(1)

    progress = 0
    wait_cycles = 0
    
    while not worker.is_done:
        if progress < 90:
            progress += 1
        else:
            time.sleep(0.1)
        wait_cycles += 1
        
        if wait_cycles < 20:
            status_text = f"正在載入歷史氣象資料... ({progress}%)"
        elif wait_cycles < 50:
            status_text = f"啟動 Hybrid AI 雙核心運算引擎... ({progress}%)"
        else:
            status_text = f"正在進行最後的數據整合... ({progress}%)"
            
        my_bar.progress(progress, text=status_text)
        time.sleep(0.1)
        
        if wait_cycles > 600: 
            st.error("連線逾時，請重新整理頁面。")
            st.stop()

    my_bar.progress(100, text="數據視覺化渲染完成！")
    time.sleep(0.5)

    # [修改點 4] UI 迴圈結束後，第一時間攔截並顯示背景任務的錯誤！
    if worker.error_msg is not None:
        st.session_state.error_msg = worker.error_msg # 依要求存入 session_state
        st.error("🚨 **AI 核心運算發生嚴重錯誤！**")
        with st.expander("點此查看詳細錯誤日誌 (開發者專用)", expanded=True):
            st.code(worker.error_msg, language="python")
        
        if st.button("🔄 重置並退回導覽頁"):
            # 清除壞掉的 worker 讓它有機會重來
            del st.session_state.bg_worker 
            st.session_state.tutorial_step = 4
            st.rerun()
            
        st.stop() # 強制停止渲染，絕對不跳轉到空的主頁

    # 若無錯誤，則正常載入並跳轉
    if worker.result is not None:
        st.session_state.prediction_result = worker.result
        st.session_state.current_data = worker.history
        st.session_state.app_ready = True
    
    st.session_state.page = "home"
    st.session_state.tutorial_finished = True
    st.rerun()