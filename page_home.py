import streamlit as st
from datetime import timedelta
import pandas as pd

# 匯入共用函式
# 請確保 app_utils 中包含 analyze_pricing_plans
from app_utils import load_data, get_core_kpis, analyze_pricing_plans

def show_home_page():
    """
    顯示「主頁」總覽
    包含：
    1. 關鍵資訊 KPI
    2. 自動化電價分析 (取代原本的預算目標)
    3. 功能導覽
    """
    st.title("💡 智慧電能管家總覽")
    
    # --- 1. 載入數據並計算 KPI ---
    df_history = load_data()
    
    # 確保數據存在才能繼續
    if df_history is None or df_history.empty:
        st.warning("🔌 尚無歷史資料，請先確認資料庫或匯入數據。")
        return

    kpis = get_core_kpis(df_history)

    if not kpis['status_data_available']:
        st.warning("🔌 歷史資料不足 (需 14 天) 或載入失敗，無法顯示完整總覽。")
        st.info("請檢查您的數據檔案，或等待數據收集。")
    else:
        # --- 顯示核心 KPI (保留原設計) ---
        st.markdown("### 關鍵資訊總覽")
        
        # 用電狀態判斷
        if kpis['weekly_delta_percent'] > 10: 
            status_display = f":red[(｡ ́︿ ̀｡) 偏高]"
        elif kpis['weekly_delta_percent'] < -10: 
            status_display = ":green[(๑•̀ㅂ•́)و✧ 良好]"
        else: 
            status_display = ":blue[(・-・) 普通]"
            
        st.subheader(f"您本週的用電狀態： {status_display}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("本週累積用電 (近 7 天)", f"{kpis['kwh_last_7_days']:.2f} kWh")
        col2.metric("今日累積用電", f"{kpis['kwh_today_so_far']:.2f} kWh")
        col3.metric("本月累積用電 (至今)", f"{kpis['kwh_this_month_so_far']:.1f} kWh")

    st.divider()

    # --- 2. AI 電價方案分析 (取代原本的預算區塊) ---
    # 樣式參考您提供的第二張截圖
    st.markdown("### 💰 您的最佳電價方案分析")
    
    # 自動設定分析範圍：資料庫最後一天 往前推 30 天
    last_date = df_history.index.max().date()
    start_date = last_date - timedelta(days=29)
    
    # 顯示分析區間
    st.caption(f"📅 自動分析區間：{start_date} 至 {last_date} (最近 30 天)")

    # 擷取這段時間的資料
    analysis_df = df_history.loc[start_date.strftime('%Y-%m-%d'):last_date.strftime('%Y-%m-%d')].copy()

    if analysis_df.empty:
        st.info("資料不足，無法進行電價試算。")
    else:
        try:
            # 呼叫分析函式
            results, _ = analyze_pricing_plans(analysis_df)
            
            cost_prog = results['cost_progressive']
            cost_tou = results['cost_tou']
            total_kwh = results['total_kwh']
            difference = cost_prog - cost_tou # 正值代表 TOU 比較便宜 (累進 - TOU > 0)

            # --- 顯示比較結果 (兩欄佈局) ---
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(
                    label="方案一：累進電價 (標準)",
                    value=f"{cost_prog:,.0f} 元",
                    help="一般住宅預設的電價計算方式"
                )
            
            with res_col2:
                # 設定顏色：如果 TOU 比較便宜顯示綠色，比較貴顯示紅色/灰色
                delta_val = None
                delta_color = "off"
                
                if difference > 0: 
                    delta_val = f"省 {difference:,.0f} 元"
                    delta_color = "inverse" # 綠色
                else:
                    delta_val = f"貴 {abs(difference):,.0f} 元"
                    delta_color = "normal" # 顯示紅色 (代表不划算)

                st.metric(
                    label="方案二：簡易型時間電價 (TOU)",
                    value=f"{cost_tou:,.0f} 元",
                    delta=delta_val,
                    delta_color=delta_color,
                    help="區分尖峰與離峰時段的電價方案"
                )

            # --- 顯示分析建議框 (依照截圖風格) ---
            # 增加一些間距
            st.markdown("####") 
            
            if difference > 0:
                # TOU 比較便宜 -> 建議切換
                st.success(
                    f"#### 💡 分析建議：(๑•̀ㅂ•́)و✧ \n"
                    f"在此期間，若選用 **「簡易型時間電價 (TOU)」**，預計可 **節省 {difference:,.0f} 元**！\n\n"
                    f"您的離峰用電比例較高，非常適合申請時間電價。"
                )
            else:
                # 累進比較便宜 -> 保持現狀
                st.warning(
                    f"#### 💡 分析建議：(・ω・) \n"
                    f"在此期間，選用 **「累進電價 (標準)」** 較為划算（可省 {abs(difference):,.0f} 元）。\n\n"
                    f"若想改用時間電價，建議您嘗試將高耗電家電（洗衣機、烘衣機）移至離峰時段使用。"
                )

        except Exception as e:
            st.error(f"分析時發生錯誤: {e}")
            st.caption("請檢查 app_utils.py 中的 analyze_pricing_plans 函式是否正常運作。")

    st.divider()

    # --- 3. 功能導覽 (保留原設計) ---
    st.markdown("### 功能導覽")
    
    st.subheader("📈 用電儀表板")
    st.markdown("- 查看 **即時用電**、昨日同期比較、以及近期的詳細用電曲線。")
    
    st.subheader("🔬 AI 決策分析室")
    st.markdown("- **AI 用電預測**：預測未來 15 分鐘用電。\n- **AI 電價分析器**：更詳細的歷史回測與圖表分析。\n- **AI 異常偵測**：找出異常耗電時段。")