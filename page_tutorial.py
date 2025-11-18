#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:11:00 2025

@author: zhouting
"""

import streamlit as st
from streamlit_lottie import st_lottie

# 從 app_utils 匯入 Lottie 載入函式
from app_utils import load_lottiefile

def show_tutorial_page():
    """
    顯示全螢幕的「首次使用教學導覽」
    """
    
    # 初始化教學步驟
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # 使用欄位將所有內容置中
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # 根據目前的步驟顯示不同內容
        step = st.session_state.tutorial_step
        
        # --- 步驟 1: 歡迎動畫 ---
        if step == 1:
            st.title("歡迎使用 💡 智慧電能管家")
            
            # 重用側邊欄的 Logo 動畫
            lottie_logo = load_lottiefile("lottiefiles/intelligent_tour_guide_robot.json")
            if lottie_logo:
                st_lottie(lottie_logo, speed=1, loop=True, quality="high", height=300, key="tutorial_logo")
            
            st.markdown("### 我將帶您快速瀏覽 App 的三大核心功能。")
            st.markdown("準備好了嗎？")
            st.divider()
            
            # 按鈕佈局
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("略過導覽"):
                    st.session_state.tutorial_complete = True
                    st.rerun()
            with btn_col2:
                if st.button("下一步", type="primary"):
                    st.session_state.tutorial_step = 2
                    st.rerun()

        # --- 步驟 2: 介紹主頁 ---
        elif step == 2:
            st.title("🏠 認識主頁")
            
            try:
                st.image("tutorial_image/tutorial_2_home.png")
            except Exception as e:
                st.error(f"無法載入圖片: tutorial_image/tutorial_2_home.png\n{e}")

            st.markdown("### 1. 關鍵資訊總覽")
            st.markdown("「主頁」是您的總覽中心。您可以在這裡快速查看**本週用電狀態**（良好、普通或警示），以及**今日**、**本週**、**本月**的累積用電。")
            
            st.markdown("### 2. 預算與目標")
            st.markdown("您也可以在主頁**設定您的電費目標**，並即時查看**預估電費**與**剩餘預算**。")
            st.divider()

            # 按鈕佈局
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col1:
                if st.button("上一步"):
                    st.session_state.tutorial_step = 1
                    st.rerun()
            with btn_col2:
                if st.button("略過導覽"):
                    st.session_state.tutorial_complete = True
                    st.rerun()
            with btn_col3:
                if st.button("下一步", type="primary"):
                    st.session_state.tutorial_step = 3
                    st.rerun()

        # --- 步驟 3: 介紹儀表板 ---
        elif step == 3:
            st.title("📈 認識儀表板")
            
            try:
                st.image("tutorial_image/tutorial_3_dashboard.png")
            except Exception as e:
                st.error(f"無法載入圖片: tutorial_image/tutorial_3_dashboard.png\n{e}")

            st.markdown("### 深入分析您的數據")
            st.markdown("「儀表板」提供最詳細的數據圖表。您可以查看**即時用電**、**最近 7 天**的詳細用電曲線，以及**近 30 天**的尖峰/離峰用電分佈。")
            st.divider()

            # 按鈕佈局
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col1:
                if st.button("上一步"):
                    st.session_state.tutorial_step = 2
                    st.rerun()
            with btn_col2:
                if st.button("略過導覽"):
                    st.session_state.tutorial_complete = True
                    st.rerun()
            with btn_col3:
                if st.button("下一步", type="primary"):
                    st.session_state.tutorial_step = 4
                    st.rerun()
        
        # --- 步驟 4: 介紹 AI 分析室 ---
        elif step == 4:
            st.title("🔬 認識 AI 決策分析室")
            
            try:
                st.image("tutorial_image/tutorial_4_analysis.png")
            except Exception as e:
                st.error(f"無法載入圖片: tutorial_image/tutorial_4_analysis.png\n{e}")

            st.markdown("### 讓 AI 成為您的專屬顧問")
            st.markdown("「AI 決策分析室」是您的大腦。在這裡您可以**預測未來用電**、**比較電價方案**（找出最省錢的方式），並自動**偵測異常用電**，最後取得客製化的**節能建議**。")
            st.divider()

            # 按鈕佈局
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("上一步"):
                    st.session_state.tutorial_step = 3
                    st.rerun()
            with btn_col2:
                if st.button("✨ 開始使用！", type="primary"):
                    st.session_state.tutorial_complete = True
                    st.rerun()