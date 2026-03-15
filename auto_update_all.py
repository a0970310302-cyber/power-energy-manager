import requests
import pandas as pd
import numpy as np
import os
import urllib3
import shutil
import logging
from datetime import datetime

# ==========================================
# ⚙️ 雲端環境變數讀取 (GitHub Secrets)
# ==========================================
# 使用 os.environ.get 抓取秘密資訊，若抓不到則留空 (供除錯判斷)
JSON_SOURCE_URL = os.environ.get("JSON_SOURCE_URL")
PANTRY_ID = os.environ.get("PANTRY_ID")
WEATHER_INDEX_URL = os.environ.get("WEATHER_INDEX_URL")

# 常規參數 (不具敏感性，可直接寫在程式裡)
PANTRY_BASKET = "2026-q1"
PANTRY_URL = f"https://getpantry.cloud/apiv1/pantry/{PANTRY_ID}/basket/{PANTRY_BASKET}"
MASTER_FILE = "final_training_data_with_humidity.csv"
LOG_FILE = "log.txt"

# 設定 Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ==========================================
# 🚀 階段一：雲端數據同步 (Cloud Sync)
# ==========================================
def sync_cloud_to_pantry():
    logging.info("="*50)
    logging.info("🚩 [Step 1] 啟動雲端數據格式化與同步")
    logging.info("="*50)
    
    # 安全檢查：確保 Secrets 有正確讀取
    if not JSON_SOURCE_URL or not PANTRY_ID:
        logging.error("❌ [錯誤] 找不到環境變數 (Secrets)，請檢查 GitHub 設定。")
        return False
    
    try:
        logging.info("📡 正在請求 JsonStorage...")
        source_res = requests.get(JSON_SOURCE_URL)
        raw_json = source_res.json()
        data_block = raw_json.get("data", {})
        
        formatted_new_data = {}
        for date_key, date_content in data_block.items():
            if not date_key.startswith("202"): continue
            data_list = date_content.get("listAMIBase15MinData", [])
            logging.info(f"🔍 掃描到日期 {date_key}: 包含 {len(data_list)} 筆原始數據")
            for item in data_list:
                time_str = item.get("time")
                power_val = item.get("power")
                formatted_new_data[f"{date_key} {time_str}:00"] = power_val
        
        if not formatted_new_data:
            logging.error("❌ 來源無效，同步終止。")
            return False

        p_res = requests.get(PANTRY_URL)
        p_data = p_res.json() if p_res.status_code == 200 else {}
        p_data.pop('_metadata', None)
        
        old_count = len(p_data)
        p_data.update(formatted_new_data)
        logging.info(f"🔄 數據合併完成: Pantry 原有 {old_count} 筆 -> 更新後總計 {len(p_data)} 筆")
        
        requests.post(PANTRY_URL, json=p_data)
        logging.info("✅ [Step 1 成功] 雲端同步達成。")
        return True
    except Exception as e:
        logging.error(f"❌ [Step 1 失敗] 發生異常: {str(e)}")
        return False

# ==========================================
# 🚀 階段二：本地 CSV 增量更新 (Local Update)
# ==========================================
def update_local_csv():
    logging.info("\n" + "="*50)
    logging.info("🚩 [Step 2] 啟動本地 CSV 增量更新")
    logging.info("="*50)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    if not os.path.exists(MASTER_FILE):
        logging.error(f"❌ 找不到目標檔案 {MASTER_FILE}")
        return

    # 1. 備份與讀取
    shutil.copy(MASTER_FILE, f"{MASTER_FILE}.bak")
    logging.info(f"📦 已建立安全性備份: {MASTER_FILE}.bak")
    
    df_master = pd.read_csv(MASTER_FILE)
    df_master['dt_obj'] = pd.to_datetime(df_master['datetime'], format='mixed')
    last_dt = df_master['dt_obj'].max()
    logging.info(f"📅 CSV 目前最後紀錄時間點: {last_dt}")

    # 2. 數據聚合 (修正警告: 使用 '1h')
    try:
        p_data = requests.get(PANTRY_URL).json()
        logging.info(f"📡 已從 Pantry 載入 {len(p_data)} 筆數據進行聚合分析")
        
        df_p = pd.DataFrame(list(p_data.items()), columns=['datetime', 'power'])
        df_p['datetime'] = pd.to_datetime(df_p['datetime'], errors='coerce')
        df_p = df_p.dropna(subset=['datetime']).set_index('datetime')
        df_p['power'] = pd.to_numeric(df_p['power'], errors='coerce')

        # 使用 '1h' 替代 '1H' 以消除 FutureWarning
        hourly_p = df_p['power'].resample('1h').sum()
        hourly_c = df_p['power'].resample('1h').count()
        df_new_api = pd.DataFrame(index=hourly_p.index)
        df_new_api['power'] = hourly_p.values
        df_new_api['isMssingData'] = ((4 - hourly_c) / 4).clip(lower=0).values
        
        # 🚩 增量過濾
        df_new_inc = df_new_api[df_new_api.index > last_dt].copy()
        if df_new_inc.empty:
            logging.info("✨ 檢查完畢：CSV 已經包含所有最新資料，無需更新。")
            return
        logging.info(f"📝 發現 {len(df_new_inc)} 小時的新數據準備寫入...")
    except Exception as e:
        logging.error(f"❌ [Step 2 數據處理失敗]: {str(e)}")
        return

    # 3. 獲取天氣 (讀取 Secret: WEATHER_INDEX_URL)
    weather_map = {}
    try:
        logging.info("🌤️ 正在同步對應時段的天氣資訊...")
        if not WEATHER_INDEX_URL:
            logging.warning("⚠️ 找不到 WEATHER_INDEX_URL，將跳過天氣填補。")
        else:
            w_idx = requests.get(WEATHER_INDEX_URL).json().get('items', {})
            for date_str, info in w_idx.items():
                if pd.to_datetime(date_str) < last_dt.normalize(): continue
                day_res = requests.get(info['uri']).json()
                rows = day_res.get('days', {}).get(date_str, {}).get('rows', [])
                for r in rows:
                    weather_map[pd.to_datetime(r[0])] = (r[1], r[2])
            logging.info(f"   -> 成功載入 {len(weather_map)} 筆天氣小時資訊")
    except Exception as e:
        logging.warning(f"⚠️ 天氣獲取異常: {str(e)}")

    # 4. 填入天氣
    df_new_inc['temperature'] = df_new_inc.index.map(lambda x: weather_map.get(x, (np.nan, np.nan))[0])
    df_new_inc['humidity'] = df_new_inc.index.map(lambda x: weather_map.get(x, (np.nan, np.nan))[1])
    
    # 5. 存檔與格式化 (相容 Linux 環境)
    date_fmt = '%Y/%-m/%-d %H:%M' # GitHub Actions 跑在 Linux 上，使用 %-m 移除補零
    df_new_inc = df_new_inc.reset_index()
    df_new_inc['datetime'] = df_new_inc['datetime'].dt.strftime(date_format=date_fmt)
    
    cols = ['datetime', 'isMssingData', 'power', 'temperature', 'humidity']
    df_final = pd.concat([df_master[cols], df_new_inc[cols]], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['datetime'], keep='first')
    
    df_final.to_csv(MASTER_FILE, index=False, encoding='utf-8')
    logging.info("-" * 50)
    logging.info(f"🎉 全部任務成功！最新時間點: {df_new_inc['datetime'].iloc[-1]}")
    logging.info(f"📊 最終 CSV 總筆數: {len(df_final)}")
    logging.info("-" * 50)

if __name__ == "__main__":
    if sync_cloud_to_pantry():
        update_local_csv()