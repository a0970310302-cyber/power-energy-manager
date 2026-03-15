import requests
import pandas as pd
import numpy as np
import os
import urllib3
import shutil
import logging
from datetime import datetime

# ==========================================
# ⚙️ 統一參數與日誌設定
# ==========================================
# 💡 透過 os.getenv 讀取環境變數。第二個參數是「本地預設 fallback 值」。
# 確保你在自己電腦(Local)測試時依然能跑，但在 GitHub 上會優先吃 Secrets！
JSON_SOURCE_URL = os.getenv("JSON_SOURCE_URL", "https://api.jsonstorage.net/v1/json/888edb43-7993-46ef-8020-767afb44a2cb/3a6282d1-b474-4274-952b-e4a0a84f0deb?apiKey=e1230ae2-6eee-433a-ad58-7ab2c622b9e5")

PANTRY_ID = os.getenv("PANTRY_ID", "6a2e85f5-4af4-4efd-bb9f-c5604fe8475e")
PANTRY_BASKET = os.getenv("PANTRY_BASKET", "2026-q1")
PANTRY_URL = f"https://getpantry.cloud/apiv1/pantry/{PANTRY_ID}/basket/{PANTRY_BASKET}"

WEATHER_INDEX_URL = os.getenv("WEATHER_INDEX_URL", "https://api.jsonstorage.net/v1/json/12d77044-531c-4984-8421-01585a961bfb/3f1ac541-adb2-4275-901b-dd1300502c0f")

MASTER_FILE = "final_training_data_with_humidity.csv"
LOG_FILE = "log.txt"

# 設定 Logging：同時輸出到文件與螢幕
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def sync_cloud_to_pantry():
    logging.info("="*50)
    logging.info("🚩 [Step 1] 啟動雲端數據格式化與同步 (Debug Mode)")
    logging.info("="*50)
    
    try:
        # 1. 檢查網址環境變數
        url_len = len(JSON_SOURCE_URL) if JSON_SOURCE_URL else 0
        logging.info(f"🔍 [Debug] JSON_SOURCE_URL 長度: {url_len}")
        if url_len > 10:
            logging.info(f"🔍 [Debug] 網址開頭前 15 字元: {JSON_SOURCE_URL[:15]}...")
        
        logging.info("📡 正在請求 JsonStorage...")
        source_res = requests.get(JSON_SOURCE_URL)
        logging.info(f"🔍 [Debug] HTTP 回傳狀態碼: {source_res.status_code}")
        
        if source_res.status_code != 200:
            logging.error(f"❌ [Debug] 請求失敗，內容為: {source_res.text[:200]}")
            return False

        raw_json = source_res.json()
        data_block = raw_json.get("data", {})
        
        # 2. 檢查 JSON 結構裡的 Key
        all_keys = list(data_block.keys())
        logging.info(f"🔍 [Debug] 找到的日期 Key 數量: {len(all_keys)}")
        if all_keys:
            logging.info(f"🔍 [Debug] 前 3 個 Key 範例: {all_keys[:3]}")
        
        formatted_new_data = {}
        processed_dates = set()
        
        for date_key, date_content in data_block.items():
            # 3. 檢查日期過濾條件
            if not date_key.startswith("202"): 
                continue
            
            processed_dates.add(date_key)
            data_list = date_content.get("listAMIBase15MinData", [])
            
            # 只有抓到當天資料時才印出細節，避免 Log 太長
            if len(data_list) > 0:
                 logging.info(f"📊 [Debug] 處理日期 {date_key}: 原始資料共 {len(data_list)} 筆")
            
            # 反向掃描，尋找最後一筆「真正有效」的數據索引
            last_valid_idx = -1
            for i in range(len(data_list) - 1, -1, -1):
                item = data_list[i]
                if not (item.get("isMssingData") == 1 and item.get("power") == 0):
                    last_valid_idx = i
                    break
            
            # 截斷陣列
            valid_data_list = data_list[:last_valid_idx + 1] if last_valid_idx != -1 else []
            logging.info(f"🔍 掃描到日期 {date_key}: 原始 {len(data_list)} 筆 -> 過濾未產生數據後剩餘 {len(valid_data_list)} 筆")
            
            for item in valid_data_list:
                time_str = item.get("time")
                power_val = item.get("power")
                formatted_new_data[f"{date_key} {time_str}:00"] = power_val
        
        if not formatted_new_data:
            logging.error("❌ 來源中找不到任何有效電力數據，同步終止。")
            return False

        # 讀取 Pantry 現有資料
        p_res = requests.get(PANTRY_URL)
        p_data = p_res.json() if p_res.status_code == 200 else {}
        p_data.pop('_metadata', None)
        
        # 💡 新增邏輯：主動清除 Pantry 中屬於處理日期，但不在有效名單內的殘留未來資料
        keys_to_delete = []
        for pk in p_data.keys():
            # 取出日期部分 (前 10 個字元，例如 "2026-03-15")
            pk_date = pk[:10] 
            if pk_date in processed_dates and pk not in formatted_new_data:
                keys_to_delete.append(pk)
                
        for pk in keys_to_delete:
            del p_data[pk]
            
        if keys_to_delete:
            logging.info(f"🧹 已從 Pantry 清除 {len(keys_to_delete)} 筆殘留的未來假資料")
        
        old_count = len(p_data)
        p_data.update(formatted_new_data)
        logging.info(f"🔄 數據合併完成: 更新後總計 {len(p_data)} 筆")
        
        # 上傳回 Pantry
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

    # 2. 數據聚合 
    try:
        p_data = requests.get(PANTRY_URL).json()
        logging.info(f"📡 已從 Pantry 載入 {len(p_data)} 筆數據進行聚合分析")
        
        df_p = pd.DataFrame(list(p_data.items()), columns=['datetime', 'power'])
        df_p['datetime'] = pd.to_datetime(df_p['datetime'], errors='coerce')
        df_p = df_p.dropna(subset=['datetime']).set_index('datetime')
        df_p['power'] = pd.to_numeric(df_p['power'], errors='coerce')

        hourly_p = df_p['power'].resample('1h').sum()
        hourly_c = df_p['power'].resample('1h').count()
        df_new_api = pd.DataFrame(index=hourly_p.index)
        df_new_api['power'] = hourly_p.values
        df_new_api['isMssingData'] = ((4 - hourly_c) / 4).clip(lower=0).values
        
        # 增量過濾加入 3 天的安全覆寫視窗
        safe_dt = last_dt - pd.Timedelta(days=3)
        df_new_inc = df_new_api[df_new_api.index > safe_dt].copy()
        if df_new_inc.empty:
            logging.info("✨ 檢查完畢：近期無新數據或需校正的資料，無需更新。")
            return
        logging.info(f"📝 發現 {len(df_new_inc)} 小時的近期數據準備更新與寫入...")
    except Exception as e:
        logging.error(f"❌ [Step 2 數據處理失敗]: {str(e)}")
        return

    # 3. 獲取天氣
    weather_map = {}
    try:
        logging.info("🌤️ 正在同步對應時段的天氣資訊...")
        w_idx = requests.get(WEATHER_INDEX_URL).json().get('items', {})
        for date_str, info in w_idx.items():
            if pd.to_datetime(date_str) < safe_dt.normalize(): continue
            day_res = requests.get(info['uri']).json()
            rows = day_res.get('days', {}).get(date_str, {}).get('rows', [])
            for r in rows:
                weather_map[pd.to_datetime(r[0])] = (r[1], r[2])
        logging.info(f"   -> 成功載入 {len(weather_map)} 筆天氣小時資訊")
    except Exception as e:
        logging.warning(f"⚠️ 天氣獲取部分異常: {str(e)}")

    # 4. 填入天氣
    df_new_inc['temperature'] = df_new_inc.index.map(lambda x: weather_map.get(x, (np.nan, np.nan))[0])
    df_new_inc['humidity'] = df_new_inc.index.map(lambda x: weather_map.get(x, (np.nan, np.nan))[1])
    
    # 5. 存檔與格式化
    date_fmt = '%Y/%#m/%#d %H:%M' if os.name == 'nt' else '%Y/%-m/%-d %H:%M'
    df_new_inc = df_new_inc.reset_index()
    df_new_inc['datetime'] = df_new_inc['datetime'].dt.strftime(date_format=date_fmt)
    
    cols = ['datetime', 'isMssingData', 'power', 'temperature', 'humidity']
    df_final = pd.concat([df_master[cols], df_new_inc[cols]], ignore_index=True)
    
    # 保留最新資料並重新排序時間
    df_final = df_final.drop_duplicates(subset=['datetime'], keep='last')
    df_final['dt_temp'] = pd.to_datetime(df_final['datetime'], format='mixed')
    df_final = df_final.sort_values('dt_temp').drop(columns=['dt_temp'])
    
    df_final.to_csv(MASTER_FILE, index=False, encoding='utf-8')
    logging.info("-" * 50)
    logging.info(f"🎉 全部任務成功！最新時間點已推進/校正至: {df_final['datetime'].iloc[-1]}")
    logging.info(f"📊 最終 CSV 總筆數: {len(df_final)}")
    logging.info("-" * 50)

# ==========================================
# 🚦 主程式進入點
# ==========================================
if __name__ == "__main__":
    if sync_cloud_to_pantry():
        update_local_csv()
    else:
        logging.error("❌ 雲端同步失敗，已終止後續動作。")