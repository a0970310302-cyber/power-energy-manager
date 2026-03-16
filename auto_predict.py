# auto_predict.py
import pandas as pd
import numpy as np  # 🌟 記得 import numpy
from datetime import datetime
from app_utils import load_data, get_current_bill_cycle
from model_service import ModelService

def run_offline_inference():
    print(f"[{datetime.now()}] 開始執行背景離線預測...")
    
    # 1. 載入最新歷史資料
    hist_df = load_data()
    if hist_df.empty:
        print("沒有歷史資料，取消預測。")
        return

    # 🌟 補上這段：手動加入模型需要的時間特徵
    hist_df['hour'] = hist_df.index.hour
    hist_df['dayofweek'] = hist_df.index.dayofweek
    hist_df['hour_sin'] = np.sin(2 * np.pi * hist_df['hour'] / 24)
    hist_df['hour_cos'] = np.cos(2 * np.pi * hist_df['hour'] / 24)
    hist_df['day_sin'] = np.sin(2 * np.pi * hist_df['dayofweek'] / 7)
    hist_df['day_cos'] = np.cos(2 * np.pi * hist_df['dayofweek'] / 7)

    # 2. 計算目標預測時數 (7天與距離結帳日的較大值)
    latest_time = hist_df.index[-1]
    _, cycle_end = get_current_bill_cycle(latest_time)
    hours_to_end = int((cycle_end - latest_time).total_seconds() // 3600)
    max_target_steps = max(1440, hours_to_end)
    
    # 3. 載入模型並預測
    service = ModelService()
    pred_df = service.generate_rolling_predictions(hist_df, steps=max_target_steps)
    
    # 4. 儲存預測結果為 CSV 快取檔
    if not pred_df.empty:
        pred_df.to_csv("prediction_cache.csv")
        print(f"[{datetime.now()}] 預測完成！共產出 {len(pred_df)} 筆預測，已儲存至 prediction_cache.csv")
    else:
        print("預測失敗或產生空資料。")

if __name__ == "__main__":
    run_offline_inference()