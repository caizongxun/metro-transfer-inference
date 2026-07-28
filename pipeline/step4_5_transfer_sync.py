"""
step4_5_transfer_sync.py
跨線換乘等待時間最佳化（對應簡報 Page 10 第 2 項：轉乘時間無縫搭配）

問題定義：
  對每個多線轉乘站，兩條（或多條）路線的列車在同一站的到達時間差
  決定了轉乘乘客的換月等待時間。
  本步驟在 Step 4 GA 班距結果的基礎上，進一步微調各路線的「初始發車偏移量（offset_min）」，
  使高流量轉乘站的路線間到站時間差最小化。

決策變數：
  各路線的每小時發車偏移量 offset[line][hour]，單位分鐘，範圍 [0, headway)

目標函數（最小化）：
  Σ 各高壓轉乘站 × 時段 ×（estimated_trips × 路線間到站時間差）

輸出：
  sync_df：DataFrame，欄位 = line, hour, offset_min
  （代表各路線在該小時建議的初始發車偏移，與班距結合可建構時刻表）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pipeline.config import ANALYSIS_HOURS, PRESSURE_THRESHOLD

# 需要做跨線換乘同步的站清單（僅限雙線以上）
TRANSFER_SYNC_STATIONS = {
    '台北車站': ['淡水信義線', '板南線'],
    '忠孝復興': ['板南線', '文湖線'],
    '忠孝新生': ['板南線', '中和新蘆線'],
    '西門':     ['板南線', '松山新店線'],
    '東門':     ['淡水信義線', '松山新店線'],
    '民權西路': ['淡水信義線', '中和新蘆線'],
    '中山':     ['淡水信義線', '松山新店線'],
    '大安':     ['淡水信義線', '文湖線'],
    '南京復興': ['松山新店線', '文湖線'],
    '松江南京': ['松山新店線', '中和新蘆線'],
    '古亭':     ['松山新店線', '中和新蘆線'],
    '公館':     ['松山新店線', '板南線'],
}


def _transfer_wait(offsets: np.ndarray, headways: list) -> float:
    """
    給定兩條路線的 offset（分鐘）和班距，計算乘客平均換乘等待時間。
    到站時間序列：t_i = offset_i, offset_i + hw_i, offset_i + 2*hw_i, ...
    平均等待 = 在對方路線任意到站 → 下一班 → 期望等待時間
    簡化：最小化兩條路線到站時間差的最小值期望
    """
    if len(offsets) < 2 or len(headways) < 2:
        return 0.0
    # 模擬一小時（60 分鐘）內兩條路線的到站時刻
    arrivals = []
    for off, hw in zip(offsets, headways):
        if hw <= 0:
            continue
        t = off % hw
        times = []
        while t < 60:
            times.append(t)
            t += hw
        arrivals.append(np.array(times))

    if len(arrivals) < 2:
        return 0.0

    # 取兩條路線到站時間的最小差值期望（循環距離）
    total_wait = 0.0
    count = 0
    for t in arrivals[0]:
        gaps = np.array([min(abs(t - s), headways[1] - abs(t - s) % headways[1])
                         for s in arrivals[1]])
        total_wait += gaps.min()
        count += 1
    return total_wait / max(count, 1)


def optimize_transfer_sync(
    headway_df: pd.DataFrame,
    flow_df: pd.DataFrame
) -> pd.DataFrame:
    """
    主要對外介面。

    Parameters
    ----------
    headway_df : GA 最佳化後的班距 DataFrame（line, hour, suggested_headway_min）
    flow_df    : 校正後的轉乘流量 DataFrame（transfer_station, hour, estimated_trips）

    Returns
    -------
    sync_df : DataFrame，欄位 line, hour, offset_min
    """
    if headway_df.empty or flow_df.empty:
        print('  [Step4.5] headway_df 或 flow_df 為空，跳過換乘同步最佳化')
        return pd.DataFrame(columns=['line', 'hour', 'offset_min'])

    # 建立 headway 查詢字典：{line: {hour: headway}}
    hw_dict = {}
    for _, row in headway_df.iterrows():
        hw_dict.setdefault(row['line'], {})[int(row['hour'])] = float(row['suggested_headway_min'])

    # 以高壓轉乘站為加權依據
    flow_lookup = {}
    for _, row in flow_df.iterrows():
        key = (row['transfer_station'], int(row['hour']))
        flow_lookup[key] = float(row.get('estimated_trips', 0))

    all_lines = list(hw_dict.keys())
    result_rows = []

    for hour in ANALYSIS_HOURS:
        hour = int(hour)
        # 初始 offset = 0 for all lines
        initial_offsets = {line: 0.0 for line in all_lines}

        def objective(x):
            """x: offset for each line at this hour"""
            offsets = {line: x[i] for i, line in enumerate(all_lines)}
            total_loss = 0.0
            for station, lines in TRANSFER_SYNC_STATIONS.items():
                valid = [l for l in lines if l in hw_dict]
                if len(valid) < 2:
                    continue
                weight = flow_lookup.get((station, hour), 0.0)
                if weight == 0:
                    continue
                station_offsets = [offsets.get(l, 0.0) for l in valid]
                station_headways = [hw_dict[l].get(hour, 5.0) for l in valid]
                wait = _transfer_wait(np.array(station_offsets), station_headways)
                total_loss += weight * wait
            return total_loss

        # 各 line 的 offset bound = [0, headway)
        bounds = [(0, hw_dict[line].get(hour, 10) - 0.01) for line in all_lines]
        x0 = np.array([initial_offsets[line] for line in all_lines])

        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-6})
            best_offsets = res.x
        except Exception:
            best_offsets = x0

        for i, line in enumerate(all_lines):
            result_rows.append({
                'line': line,
                'hour': hour,
                'offset_min': round(float(best_offsets[i]), 2)
            })

    sync_df = pd.DataFrame(result_rows).sort_values(['line', 'hour']).reset_index(drop=True)
    improved = (sync_df['offset_min'] > 0).sum()
    print(f'  [Step4.5] 換乘同步完成，{improved}/{len(sync_df)} 個時段有非零偏移建議')
    return sync_df


if __name__ == '__main__':
    sample_headway = pd.DataFrame([
        {'line': '板南線',     'hour': 8,  'suggested_headway_min': 4.0},
        {'line': '淡水信義線', 'hour': 8,  'suggested_headway_min': 5.0},
        {'line': '板南線',     'hour': 18, 'suggested_headway_min': 4.5},
        {'line': '淡水信義線', 'hour': 18, 'suggested_headway_min': 5.0},
    ])
    sample_flow = pd.DataFrame([
        {'transfer_station': '台北車站', 'hour': 8,  'estimated_trips': 500},
        {'transfer_station': '台北車站', 'hour': 18, 'estimated_trips': 480},
    ])
    sync = optimize_transfer_sync(sample_headway, sample_flow)
    print(sync)
