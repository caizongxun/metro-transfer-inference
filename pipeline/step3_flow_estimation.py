"""
step3_flow_estimation.py
轉乘站分時流量估算。

輸入：step2 輸出的 OD-路徑 DataFrame
輸出：每個轉乘站 × 每個時段的估計承壓量（絕對值 + 相對比例）

「相對比例」說明：
  - 公開資料下，我們能確定的是「比例」而非「精確人次」
  - 例如：台北車站在早尖峰承接 18.3% 的總轉乘量
  - 等私有 AFC 資料接入後，trips 欄位會是真實人次，比例自動變成絕對值
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pipeline.config import ANALYSIS_HOURS, PRESSURE_THRESHOLD


def estimate_transfer_flow(path_df: pd.DataFrame) -> pd.DataFrame:
    """
    把路徑 DataFrame 展開成「轉乘站 × 時段」的流量表。
    每筆旅次依其路徑上的轉乘站，計入各轉乘站的流量。
    """
    rows = []
    for _, row in path_df.iterrows():
        stations = row['transfer_stations']
        if not stations:  # 直達，沒有轉乘站
            continue
        for station in stations:
            rows.append({
                'transfer_station': station,
                'hour': row['hour'],
                'trips': row['trips']
            })

    if not rows:
        return pd.DataFrame(columns=['transfer_station', 'hour', 'estimated_trips',
                                     'transfer_ratio', 'pressure_level'])

    expanded = pd.DataFrame(rows)
    flow = (
        expanded.groupby(['transfer_station', 'hour'])
        ['trips'].sum()
        .reset_index()
        .rename(columns={'trips': 'estimated_trips'})
    )

    # 各時段總轉乘量（用於計算比例）
    hour_total = flow.groupby('hour')['estimated_trips'].transform('sum')
    flow['transfer_ratio'] = (flow['estimated_trips'] / hour_total).round(4)

    # 壓力等級
    flow['pressure_level'] = pd.cut(
        flow['transfer_ratio'],
        bins=[0, 0.05, PRESSURE_THRESHOLD, 0.30, 1.0],
        labels=['低', '中', '高', '極高'],
        include_lowest=True
    )

    return flow.sort_values(['hour', 'estimated_trips'], ascending=[True, False])


def build_heatmap_matrix(flow_df: pd.DataFrame) -> pd.DataFrame:
    """
    把流量表轉換為熱力圖矩陣：rows=轉乘站，cols=時段，values=transfer_ratio
    """
    pivot = flow_df.pivot_table(
        index='transfer_station',
        columns='hour',
        values='transfer_ratio',
        aggfunc='sum',
        fill_value=0
    )
    # 確保所有分析時段都有欄位
    for h in ANALYSIS_HOURS:
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot[sorted(pivot.columns)]
    return pivot


if __name__ == '__main__':
    # 測試用：產生假資料跑一次
    sample = pd.DataFrame([
        {'origin': 'A', 'destination': 'C', 'hour': 8, 'trips': 100,
         'transfer_stations': ['台北車站'], 'path_prob': 1.0},
        {'origin': 'B', 'destination': 'D', 'hour': 8, 'trips': 200,
         'transfer_stations': ['忠孝復興'], 'path_prob': 0.6},
        {'origin': 'B', 'destination': 'D', 'hour': 8, 'trips': 200,
         'transfer_stations': ['台北車站', '忠孝復興'], 'path_prob': 0.4},
    ])
    flow = estimate_transfer_flow(sample)
    print(flow)
