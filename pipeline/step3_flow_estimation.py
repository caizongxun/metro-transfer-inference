"""
step3_flow_estimation.py
轉乘站分時流量估算。

transfer_ratio 定義：
  營運時段（6:00~23:00）內，該站該時段旅次除以全路網最大旅次（絕對正規化）
  深夜時段（0:00~5:00）直接設為 0，不參與計算

優點（相對 percentile rank）：
  - 熱力圖顏色反映真實流量差異，高峰站才紅，低流量站保持藍色
  - 不會因為 rank 均勻分佈導致所有格子都顯示紅色
  - 語意直觀：1.0 = 全路網最高流量，0.5 = 半滿
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pipeline.config import ANALYSIS_HOURS, PRESSURE_THRESHOLD

# 營運時段定義：6:00 以後才參與計算
_OPERATING_HOUR_MIN = 6


def estimate_transfer_flow(path_df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化展開：每筆路徑的 transfer_stations list → explode → groupby 彙整。

    transfer_ratio 計算流程：
      1. 深夜時段（hour < 6）：ratio = 0
      2. 營運時段（hour >= 6）：estimated_trips / max(estimated_trips)
         語意：1.0 = 全路網該時段最高轉乘流量，其餘按比例縮放
    """
    has_transfer = path_df['transfer_stations'].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    df = path_df[has_transfer][['transfer_stations', 'hour', 'trips']].copy()

    if df.empty:
        print('  [Step3] 警告：所有路徑的 transfer_stations 均為空，'
              '請確認 Step2 路徑解析是否正常')
        return pd.DataFrame(columns=['transfer_station', 'hour', 'estimated_trips',
                                     'transfer_ratio', 'pressure_level'])

    expanded = df.explode('transfer_stations').rename(
        columns={'transfer_stations': 'transfer_station'}
    )
    expanded = expanded[expanded['transfer_station'].notna()]
    expanded['trips'] = pd.to_numeric(expanded['trips'], errors='coerce').fillna(0)

    flow = (
        expanded.groupby(['transfer_station', 'hour'])['trips']
        .sum()
        .reset_index()
        .rename(columns={'trips': 'estimated_trips'})
    )

    # --- 分兩段計算 transfer_ratio ---
    is_operating = flow['hour'] >= _OPERATING_HOUR_MIN

    # 深夜時段直接為 0
    flow['transfer_ratio'] = 0.0

    # 營運時段：絕對正規化，trips / max(trips)
    if is_operating.any():
        operating_trips = flow.loc[is_operating, 'estimated_trips']
        max_trips = operating_trips.max()
        if max_trips > 0:
            flow.loc[is_operating, 'transfer_ratio'] = (
                (operating_trips / max_trips).round(4)
            )

    flow['pressure_level'] = pd.cut(
        flow['transfer_ratio'],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=['低', '中', '高', '極高'],
        include_lowest=True
    )

    return flow.sort_values(['hour', 'estimated_trips'], ascending=[True, False])


def build_heatmap_matrix(flow_df: pd.DataFrame) -> pd.DataFrame:
    pivot = flow_df.pivot_table(
        index='transfer_station',
        columns='hour',
        values='transfer_ratio',
        aggfunc='mean',
        fill_value=0
    )
    for h in ANALYSIS_HOURS:
        if h not in pivot.columns:
            pivot[h] = 0
    return pivot[sorted(pivot.columns)]


if __name__ == '__main__':
    sample = pd.DataFrame([
        {'origin': 'A', 'destination': 'C', 'hour': 8,  'trips': 100,
         'transfer_stations': ['台北車站'], 'path_prob': 1.0},
        {'origin': 'B', 'destination': 'D', 'hour': 8,  'trips': 200,
         'transfer_stations': ['忠孝復興'], 'path_prob': 0.6},
        {'origin': 'B', 'destination': 'D', 'hour': 9,  'trips': 150,
         'transfer_stations': ['台北車站', '忠孝復興'], 'path_prob': 0.4},
        {'origin': 'C', 'destination': 'E', 'hour': 8,  'trips': 50,
         'transfer_stations': ['古亭'], 'path_prob': 0.8},
        {'origin': 'C', 'destination': 'E', 'hour': 17, 'trips': 300,
         'transfer_stations': ['古亭'], 'path_prob': 0.8},
        {'origin': 'D', 'destination': 'F', 'hour': 2,  'trips': 5,
         'transfer_stations': ['台北車站'], 'path_prob': 1.0},
    ])
    flow = estimate_transfer_flow(sample)
    print(flow.to_string())
    print('\nheatmap matrix:')
    print(build_heatmap_matrix(flow))
