"""
step3_flow_estimation.py
轉乘站分時流量估算。

transfer_ratio 定義：該站該時段旅次 在全路網所有（站, 時段）組合中的 percentile rank
語意：0% = 流量屬全路網最低，100% = 流量屬全路網最高
優點：
  1. 小站大站顏色會有意義的差異（台北車站 vs 七張在同時段不再同色）
  2. 熱力圖顏色分佈會自然拉開，不會全紅
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pipeline.config import ANALYSIS_HOURS, PRESSURE_THRESHOLD


def estimate_transfer_flow(path_df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化展開：每筆路徑的 transfer_stations list → explode → groupby 彙整。

    transfer_ratio 定義：全路網 percentile rank
      - 小站小時段會是低百分位，大站尖峰時段會是高百分低
      - 熱力圖顏色分佈不會全部堤唯在最高值
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

    # --- 全路網 percentile rank ---
    # 每個 (station, hour) 的 estimated_trips 在所有組合中的百分位排名
    # 語意：這個站這個時段的人流，在全路網所有轉乘記錄裡排第幾%
    flow['transfer_ratio'] = (
        flow['estimated_trips']
        .rank(pct=True, method='average')
        .round(4)
    )

    # pressure_level 改為四分位數切分
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
    ])
    flow = estimate_transfer_flow(sample)
    print(flow)
    print('\nheatmap matrix:')
    print(build_heatmap_matrix(flow))
