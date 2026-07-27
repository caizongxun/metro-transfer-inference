"""
step3_flow_estimation.py
轉乘站分時流量估算。

修正：移除 iterrows（在 24 萬筆上極慢），改用 explode + groupby 向量化。
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
    比 iterrows 快 100x+。
    """
    # 只保留有轉乘站的筆
    has_transfer = path_df['transfer_stations'].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    df = path_df[has_transfer][['transfer_stations', 'hour', 'trips']].copy()

    if df.empty:
        print('  [Step3] 警告：所有路徑的 transfer_stations 均為空，'
              '請確認 Step2 路徑解析是否正常')
        return pd.DataFrame(columns=['transfer_station', 'hour', 'estimated_trips',
                                     'transfer_ratio', 'pressure_level'])

    # explode list → 每個轉乘站一行
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

    hour_total = flow.groupby('hour')['estimated_trips'].transform('sum')
    flow['transfer_ratio'] = (flow['estimated_trips'] / hour_total.replace(0, np.nan)).round(4)

    flow['pressure_level'] = pd.cut(
        flow['transfer_ratio'],
        bins=[0, 0.05, PRESSURE_THRESHOLD, 0.30, 1.0],
        labels=['低', '中', '高', '極高'],
        include_lowest=True
    )

    return flow.sort_values(['hour', 'estimated_trips'], ascending=[True, False])


def build_heatmap_matrix(flow_df: pd.DataFrame) -> pd.DataFrame:
    pivot = flow_df.pivot_table(
        index='transfer_station',
        columns='hour',
        values='transfer_ratio',
        aggfunc='sum',
        fill_value=0
    )
    for h in ANALYSIS_HOURS:
        if h not in pivot.columns:
            pivot[h] = 0
    return pivot[sorted(pivot.columns)]


if __name__ == '__main__':
    sample = pd.DataFrame([
        {'origin': 'A', 'destination': 'C', 'hour': 8, 'trips': 100,
         'transfer_stations': ['台北車站'], 'path_prob': 1.0},
        {'origin': 'B', 'destination': 'D', 'hour': 8, 'trips': 200,
         'transfer_stations': ['忠孝復興'], 'path_prob': 0.6},
        {'origin': 'B', 'destination': 'D', 'hour': 9, 'trips': 150,
         'transfer_stations': ['台北車站', '忠孝復興'], 'path_prob': 0.4},
    ])
    flow = estimate_transfer_flow(sample)
    print(flow)
