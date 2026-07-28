"""
step3_flow_estimation.py
轉乘站分時流量估算。

transfer_ratio 改為：該站該時段旅次 / 該站全天最高旅次
（原本是全部轉乘站的占比，導致所有站永遠跟其他站比，無法反映單站真實乘載狀態）
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

    transfer_ratio 定義：該站該時段旅次 / 該站全天最高時段旅次
    語意：0% = 實際上沒人，100% = 達到該站自己的當天峰値
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

    # --- 改為跟同站自己全天峰値比較 ---
    # peak_trips[station] = 該站在所有時段中的最高旅次
    station_peak = (
        flow.groupby('transfer_station')['estimated_trips']
        .max()
        .rename('peak_trips')
    )
    flow = flow.join(station_peak, on='transfer_station')
    flow['transfer_ratio'] = (
        flow['estimated_trips'] / flow['peak_trips'].replace(0, np.nan)
    ).round(4).fillna(0)
    flow.drop(columns='peak_trips', inplace=True)

    flow['pressure_level'] = pd.cut(
        flow['transfer_ratio'],
        bins=[0, 0.40, 0.65, 0.85, 1.0],
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
