"""
crowding_processor.py
列車擁擠度資料分析模組。
依賻 data_schema.load_crowding() 就已讀入並做好基礎轉型。
欄位定義請參考 modules/data_schema.py。
"""

from __future__ import annotations
import pandas as pd
from modules.data_schema import load_crowding


def load(path: str) -> pd.DataFrame:
    """shortcut"""
    return load_crowding(path)


def get_station_peak(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    各站小時平均載客率，找出尖峰小時。
    回傳欄位： StationID, hour, avg_load_rate, max_load_rate
    """
    result = (
        df.groupby(['StationID', 'hour'])
        .agg(
            avg_load_rate=('load_rate', 'mean'),
            max_load_rate=('load_rate', 'max'),
            sample_count=('total_load', 'count'),
        )
        .reset_index()
        .sort_values('avg_load_rate', ascending=False)
    )
    return result


def get_transfer_station_crowding(
    df: pd.DataFrame,
    station_ids: list[str],
) -> pd.DataFrame:
    """
    取得指定轉乘站的擁擠度時序列。

    Parameters
    ----------
    df          : load_crowding() 讀入的 DataFrame
    station_ids : 要查詢的車站代碼 list

    Returns
    -------
    欄位： StationID, TrainNumber, DU, update_dt, total_load, load_rate
    """
    mask = df['StationID'].isin(station_ids)
    return df[mask][[
        'StationID', 'TrainNumber', 'DU',
        'update_dt', 'total_load', 'load_rate', 'hour', 'date'
    ]].copy().reset_index(drop=True)


def detect_crowding_anomaly(
    df: pd.DataFrame,
    z_thresh: float = 2.5,
) -> pd.DataFrame:
    """
    對每個車站 × 小時 組合計算 Z-score，標記異常高載。
    回傳新增欄位： mean_load, std_load, z_score, is_anomaly
    """
    stat = (
        df.groupby(['StationID', 'hour'])['total_load']
        .agg(mean_load='mean', std_load='std')
        .reset_index()
    )
    merged = df.merge(stat, on=['StationID', 'hour'], how='left')
    merged['z_score'] = (
        (merged['total_load'] - merged['mean_load'])
        / merged['std_load'].clip(lower=1e-6)
    ).round(3)
    merged['is_anomaly'] = (merged['z_score'].abs() >= z_thresh).astype(int)
    merged['anomaly_dir'] = 'normal'
    merged.loc[merged['z_score'] >= z_thresh,  'anomaly_dir'] = 'surge'
    merged.loc[merged['z_score'] <= -z_thresh, 'anomaly_dir'] = 'drop'
    return merged.reset_index(drop=True)


def summary_by_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    依上下行 (DU) 分組統計各站平均載客率。
    DU='1' 上行 / DU='2' 下行
    """
    return (
        df.groupby(['StationID', 'DU'])
        .agg(
            avg_load_rate=('load_rate', 'mean'),
            avg_total_load=('total_load', 'mean'),
            trip_count=('TrainNumber', 'count'),
        )
        .reset_index()
    )
