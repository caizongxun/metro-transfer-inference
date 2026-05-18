"""
afc_processor.py
票卡交易紀錄處理模組。
支援 Hackrail 2026 官方 CSV 欄位（簡報 page-6）。
欄位定義請參考 modules/data_schema.py。

官方欄位：
    id                  流水號
    device_id           閘門編碼
    svc_loc_id_entry    進站車站代碼
    Entry_datetime      進站交易時間
    svc_loc_id          出站車站代碼
    txn_time            出站交易時間
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from modules.data_schema import load_afc


def load(path: str) -> pd.DataFrame:
    """shortcut"""
    return load_afc(path)


def get_od_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    聚合 OD 配對統計。
    回傳欄位： origin, destination, total_trips, avg_trips_per_hour
    """
    grouped = (
        df.groupby(['svc_loc_id_entry', 'svc_loc_id'])
        .agg(
            total_trips=('id', 'count'),
            avg_travel_min=('travel_min', 'mean'),
        )
        .reset_index()
        .rename(columns={
            'svc_loc_id_entry': 'origin',
            'svc_loc_id':       'destination',
        })
        .sort_values('total_trips', ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def filter_by_hour(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """過濾小時區間，例如早尖峰 filter_by_hour(df, 7, 9)"""
    return df[
        (df['entry_hour'] >= start_hour) & (df['entry_hour'] < end_hour)
    ].reset_index(drop=True)


def filter_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """只保留平日（週一平日 dayofweek < 5）"""
    return df[df['entry_dt'].dt.dayofweek < 5].reset_index(drop=True)


def filter_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """只保留假日"""
    return df[df['entry_dt'].dt.dayofweek >= 5].reset_index(drop=True)


def get_travel_time_distribution(
    df: pd.DataFrame,
    origin: str,
    destination: str,
) -> pd.Series:
    """
    取得指定 OD 旅行時間（分鐘）的分佈。
    用於路徑機率推斷的旅行時間校正。
    """
    mask = (
        (df['svc_loc_id_entry'] == origin)
        & (df['svc_loc_id'] == destination)
    )
    return df[mask]['travel_min'].dropna()


def get_station_list(df: pd.DataFrame) -> list[str]:
    """取得所有出現過的站編碼"""
    return sorted(
        set(df['svc_loc_id_entry'].unique())
        | set(df['svc_loc_id'].unique())
    )


def get_hourly_volume(
    df: pd.DataFrame,
    origin: str | None = None,
    destination: str | None = None,
) -> pd.DataFrame:
    """
    分時流量統計（可選擇過濾特定 OD）。
    回傳欄位： entry_hour, trips
    """
    tmp = df.copy()
    if origin:
        tmp = tmp[tmp['svc_loc_id_entry'] == origin]
    if destination:
        tmp = tmp[tmp['svc_loc_id'] == destination]
    return (
        tmp.groupby('entry_hour')
        .agg(trips=('id', 'count'))
        .reset_index()
    )
