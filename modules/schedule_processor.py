"""
schedule_processor.py
列車到離站時間資料分析模組。
依賻 data_schema.load_schedule() 就已讀入并做好基礎轉型。
欄位定義請參考 modules/data_schema.py。
"""

from __future__ import annotations
import pandas as pd
from modules.data_schema import load_schedule


def load(path: str) -> pd.DataFrame:
    """shortcut"""
    return load_schedule(path)


def get_train_timetable(df: pd.DataFrame, pvid: str) -> pd.DataFrame:
    """
    取得引定車次的全程到離站時間表。
    回傳欄位： Station, plan_arrive_dt, actual_arrive_dt, delay_sec, dwell_sec
    """
    return (
        df[df['PVID'] == pvid]
        [['Station', 'plan_arrive_dt', 'plan_depart_dt',
          'actual_arrive_dt', 'actual_depart_dt', 'delay_sec', 'dwell_sec']]
        .sort_values('actual_arrive_dt')
        .reset_index(drop=True)
    )


def get_trains_at_station_in_window(
    df: pd.DataFrame,
    station: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    """
    查詢指定車站在時間窗口內到達的所有車次。
    用於推斷旅客可能時段貜候的車次。
    """
    mask = (
        (df['Station'] == station)
        & (df['actual_arrive_dt'] >= start_dt)
        & (df['actual_arrive_dt'] <= end_dt)
    )
    return (
        df[mask][
            ['PVID', 'Line', 'Station',
             'actual_arrive_dt', 'actual_depart_dt', 'delay_sec', 'dwell_sec']
        ]
        .sort_values('actual_arrive_dt')
        .reset_index(drop=True)
    )


def compute_headway(df: pd.DataFrame, station: str, line: str) -> pd.DataFrame:
    """
    計算指定站、路線的實際班距（分鐘）。
    回傳欄位： PVID, Line, Station, actual_arrive_dt, headway_min
    """
    mask = (df['Station'] == station) & (df['Line'] == line)
    tmp = (
        df[mask][['PVID', 'Line', 'Station', 'actual_arrive_dt']]
        .sort_values('actual_arrive_dt')
        .copy()
    )
    tmp['headway_min'] = (
        tmp['actual_arrive_dt'].diff().dt.total_seconds() / 60
    ).round(2)
    return tmp.dropna(subset=['headway_min']).reset_index(drop=True)


def get_delay_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    各站延誤統計：平均、最大、P90 延誤秒。
    """
    return (
        df.groupby('Station')['delay_sec']
        .agg(
            avg_delay='mean',
            max_delay='max',
            p90_delay=lambda x: x.quantile(0.9),
            count='count',
        )
        .reset_index()
        .sort_values('avg_delay', ascending=False)
        .head(top_n)
    )


def match_train_to_od(
    schedule_df: pd.DataFrame,
    station: str,
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
    buffer_min: float = 3.0,
) -> pd.DataFrame:
    """
    核心推斷函數：給定旅客進出站時間 (OD)，
    找出旅客在指定站可能候車的車次。

    Parameters
    ----------
    schedule_df : load_schedule() 讀入的 DataFrame
    station     : 轉乘站代碼
    entry_dt    : 旅客進展站時間 (entry_datetime)
    exit_dt     : 旅客出站時間 (txn_time)
    buffer_min  : 旅客候車等候緩衝時間（分鐘）

    Returns
    -------
    在時間窗口內到達該站的候車車次列表
    """
    window_start = entry_dt
    window_end   = exit_dt + pd.Timedelta(minutes=buffer_min)
    return get_trains_at_station_in_window(
        schedule_df, station, window_start, window_end
    )
