"""
afc_processor.py
AFC 票卡資料清洗、OD 配對、旅行時間計算
"""

import pandas as pd
import numpy as np


def load_afc(afc_path: str = "data/sample_afc.csv") -> pd.DataFrame:
    """載入 AFC 資料，計算實際旅行時間"""
    df = pd.read_csv(afc_path, parse_dates=["entry_time", "exit_time"])
    df["travel_time_min"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
    return df


def get_od_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    OD 配對統計：每個 origin-destination 組合的旅次數與平均旅行時間
    """
    grouped = (
        df.groupby(["entry_station_id", "exit_station_id"])
        .agg(
            trip_count=("card_id_hash", "count"),
            avg_travel_time=("travel_time_min", "mean"),
            std_travel_time=("travel_time_min", "std"),
        )
        .reset_index()
        .rename(columns={"entry_station_id": "origin", "exit_station_id": "destination"})
    )
    grouped["std_travel_time"] = grouped["std_travel_time"].fillna(0)
    return grouped


def filter_by_timeslot(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    過濾特定時段的旅次
    start / end: 格式 'HH:MM'
    """
    mask = (
        df["entry_time"].dt.strftime("%H:%M") >= start
    ) & (
        df["entry_time"].dt.strftime("%H:%M") < end
    )
    return df[mask].reset_index(drop=True)


if __name__ == "__main__":
    df = load_afc()
    print(f"AFC 資料筆數：{len(df)}")
    print(df[["entry_station_id", "exit_station_id", "travel_time_min"]].head())
    od = get_od_pairs(df)
    print("\nOD 配對：")
    print(od)
