"""
afc_processor.py
支援真實臺北捷運 OD 資料格式：日期, 時段, 進站, 出站, 人次

真實資料欄位：
    日期   : YYYY-MM-DD
    時段   : 0~23 (整數小時)
    進站   : 中文站名（例：松山機場）
    出站   : 中文站名
    人次   : 該 OD 對在該時段的旅次數
"""

import os
import glob
import pandas as pd
import numpy as np


# 時段對應小時数 (時段 0 = 00:00~01:00, 以此類推)
SLOT_TO_HOUR = {i: i for i in range(24)}


def load_od_csv(path: str) -> pd.DataFrame:
    """讀取單個月份 OD CSV"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["日期"] = pd.to_datetime(df["日期"])
    df["時段"] = df["時段"].astype(int)
    df["人次"] = pd.to_numeric(df["人次"], errors="coerce").fillna(0).astype(int)
    return df


def load_od_directory(directory: str = "data/od_raw") -> pd.DataFrame:
    """
    讀取目錄下所有 OD CSV 並合並
    """
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        raise FileNotFoundError(f"在 {directory} 找不到任何 CSV，請先執行 data_fetcher.py")

    print(f"找到 {len(files)} 個檔案，讀取中...")
    dfs = []
    for f in files:
        try:
            tmp = load_od_csv(f)
            dfs.append(tmp)
            print(f"  讀入 {os.path.basename(f)}：{len(tmp):,} 筆")
        except Exception as e:
            print(f"  [警告] 讀取失敗 {f}：{e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"合併完成：共 {len(df):,} 筆")
    return df


def filter_by_timeslot(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """
    過濾時段（以整數小時為單位）
    例：早尖峰 filter_by_timeslot(df, 7, 9) 即 07:00~09:00
    """
    mask = (df["時段"] >= start_hour) & (df["時段"] < end_hour)
    result = df[mask].reset_index(drop=True)
    return result


def filter_by_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """過濾単一日期, date 格式 YYYY-MM-DD"""
    return df[df["日期"].dt.strftime("%Y-%m-%d") == date].reset_index(drop=True)


def filter_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """只保留平日（週一~週五）"""
    return df[df["日期"].dt.dayofweek < 5].reset_index(drop=True)


def get_od_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    聚合 OD 配對統計：計算每個進出站組合的總人次與平均人次
    回傳欄位： origin, destination, total_trips, avg_trips_per_slot
    """
    grouped = (
        df[df["人次"] > 0]
        .groupby(["進站", "出站"])
        .agg(
            total_trips=("人次", "sum"),
            slot_count=("時段", "count"),
        )
        .reset_index()
        .rename(columns={"進站": "origin", "出站": "destination"})
    )
    grouped["avg_trips_per_slot"] = (grouped["total_trips"] / grouped["slot_count"]).round(1)
    return grouped.sort_values("total_trips", ascending=False).reset_index(drop=True)


def get_hourly_od(df: pd.DataFrame, origin: str, destination: str) -> pd.DataFrame:
    """取得指定 OD 的分時人次分佈"""
    mask = (df["進站"] == origin) & (df["出站"] == destination)
    result = (
        df[mask]
        .groupby("時段")["人次"]
        .sum()
        .reset_index()
        .rename(columns={"人次": "trips"})
    )
    return result


def get_top_od_pairs(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """取得人次最多的 Top N OD 對"""
    od = get_od_pairs(df)
    return od.head(top_n)


def get_station_list(df: pd.DataFrame) -> list:
    """取得所有出現過的站名"""
    stations = sorted(set(df["進站"].unique()) | set(df["出站"].unique()))
    return stations


if __name__ == "__main__":
    # 測試讀入單個檔案
    import glob
    files = sorted(glob.glob("data/od_raw/*.csv"))
    if not files:
        print("請先執行 data_fetcher.py 下載資料")
    else:
        df = load_od_csv(files[0])
        print(f"讀入：{os.path.basename(files[0])}")
        print(f"筆數：{len(df):,}")
        print(f"\n前 3 筆：")
        print(df.head(3).to_string())

        print(f"\n早尖峰 OD Top 10（時段 7~9）：")
        peak = filter_by_timeslot(df, 7, 9)
        top = get_top_od_pairs(peak, top_n=10)
        print(top.to_string(index=False))

        print(f"\n所有站點：{len(get_station_list(df))} 站")
