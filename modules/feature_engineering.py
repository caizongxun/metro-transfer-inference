"""
feature_engineering.py
將 OD CSV 原始資料轉換為 ML 訓練特徵

輸入：load_od_directory() 回傳的 DataFrame
輸出：特徵 DataFrame，供 ml_model.py 訓練

特徵列表：
    - weekday       : 0=週一 ... 6=週日
    - hour          : 0~23
    - month         : 1~12
    - is_holiday    : 0/1（台灣國定假日）
    - is_peak       : 0/1（早尖峰 7-9 或晚尖峰 17-19）
    - origin_id     : 站名 label encoded
    - dest_id       : 站名 label encoded
    - origin_monthly_vol  : 當月該進站總人次（流量規模）
    - dest_monthly_vol    : 當月該出站總人次
    - od_historical_mean  : 該 OD 對的歷史平均時段人次
    - od_historical_std   : 該 OD 對的歷史標準差
    - trips               : 本筆人次（訓練目標輔助用）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 台灣國定假日（2023-2026 主要假日，可依需求擴充）
TW_HOLIDAYS = set([
    # 2023
    "2023-01-01", "2023-01-02", "2023-01-20", "2023-01-21", "2023-01-22",
    "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
    "2023-02-28", "2023-04-04", "2023-04-05", "2023-05-01",
    "2023-06-22", "2023-06-23", "2023-09-29", "2023-10-09",
    "2023-10-10",
    # 2024
    "2024-01-01", "2024-02-08", "2024-02-09", "2024-02-10",
    "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14",
    "2024-02-28", "2024-04-04", "2024-04-05", "2024-05-01",
    "2024-06-10", "2024-09-17", "2024-10-10",
    # 2025
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29",
    "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02",
    "2025-02-28", "2025-04-03", "2025-04-04", "2025-05-01",
    "2025-05-30", "2025-10-06", "2025-10-10",
    # 2026
    "2026-01-01", "2026-01-15", "2026-01-16", "2026-01-17",
    "2026-01-18", "2026-01-19", "2026-01-20", "2026-01-21",
    "2026-02-28", "2026-04-03", "2026-04-04", "2026-05-01",
    "2026-06-19", "2026-09-25", "2026-10-10",
])

PEAK_HOURS = set(list(range(7, 10)) + list(range(17, 20)))


def build_features(df: pd.DataFrame, fit_encoders: bool = True,
                   encoders: dict = None) -> tuple:
    """
    主特徵工程函式

    Args:
        df: OD DataFrame（含 日期, 時段, 進站, 出站, 人次）
        fit_encoders: True = 訓練新 encoder；False = 使用傳入的 encoders
        encoders: {'origin': LabelEncoder, 'dest': LabelEncoder}

    Returns:
        (feature_df, encoders)
        feature_df 包含所有特徵欄位
    """
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"])

    # ── 時間特徵 ──────────────────────────────────────────────
    df["weekday"]   = df["日期"].dt.dayofweek          # 0=Mon, 6=Sun
    df["hour"]      = df["時段"].astype(int)
    df["month"]     = df["日期"].dt.month
    df["year"]      = df["日期"].dt.year
    df["day"]       = df["日期"].dt.day
    df["yearmonth"] = df["日期"].dt.to_period("M").astype(str)

    date_str = df["日期"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = date_str.isin(TW_HOLIDAYS).astype(int)
    df["is_peak"]    = df["hour"].isin(PEAK_HOURS).astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # ── 站名 Label Encoding ───────────────────────────────────
    if fit_encoders or encoders is None:
        enc_origin = LabelEncoder()
        enc_dest   = LabelEncoder()
        all_stations = pd.concat([df["進站"], df["出站"]]).unique()
        enc_origin.fit(all_stations)
        enc_dest.fit(all_stations)
        encoders = {"origin": enc_origin, "dest": enc_dest}
    else:
        enc_origin = encoders["origin"]
        enc_dest   = encoders["dest"]

    # 處理 unseen labels
    def safe_transform(enc, series):
        known = set(enc.classes_)
        return series.apply(lambda x: enc.transform([x])[0] if x in known else -1)

    df["origin_id"] = safe_transform(enc_origin, df["進站"])
    df["dest_id"]   = safe_transform(enc_dest,   df["出站"])

    # ── 月流量規模特徵 ────────────────────────────────────────
    monthly_origin = (
        df.groupby(["yearmonth", "進站"])["人次"]
        .transform("sum")
    )
    monthly_dest = (
        df.groupby(["yearmonth", "出站"])["人次"]
        .transform("sum")
    )
    df["origin_monthly_vol"] = monthly_origin
    df["dest_monthly_vol"]   = monthly_dest

    # ── OD 對歷史統計特徵 ─────────────────────────────────────
    # 按 (進站, 出站, hour, weekday) 的歷史均值與標準差
    od_stats = (
        df.groupby(["進站", "出站", "hour", "weekday"])["人次"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "od_historical_mean", "std": "od_historical_std"})
        .reset_index()
    )
    df = df.merge(od_stats, on=["進站", "出站", "hour", "weekday"], how="left")
    df["od_historical_std"] = df["od_historical_std"].fillna(0)

    # ── 環狀特徵（幫助模型學週期性）────────────────────────────
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    return df, encoders


FEATURE_COLS = [
    "weekday", "hour", "month",
    "is_holiday", "is_peak", "is_weekend",
    "origin_id", "dest_id",
    "origin_monthly_vol", "dest_monthly_vol",
    "od_historical_mean", "od_historical_std",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
]


if __name__ == "__main__":
    import glob
    from modules.afc_processor import load_od_csv

    files = sorted(glob.glob("data/od_raw/*.csv"))
    if not files:
        print("請先執行 data_fetcher.py")
    else:
        df = load_od_csv(files[0])
        feat_df, enc = build_features(df)
        print(f"特徵維度：{feat_df[FEATURE_COLS].shape}")
        print(feat_df[FEATURE_COLS].head(3).to_string())
        print(f"\n特徵欄位：{FEATURE_COLS}")
