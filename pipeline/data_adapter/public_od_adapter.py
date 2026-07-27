"""
public_od_adapter.py
公開資料介接器—讀取台北捷運每日分時 OD 資料。

真實資料格式（已確認）：
  日期, 時段, 進站, 出站, 人次

記憶體策略：
  - chunk 讀取，邊讀邊對 (origin, destination, hour) 做彙整加總
  - 永遠不將全量原始資料載入記憶體
  - 最終輸出為 (origin, destination, hour, trips) 平均典型日

標準輸出格式：
  columns: origin, destination, hour, trips
  trips = 日平均人次

切換到私有資料：改用 private_afc_adapter.py，輸出相同欄位。
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict


def load_public_od(od_dir: str = None, od_path: str = None,
                   filter_years: list = None, filter_yearmonths: list = None,
                   chunk_size: int = 50_000) -> pd.DataFrame:
    """
    讀取公開分時 OD 資料，回傳典型日日平均 DataFrame。

    Args:
        od_dir:  data/od_raw/ 目錄
        od_path: 單一 CSV（備用）
        filter_years:      只保留這些年份，空 list = 全部
        filter_yearmonths: 只保留這些年月 (e.g. ['202401'])，空 = 全部
        chunk_size:        每次讀取行數

    Returns:
        DataFrame: origin, destination, hour, trips (日平均)
    """
    # 決定要讀哪些檔案
    files = _select_files(od_dir, od_path, filter_yearmonths, filter_years)

    if not files:
        print('[Adapter] 找不到對應資料，使用內建模擬資料')
        return _generate_sample_od()

    print(f'[Adapter] 讀取 {len(files)} 個檔案 (chunk={chunk_size:,})…')

    # accumulator: key=(origin, dest, hour) -> (sum_trips, n_days)
    acc = defaultdict(lambda: [0.0, set()])   # [trips_sum, set_of_dates]

    total_rows = 0
    for fpath in files:
        try:
            for chunk in pd.read_csv(fpath, encoding='utf-8-sig',
                                     dtype={'時段': str},
                                     chunksize=chunk_size):
                chunk.columns = chunk.columns.str.strip()
                chunk = _rename_columns(chunk)
                if not {'origin', 'destination', 'hour', 'trips'}.issubset(chunk.columns):
                    continue

                chunk['trips'] = pd.to_numeric(chunk['trips'], errors='coerce').fillna(0)
                chunk['hour']  = pd.to_numeric(chunk['hour'],  errors='coerce').fillna(0).astype(int)

                # 收集日期用於後續除以天數
                if 'date' in chunk.columns:
                    dates = pd.to_datetime(chunk['date'], errors='coerce')
                else:
                    dates = pd.Series([pd.NaT] * len(chunk))

                for _, row in chunk.iterrows():
                    key = (row['origin'], row['destination'], int(row['hour']))
                    acc[key][0] += row['trips']
                    d = dates.iloc[_]
                    if pd.notna(d):
                        acc[key][1].add(d.date())

                total_rows += len(chunk)
        except Exception as e:
            print(f'[Adapter] 跳過 {os.path.basename(fpath)}: {e}')

    if not acc:
        print('[Adapter] 彙整後無資料，使用模擬資料')
        return _generate_sample_od()

    # 轉為 DataFrame、除以天數得日平均
    rows = []
    for (origin, dest, hour), (trips_sum, date_set) in acc.items():
        n_days = len(date_set) if date_set else 1
        rows.append({'origin': origin, 'destination': dest,
                     'hour': hour, 'trips': round(trips_sum / n_days, 2)})

    df = pd.DataFrame(rows)
    df = df[df['trips'] > 0]
    print(f'[Adapter] 完成：讀了 {total_rows:,} 行原始資料 '
          f'→ {len(df):,} 筆典型日彙整 '
          f'({df["origin"].nunique()} 個起站)')
    return df


# 即 aggregate_to_typical_day 的別名，供 step1 import
def aggregate_to_typical_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    將已含 date 欄位的 raw_od 彙整為典型日。
    主要用於 private adapter 回傳的資料。
    公開資料已在 load_public_od 內部完成彙整。
    """
    if 'date' not in df.columns:
        return df[['origin', 'destination', 'hour', 'trips']]
    n_days = df['date'].nunique()
    if n_days == 0:
        n_days = 1
    agg = (df.groupby(['origin', 'destination', 'hour'])['trips']
             .sum().reset_index())
    agg['trips'] = (agg['trips'] / n_days).round(2)
    agg = agg[agg['trips'] > 0]
    return agg


# ---- 內部工具函數 ------------------------------------------------

def _select_files(od_dir, od_path, filter_yearmonths, filter_years):
    """???????????????????????????????????????????????????????????????"""  # noqa
    files = []
    if od_dir and os.path.isdir(od_dir):
        all_files = sorted(glob.glob(os.path.join(od_dir, '*.csv')))
        if filter_yearmonths:
            files = [f for f in all_files
                     if any(ym in os.path.basename(f) for ym in filter_yearmonths)]
            print(f'[Adapter] 年月篩選 {filter_yearmonths}: '
                  f'{len(files)}/{len(all_files)} 個檔案')
        elif filter_years:
            files = [f for f in all_files
                     if any(str(y) in os.path.basename(f) for y in filter_years)]
            print(f'[Adapter] 年份篩選 {filter_years}: '
                  f'{len(files)}/{len(all_files)} 個檔案')
        else:
            files = all_files
            print(f'[Adapter] 全部 {len(files)} 個檔案')
    elif od_path and os.path.exists(od_path):
        files = [od_path]
    return files


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        '進站': 'origin',  '起站': 'origin',
        '出站': 'destination', '迄站': 'destination',
        '時段': 'hour',   '時間': 'hour',
        '人次': 'trips',  '旅次': 'trips',
        '日期': 'date',
    }
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


def _generate_sample_od() -> pd.DataFrame:
    np.random.seed(42)
    od_pairs = [
        ('台北車站', '忠孝復興'), ('忠孝復興', '台北車站'),
        ('台北車站', '南京復興'), ('南京復興', '台北車站'),
        ('台北車站', '忠孝敦化'), ('忠孝敦化', '台北車站'),
        ('板橋', '台北車站'), ('台北車站', '板橋'),
        ('新店', '台北車站'), ('台北車站', '新店'),
        ('淡水', '台北車站'), ('台北車站', '淡水'),
        ('南勢角', '忠孝復興'), ('忠孝復興', '南勢角'),
        ('動物園', '大安'), ('大安', '動物園'),
        ('松山', '板橋'), ('板橋', '松山'),
    ]
    hour_weights = {
        6: 0.03, 7: 0.10, 8: 0.13, 9: 0.09, 10: 0.05,
        11: 0.04, 12: 0.05, 13: 0.04, 14: 0.04, 15: 0.04,
        16: 0.06, 17: 0.10, 18: 0.12, 19: 0.08, 20: 0.05,
        21: 0.04, 22: 0.03, 23: 0.01,
    }
    rows = []
    for origin, dest in od_pairs:
        base = np.random.randint(800, 3000)
        for hour, weight in hour_weights.items():
            trips = round(base * weight * (1 + np.random.normal(0, 0.05)))
            rows.append({'origin': origin, 'destination': dest,
                         'hour': hour, 'trips': max(0, trips)})
    df = pd.DataFrame(rows)
    print(f'[Adapter] 使用模擬 OD 資料（{len(od_pairs)} 個 OD 對，{len(df)} 筆）')
    return df
