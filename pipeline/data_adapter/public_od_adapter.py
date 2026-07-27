"""
public_od_adapter.py
公開資料介接器：讀取台北市開放資料（臺北捷運每日分時各站OD流量統計資料）

真實資料格式（已確認）：
  欄位：日期, 時段, 進站, 出站, 人次
  - 日期：2024-01-01 格式
  - 時段：00~23（字串，需轉 int）
  - 進站/出站：站名中文
  - 人次：當日當時段該 OD 的實際人次

標準輸出格式（DataFrame）：
  columns: origin, destination, hour, trips, date
  - trips 為真實人次（非比例），可直接加總

切換到私有資料：改用 private_afc_adapter.py，輸出相同欄位格式。
"""

import os
import glob
import pandas as pd
import numpy as np


def load_public_od(od_dir: str = None, od_path: str = None,
                   filter_years: list = None, filter_yearmonths: list = None) -> pd.DataFrame:
    """
    讀取公開分時 OD 資料。

    優先順序：
    1. od_dir 指定目錄 → 自動合併目錄內所有月份 CSV
    2. od_path 指定單一 CSV
    3. 以上都找不到 → 使用內建模擬資料（讓 pipeline 可以跑通）

    Args:
        od_dir:  data/od_raw/ 目錄路徑（包含多個月份 CSV）
        od_path: 單一合併 CSV 路徑（備用）
        filter_years: 只保留這些年份，例如 [2024]，空 list 代表全部
        filter_yearmonths: 只保留這些年月，例如 ['202401','202402']，空 list 代表全部

    Returns:
        標準格式 DataFrame: origin, destination, hour, trips, date
    """
    df = None

    # 優先讀 od_dir
    if od_dir and os.path.isdir(od_dir):
        df = _load_from_dir(od_dir)
    elif od_path and os.path.exists(od_path):
        df = _load_single_csv(od_path)
    
    if df is None or df.empty:
        print("[Adapter] 找不到公開 OD 資料，使用內建模擬資料")
        return _generate_sample_od()

    # 篩選年份或年月
    if filter_yearmonths:
        df = df[df['yearmonth'].isin(filter_yearmonths)]
        print(f"[Adapter] 篩選年月 {filter_yearmonths}：剩 {len(df):,} 筆")
    elif filter_years:
        df = df[df['year'].isin(filter_years)]
        print(f"[Adapter] 篩選年份 {filter_years}：剩 {len(df):,} 筆")

    if df.empty:
        print("[Adapter] 篩選後無資料，使用模擬資料")
        return _generate_sample_od()

    result = df[['origin', 'destination', 'hour', 'trips', 'date']].copy()
    print(f"[Adapter] 公開 OD 資料載入完成：{len(result):,} 筆 "
          f"（{result['date'].nunique()} 天，"
          f"{result['origin'].nunique()} 個起站）")
    return result


def _load_from_dir(od_dir: str) -> pd.DataFrame:
    """讀取目錄內所有月份 CSV 並合併"""
    pattern = os.path.join(od_dir, '*.csv')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[Adapter] {od_dir} 目錄內無 CSV 檔案")
        return None

    print(f"[Adapter] 找到 {len(files)} 個月份檔案，開始合併...")
    chunks = []
    for f in files:
        try:
            chunk = _load_single_csv(f)
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
        except Exception as e:
            print(f"[Adapter] 跳過 {os.path.basename(f)}：{e}")

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    print(f"[Adapter] 合併完成：{len(df):,} 筆")
    return df


def _load_single_csv(filepath: str) -> pd.DataFrame:
    """讀取單一月份 CSV，轉換為標準格式"""
    df = pd.read_csv(filepath, encoding='utf-8-sig', dtype={'時段': str})
    df.columns = df.columns.str.strip()

    # 欄位對應（相容不同版本格式）
    col_map = {
        '進站': 'origin', '起站': 'origin',
        '出站': 'destination', '迄站': 'destination',
        '時段': 'hour', '時間': 'hour',
        '人次': 'trips', '旅次': 'trips',
        '日期': 'date',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = {'origin', 'destination', 'hour', 'trips'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"缺少欄位：{missing}")

    df['trips'] = pd.to_numeric(df['trips'], errors='coerce').fillna(0)
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype(int)

    # 保留 date 欄位，衍生 year / yearmonth
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['yearmonth'] = df['date'].dt.strftime('%Y%m')
    else:
        df['date'] = pd.NaT
        df['year'] = None
        df['yearmonth'] = None

    return df[['origin', 'destination', 'hour', 'trips', 'date', 'year', 'yearmonth']]


def aggregate_to_typical_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    將多日資料彙整為「典型日」（各 OD × 時段 的日平均人次）。
    用於熱力圖和 GA 最佳化的輸入。

    Returns:
        DataFrame: origin, destination, hour, trips（日平均）
    """
    n_days = df['date'].nunique()
    if n_days == 0:
        n_days = 1

    agg = (
        df.groupby(['origin', 'destination', 'hour'])['trips']
        .sum()
        .reset_index()
    )
    agg['trips'] = (agg['trips'] / n_days).round(2)
    agg = agg[agg['trips'] > 0]  # 過濾零流量
    print(f"[Adapter] 典型日彙整：{n_days} 天平均，{len(agg):,} 筆有效 OD 紀錄")
    return agg


def _generate_sample_od() -> pd.DataFrame:
    """
    當公開資料尚未下載時，用合理的模擬資料讓 pipeline 可以跑通。
    模擬北捷主要 OD 對 + 真實時段分佈形狀（早晚尖峰）。
    """
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
        ('新北投', '台北車站'), ('台北車站', '新北投'),
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
                         'hour': hour, 'trips': max(0, trips),
                         'date': pd.NaT, 'year': None, 'yearmonth': None})

    df = pd.DataFrame(rows)
    print(f"[Adapter] 使用模擬 OD 資料（{len(od_pairs)} 個 OD 對，{len(df)} 筆）")
    return df
