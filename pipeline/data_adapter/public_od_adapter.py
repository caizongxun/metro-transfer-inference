"""
public_od_adapter.py
公開資料介接器：讀取台北市開放資料，輸出標準 OD 格式。

標準輸出格式（DataFrame）：
  columns: origin, destination, hour, trips
  - origin/destination: 站名（中文）
  - hour: 0~23
  - trips: 該時段該 OD 的流量（公開資料為月彙整，會做日平均）

當取得比賽私有資料後，改用 private_afc_adapter.py，輸出相同欄位格式。
"""

import os
import pandas as pd
import numpy as np


def load_public_od(od_path: str, inout_path: str = None) -> pd.DataFrame:
    """
    讀取公開分時 OD 資料。
    支援台北市資料大平台格式，也支援 repo 內的 sample 格式。
    回傳標準格式 DataFrame。
    """
    if not os.path.exists(od_path):
        print(f"[Adapter] 找不到 {od_path}，使用內建 sample 資料")
        return _generate_sample_od()

    df = pd.read_csv(od_path, encoding='utf-8-sig')
    df = _normalize_columns(df)
    df = _to_standard_format(df)
    print(f"[Adapter] 公開 OD 資料載入：{len(df):,} 筆")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """統一欄位名稱，相容不同版本的台北開放資料格式"""
    col_map = {
        '進站': 'origin',
        '出站': 'destination',
        '起站': 'origin',
        '迄站': 'destination',
        '時段': 'hour',
        '時間': 'hour',
        '人次': 'trips',
        '旅次': 'trips',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


def _to_standard_format(df: pd.DataFrame) -> pd.DataFrame:
    """轉換為標準格式，做日平均（月資料 ÷ 30）"""
    required = {'origin', 'destination', 'hour', 'trips'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"公開資料缺少欄位：{missing}，請確認資料格式")

    df['trips'] = pd.to_numeric(df['trips'], errors='coerce').fillna(0)
    # 月資料做日平均
    if df['trips'].max() > 50000:
        df['trips'] = (df['trips'] / 30).round(1)

    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype(int)
    return df[['origin', 'destination', 'hour', 'trips']]


def _generate_sample_od() -> pd.DataFrame:
    """
    當公開資料尚未下載時，用合理的模擬資料讓 pipeline 可以跑通。
    模擬北捷主要 OD 對 + 真實時段分佈形狀（早晚尖峰）。
    """
    np.random.seed(42)

    # 北捷常見高流量 OD 對（依公開人次資料推算）
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

    # 時段分佈（模擬早尖峰 7-9、晚尖峰 17-19）
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
            trips = max(0, trips)
            rows.append({'origin': origin, 'destination': dest, 'hour': hour, 'trips': trips})

    df = pd.DataFrame(rows)
    print(f"[Adapter] 使用模擬 OD 資料（{len(od_pairs)} 個 OD 對，{len(df)} 筆）")
    return df
