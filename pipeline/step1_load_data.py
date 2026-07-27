"""
step1_load_data.py
載入資料 — 根據 config.DATA_SOURCE 選擇公開或私有資料。
輸出標準格式 DataFrame 供後續步驟使用。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_adapter.public_od_adapter import load_public_od, aggregate_to_typical_day
from data_adapter.private_afc_adapter import load_private_afc


def run() -> dict:
    """
    Returns:
        dict with keys:
          'raw_od'     — 原始逐日資料（origin, destination, hour, trips, date）
          'typical_od' — 典型日日平均（origin, destination, hour, trips）
          'data_source'— 'public' or 'private'
    """
    if config.DATA_SOURCE == 'private':
        print("[Step1] 使用比賽私有 AFC 資料")
        raw_od = load_private_afc(config.PRIVATE_AFC_PATH)
        typical_od = aggregate_to_typical_day(raw_od)
        return {'raw_od': raw_od, 'typical_od': typical_od, 'data_source': 'private'}
    else:
        print("[Step1] 使用公開 OD 資料")
        raw_od = load_public_od(
            od_dir=config.PUBLIC_OD_DIR,
            od_path=config.PUBLIC_OD_PATH,
            filter_years=config.FILTER_YEARS,
            filter_yearmonths=config.FILTER_YEARMONTHS,
        )
        typical_od = aggregate_to_typical_day(raw_od)
        return {'raw_od': raw_od, 'typical_od': typical_od, 'data_source': 'public'}


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run()
    df = result['typical_od']
    print(f"\n典型日 OD 資料筆數：{len(df):,}")
    print(f"時段分佈：")
    print(df.groupby('hour')['trips'].sum().sort_index())
    print(f"\n流量最高前 10 個 OD 對：")
    top = df.groupby(['origin','destination'])['trips'].sum().nlargest(10)
    print(top)
