"""
step1_load_data.py
資料載入入口。根據 config.DATA_SOURCE 自動選擇公開或私有資料介接器。
輸出標準格式，後續 step 不需要知道資料來源。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pipeline.config import (
    DATA_SOURCE,
    PUBLIC_OD_PATH, PUBLIC_INOUT_PATH,
    PRIVATE_AFC_PATH, PRIVATE_TIMETABLE_PATH, PRIVATE_CROWDING_PATH
)


def load_od_data() -> pd.DataFrame:
    """
    主要入口。根據 DATA_SOURCE 選擇介接器。
    回傳標準 OD DataFrame: origin, destination, hour, trips
    """
    if DATA_SOURCE == 'private':
        from pipeline.data_adapter.private_afc_adapter import load_private_afc
        return load_private_afc(PRIVATE_AFC_PATH, PRIVATE_TIMETABLE_PATH, PRIVATE_CROWDING_PATH)
    else:
        from pipeline.data_adapter.public_od_adapter import load_public_od
        return load_public_od(PUBLIC_OD_PATH, PUBLIC_INOUT_PATH)


if __name__ == '__main__':
    df = load_od_data()
    print(df.head(10))
    print(f"\n總筆數：{len(df):,}")
    print(f"時段範圍：{df['hour'].min()} ~ {df['hour'].max()}")
    print(f"OD 對數量：{df.groupby(['origin','destination']).ngroups}")
