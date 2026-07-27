"""
private_afc_adapter.py
比賽私有票卡資料介接器（預留窗口）。

比賽開始後，將私有 AFC / 擁擠度 / 到離站資料放到 config.py 指定路徑，
然後在 config.py 把 DATA_SOURCE 改為 'private'，
step1_load_data.py 會自動切換到這個介接器。

預期私有資料格式（根據 PDF 欄位說明）：
  AFC：id, device_id, svc_loc_id_entry, Entry_datetime, svc_loc_id, txn_time
  擁擠度：StationID, TrainNumber, DU, UPDATETIME, CW1_A_E, CW1_B_E, NCW_E, CAR_NUMBER
  到離站：PVID, Line, Station, PLAN_ArriverTime, PLAN_DertiveTime, ArriveTime, DertiveTime

輸出同樣為標準 OD 格式：origin, destination, hour, trips
但額外附帶 travel_time_sec 欄位，讓 step3 可以做更精確的路徑推估。
"""

import os
import pandas as pd


def load_private_afc(afc_path: str, timetable_path: str = None, crowding_path: str = None) -> pd.DataFrame:
    """
    讀取比賽私有 AFC 票卡資料。
    回傳標準格式 DataFrame，額外包含 travel_time_sec 欄位。
    """
    if not os.path.exists(afc_path):
        raise FileNotFoundError(
            f"找不到私有 AFC 資料：{afc_path}\n"
            "請確認比賽資料已放置在正確路徑，或在 config.py 將 DATA_SOURCE 改回 'public'"
        )

    print("[Adapter] 載入私有 AFC 資料...")
    afc = pd.read_csv(afc_path, encoding='utf-8-sig')

    # 計算旅行時間
    afc['Entry_datetime'] = pd.to_datetime(afc['Entry_datetime'])
    afc['txn_time'] = pd.to_datetime(afc['txn_time'])
    afc['travel_time_sec'] = (afc['txn_time'] - afc['Entry_datetime']).dt.total_seconds()
    afc['hour'] = afc['Entry_datetime'].dt.hour

    # 站碼對應站名（需要 station_mapping）
    # TODO: 比賽資料確認欄位格式後補全這段
    afc = afc.rename(columns={
        'svc_loc_id_entry': 'origin',
        'svc_loc_id': 'destination',
    })

    # 彙整為小時流量
    df = (
        afc.groupby(['origin', 'destination', 'hour'])
        .agg(trips=('travel_time_sec', 'count'),
             avg_travel_time=('travel_time_sec', 'mean'))
        .reset_index()
    )

    print(f"[Adapter] 私有 AFC 資料載入：{len(df):,} 筆 OD-時段組合")
    return df
