"""
data_schema.py
Hackrail 2026 北捐 三份 CSV 資料的欄位定義與讀入入口。

資料區間： 115-04-01 ~ 115-04-30（民國 115 年 4 月）
資料格式： CSV，去識別化處理

== 資料一：票卡交易紀錄 (AFC) ==
欄位：
    id                  流水號
    device_id           閘門編碼
    svc_loc_id_entry    進站車站代碼
    Entry_datetime      進站交易時間
    svc_loc_id          出站車站代碼
    txn_time            出站交易時間

== 資料二：列車擁擠度 ==
欄位：
    StationID           車站代碼
    TrainNumber         列車車次
    DU                  上下行（1=上行 / 2=下行）
    UPDATETIME          離站時間
    CW1_A_E             車卧1A 載重人數
    CW1_B_E             車卧1B 載重人數
    CW2_A_E             車卧2A 載重人數
    CW2_B_E             車卧2B 載重人數
    NCW_E               列車總載重
    CAR_NUMBER          列車滿載人數

== 資料三：列車到離站時間 ==
欄位：
    PVID                車次編碼
    Line                路線
    Station             車站工程代碼
    PLAN_ArriverTime    預計到站時間
    PLAN_DertiveTime    預計離站時間
    ArriveTime          實際到站時間
    DertiveTime         實際離站時間
"""

from __future__ import annotations
import pandas as pd


# ══ 欄位常數（其他模組 import 就引用這層，不要 hardcode） ═══════

# 票卡交易紀錄
AFC_COLS = [
    'id',
    'device_id',
    'svc_loc_id_entry',   # 進站代碼
    'Entry_datetime',     # 進站時間
    'svc_loc_id',         # 出站代碼
    'txn_time',           # 出站時間
]

# 列車擁擠度
CROWDING_COLS = [
    'StationID',
    'TrainNumber',
    'DU',           # 1=上行 2=下行
    'UPDATETIME',   # 離站時間
    'CW1_A_E',
    'CW1_B_E',
    'CW2_A_E',
    'CW2_B_E',
    'NCW_E',        # 總載重
    'CAR_NUMBER',   # 滿載人數
]

# 到離站時間
SCHEDULE_COLS = [
    'PVID',               # 車次編碼
    'Line',               # 路線
    'Station',            # 車站工程代碼
    'PLAN_ArriverTime',
    'PLAN_DertiveTime',
    'ArriveTime',
    'DertiveTime',
]


# ══ Loader 函數 ════════════════════════════════════════════

def load_afc(path: str) -> pd.DataFrame:
    """
    讀入票卡交易紀錄 CSV。

    回傳新增行：
        entry_dt    (datetime64) 進站時間
        exit_dt     (datetime64) 出站時間
        travel_min  (float)      實際旅行分鐘數
        entry_hour  (int)        進站時（尖峰分析用）
    """
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    # 時間轉型
    df['entry_dt'] = pd.to_datetime(df['Entry_datetime'], errors='coerce')
    df['exit_dt']  = pd.to_datetime(df['txn_time'],       errors='coerce')

    # 旅行時間（分鐘）
    df['travel_min'] = (df['exit_dt'] - df['entry_dt']).dt.total_seconds() / 60

    # 移除異常旅行時間（<0 或 >180 分鐘）
    df = df[(df['travel_min'] >= 0) & (df['travel_min'] <= 180)].copy()

    df['entry_hour'] = df['entry_dt'].dt.hour
    df['entry_date'] = df['entry_dt'].dt.date

    return df.reset_index(drop=True)


def load_crowding(path: str) -> pd.DataFrame:
    """
    讀入列車擁擠度 CSV。

    回傳新增行：
        update_dt       (datetime64) 離站時間
        load_rate       (float)      載客率 NCW_E / CAR_NUMBER
        total_load      (int)        等同 NCW_E，語意更清晰
    """
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    df['update_dt'] = pd.to_datetime(df['UPDATETIME'], errors='coerce')
    df['total_load'] = pd.to_numeric(df['NCW_E'], errors='coerce').fillna(0)
    df['capacity']   = pd.to_numeric(df['CAR_NUMBER'], errors='coerce').replace(0, pd.NA)
    df['load_rate']  = (df['total_load'] / df['capacity']).round(4)

    # 各車厳載重
    for col in ['CW1_A_E', 'CW1_B_E', 'CW2_A_E', 'CW2_B_E']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['hour'] = df['update_dt'].dt.hour
    df['date'] = df['update_dt'].dt.date
    df['DU']   = df['DU'].astype(str).str.strip()  # '1'=上行 '2'=下行

    return df.reset_index(drop=True)


def load_schedule(path: str) -> pd.DataFrame:
    """
    讀入列車到離站時間 CSV。

    回傳新增行：
        plan_arrive_dt  (datetime64) 預計到站
        plan_depart_dt  (datetime64) 預計離站
        actual_arrive_dt(datetime64) 實際到站
        actual_depart_dt(datetime64) 實際離站
        delay_sec       (float)      離站延誤秒數（實際−預計）
        dwell_sec       (float)      停靠時間秒（實際離到差）
    """
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    for raw, col in [
        ('PLAN_ArriverTime', 'plan_arrive_dt'),
        ('PLAN_DertiveTime', 'plan_depart_dt'),
        ('ArriveTime',       'actual_arrive_dt'),
        ('DertiveTime',      'actual_depart_dt'),
    ]:
        df[col] = pd.to_datetime(df[raw], errors='coerce')

    df['delay_sec'] = (
        df['actual_depart_dt'] - df['plan_depart_dt']
    ).dt.total_seconds()

    df['dwell_sec'] = (
        df['actual_depart_dt'] - df['actual_arrive_dt']
    ).dt.total_seconds().clip(lower=0)

    df['hour'] = df['actual_arrive_dt'].dt.hour
    df['date'] = df['actual_arrive_dt'].dt.date

    return df.reset_index(drop=True)
