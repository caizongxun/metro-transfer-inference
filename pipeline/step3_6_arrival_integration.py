"""
step3_6_arrival_integration.py
列車到離站時間整合（對應簡報 Page 8：The Clockwork）

將 data/arrival_raw/ 的 CSV 讀入，計算各轉乘站每小時的「實際候車等待時間」，
並以此修正 step3_5 校正後的 estimated_trips 使用的「平均班距 / 2」假設。

輸入資料格式（data/arrival_raw/*.csv）：
  PVID, Line, Station, PLAN_ArriveTime, PLAN_DeriveTime, ArriveTime, DeriveTime

輸出：
  flow_df 加上 actual_avg_wait_min 欄位（如無到離站資料則保留 headway/2 預估值）

使用方式：
  只需在 run_pipeline.py 的 step3_5 之後、step4 之前呼叫 integrate_arrival() 即可。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

ARRIVAL_DIR = Path('data/arrival_raw')

# network.json Line code -> 中文路線名（與 step4 保持一致）
LINE_CODE_MAP = {
    'R':  '淡水信義線',
    'BL': '板南線',
    'G':  '松山新店線',
    'O':  '中和新蘆線',
    'BR': '文湖線',
    'V':  '淡水信義線',
    'Y':  '板南線',
}

# 轉乘站 → 路線對應（與 step4 STATION_LINE_MAP 保持一致）
STATION_LINE_MAP = {
    '中山':     ['淡水信義線', '松山新店線'],
    '台北車站': ['淡水信義線', '板南線'],
    '民權西路': ['淡水信義線', '中和新蘆線'],
    '東門':     ['淡水信義線', '松山新店線'],
    '大安':     ['淡水信義線', '文湖線'],
    '忠孝新生': ['板南線',     '中和新蘆線'],
    '忠孝復興': ['板南線',     '文湖線'],
    '西門':     ['板南線',     '松山新店線'],
    '南京復興': ['松山新店線', '文湖線'],
    '松江南京': ['松山新店線', '中和新蘆線'],
    '古亭':     ['松山新店線', '中和新蘆線'],
    '公館':     ['松山新店線', '板南線'],
}


def _load_arrival_csv() -> pd.DataFrame:
    """掃 data/arrival_raw/ 下所有 CSV，合併成單一 DataFrame。"""
    if not ARRIVAL_DIR.exists():
        print(f'  [Step3.6] 找不到 {ARRIVAL_DIR}，跳過到離站整合')
        return pd.DataFrame()

    files = list(ARRIVAL_DIR.glob('*.csv'))
    if not files:
        print(f'  [Step3.6] {ARRIVAL_DIR} 內無 CSV，跳過到離站整合')
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            dfs.append(df)
        except Exception as e:
            print(f'  [Step3.6] 讀取失敗 {f.name}: {e}')
    if not dfs:
        return pd.DataFrame()

    raw = pd.concat(dfs, ignore_index=True)
    print(f'  [Step3.6] 到離站紀錄：{len(raw):,} 筆（來自 {len(files)} 個檔案）')
    return raw


def _compute_actual_headway(raw: pd.DataFrame) -> pd.DataFrame:
    """
    從到離站時間反推各路線各站各小時的「實際班距」（前後兩班到站時間差的中位數）。
    回傳 DataFrame：Line, Station, hour, actual_headway_min
    """
    # 欄位正規化
    col_map = {}
    for c in raw.columns:
        cl = c.strip().lower()
        if 'line' in cl:              col_map[c] = 'Line'
        elif 'station' in cl:         col_map[c] = 'Station'
        elif 'arrivetime' in cl.replace('_', '') and 'plan' not in cl:
            col_map[c] = 'ArriveTime'
    raw = raw.rename(columns=col_map)

    needed = ['Line', 'Station', 'ArriveTime']
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        print(f'  [Step3.6] 到離站 CSV 缺少欄位 {missing}，跳過')
        return pd.DataFrame()

    raw['ArriveTime'] = pd.to_datetime(raw['ArriveTime'], errors='coerce')
    raw = raw.dropna(subset=['ArriveTime'])
    raw['hour'] = raw['ArriveTime'].dt.hour
    raw['Line'] = raw['Line'].astype(str).map(lambda x: LINE_CODE_MAP.get(x.strip(), x.strip()))

    # 對每條路線、每個站、每小時排序後求相鄰到站間隔
    records = []
    for (line, station, hour), grp in raw.groupby(['Line', 'Station', 'hour']):
        times = grp['ArriveTime'].sort_values()
        if len(times) < 2:
            continue
        diffs = times.diff().dropna().dt.total_seconds() / 60.0
        # 過濾合理範圍（1 ~ 30 分鐘）
        diffs = diffs[(diffs >= 1) & (diffs <= 30)]
        if diffs.empty:
            continue
        records.append({
            'Line': line,
            'Station': station,
            'hour': int(hour),
            'actual_headway_min': round(diffs.median(), 2)
        })

    return pd.DataFrame(records)


def integrate_arrival(flow_df: pd.DataFrame) -> pd.DataFrame:
    """
    主要對外介面：在 flow_df 上加入 actual_avg_wait_min 欄位。

    邏輯：
    - 有到離站資料 → 對各轉乘站取其所屬路線的實際班距平均，除以 2 → actual_avg_wait_min
    - 無資料        → actual_avg_wait_min = None（step4 fitness 會 fallback 用 headway/2）
    """
    flow_df = flow_df.copy()
    flow_df['actual_avg_wait_min'] = None

    raw = _load_arrival_csv()
    if raw.empty:
        print('  [Step3.6] 無到離站資料，actual_avg_wait_min 全部為 None')
        return flow_df

    headway_df = _compute_actual_headway(raw)
    if headway_df.empty:
        print('  [Step3.6] 無法計算實際班距，actual_avg_wait_min 全部為 None')
        return flow_df

    # 建立 station × hour → avg actual_headway 查詢表
    records = []
    for station, lines in STATION_LINE_MAP.items():
        for hour in range(6, 24):
            relevant = headway_df[
                (headway_df['Station'] == station) &
                (headway_df['hour'] == hour) &
                (headway_df['Line'].isin(lines))
            ]
            if relevant.empty:
                continue
            avg_hw = relevant['actual_headway_min'].mean()
            records.append({
                'transfer_station': station,
                'hour': hour,
                'actual_avg_wait_min': round(avg_hw / 2, 2)
            })

    wait_df = pd.DataFrame(records)
    if wait_df.empty:
        return flow_df

    flow_df = flow_df.merge(wait_df, on=['transfer_station', 'hour'], how='left',
                            suffixes=('_old', ''))
    # 清理多餘欄位
    if 'actual_avg_wait_min_old' in flow_df.columns:
        flow_df = flow_df.drop(columns=['actual_avg_wait_min_old'])

    n_filled = flow_df['actual_avg_wait_min'].notna().sum()
    print(f'  [Step3.6] actual_avg_wait_min 填入 {n_filled}/{len(flow_df)} 筆')
    return flow_df


if __name__ == '__main__':
    # 簡單 smoke test
    sample = pd.DataFrame([
        {'transfer_station': '台北車站', 'hour': 8,  'estimated_trips': 500, 'transfer_ratio': 0.20},
        {'transfer_station': '忠孝復興', 'hour': 18, 'estimated_trips': 300, 'transfer_ratio': 0.15},
    ])
    result = integrate_arrival(sample)
    print(result)
