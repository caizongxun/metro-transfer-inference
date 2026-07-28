"""
step5_timetable.py
完整時刻表生成（對應簡報 Page 10 第 3 項：時刻表全面優化）

將 GA 班距（step4）與換乘偏移（step4.5）結合，
推算出 6:00~23:00 每條路線的整點後各班次發車時刻，
輸出為：
  1. data/output/timetable_full.csv   - 長格式，每列一班次
  2. data/output/timetable_summary.json - 前端儀表板用 JSON

時刻表邏輯：
  對每條路線、每個小時 h：
    - 取 suggested_headway_min（班距）
    - 取 offset_min（初始偏移，來自 step4.5；無則為 0）
    - 在 [h:00 + offset, h:59] 內每隔 headway 分鐘安排一班
    - 跨小時班次歸入下一小時管理（避免重複）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import pandas as pd
from pathlib import Path
from pipeline.config import ANALYSIS_HOURS

OUTPUT_DIR = Path('data/output')
CSV_PATH   = OUTPUT_DIR / 'timetable_full.csv'
JSON_PATH  = OUTPUT_DIR / 'timetable_summary.json'


def generate_timetable(
    headway_df: pd.DataFrame,
    sync_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    生成時刻表 DataFrame。

    Parameters
    ----------
    headway_df : DataFrame，欄位 line, hour, suggested_headway_min
    sync_df    : DataFrame，欄位 line, hour, offset_min（可為 None）

    Returns
    -------
    timetable_df : 長格式 DataFrame
        欄位：line, hour, train_index, depart_time_str, depart_minute
    """
    if headway_df.empty:
        print('  [Step5T] headway_df 為空，無法生成時刻表')
        return pd.DataFrame()

    # 建立 offset 查詢字典
    offset_dict = {}
    if sync_df is not None and not sync_df.empty:
        for _, row in sync_df.iterrows():
            offset_dict[(row['line'], int(row['hour']))] = float(row['offset_min'])

    rows = []
    for _, hw_row in headway_df.iterrows():
        line  = hw_row['line']
        hour  = int(hw_row['hour'])
        hw    = float(hw_row['suggested_headway_min'])
        if hw <= 0:
            continue

        offset = offset_dict.get((line, hour), 0.0)
        # 在 [hour:offset, hour:59] 範圍內排班
        start_min = offset % 60  # offset 相對於整點
        t = start_min
        train_idx = 1
        while t < 60:
            abs_minute = hour * 60 + t
            # 明確轉成 int，避免 float format 錯誤
            h_disp = int(abs_minute) // 60
            m_disp = int(abs_minute) % 60
            # 只保留 6:00 ~ 23:59
            if 6 * 60 <= int(abs_minute) < 24 * 60:
                rows.append({
                    'line':           line,
                    'hour':           hour,
                    'train_index':    train_idx,
                    'depart_time_str': f'{h_disp:02d}:{m_disp:02d}',
                    'depart_minute':  int(abs_minute),
                    'headway_min':    round(hw, 1),
                    'offset_min':     round(offset, 2),
                })
            t += hw
            train_idx += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['line', 'depart_minute']).reset_index(drop=True)
    return df


def export_timetable(
    timetable_df: pd.DataFrame,
    headway_df: pd.DataFrame,
    sync_df: pd.DataFrame | None = None
) -> None:
    """
    寫出 CSV 和 JSON 兩份檔案。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    if not timetable_df.empty:
        timetable_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        print(f'  [Step5T] 時刻表 CSV：{CSV_PATH}  ({len(timetable_df):,} 班次)')
    else:
        print('  [Step5T] 時刻表為空，跳過 CSV 寫出')

    # --- JSON（儀表板用）---
    summary = {}

    # 每條路線的班距概覽
    if not headway_df.empty:
        for line, grp in headway_df.groupby('line'):
            summary[line] = {
                'headway_by_hour': {
                    str(int(r['hour'])): round(float(r['suggested_headway_min']), 1)
                    for _, r in grp.iterrows()
                }
            }

    # 換乘偏移
    if sync_df is not None and not sync_df.empty:
        for _, row in sync_df.iterrows():
            line = row['line']
            if line not in summary:
                summary[line] = {}
            summary[line].setdefault('offset_by_hour', {})
            summary[line]['offset_by_hour'][str(int(row['hour']))] = round(float(row['offset_min']), 2)

    # 各路線的發車清單（精簡版，只存 depart_time_str）
    if not timetable_df.empty:
        for line, grp in timetable_df.groupby('line'):
            if line not in summary:
                summary[line] = {}
            summary[line]['trains'] = sorted(grp['depart_time_str'].tolist())
            summary[line]['total_trains'] = len(grp)

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  [Step5T] 時刻表 JSON：{JSON_PATH}')


if __name__ == '__main__':
    # Smoke test
    sample_hw = pd.DataFrame([
        {'line': '板南線',     'hour': 8,  'suggested_headway_min': 4.0},
        {'line': '板南線',     'hour': 9,  'suggested_headway_min': 5.0},
        {'line': '淡水信義線', 'hour': 8,  'suggested_headway_min': 5.0},
    ])
    sample_sync = pd.DataFrame([
        {'line': '板南線',     'hour': 8, 'offset_min': 2.0},
        {'line': '淡水信義線', 'hour': 8, 'offset_min': 0.0},
    ])
    tt = generate_timetable(sample_hw, sample_sync)
    print(tt.to_string())
    export_timetable(tt, sample_hw, sample_sync)
