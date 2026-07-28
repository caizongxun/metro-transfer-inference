"""
step3_5_calibration.py
Step 3.5 — 用各站進出人次觀測值校正 estimated_trips

原理：
  OD 推算出的轉乘流量是模型估計值，沒有觀測依據。
  各站進出人次（公開資料）告訴我們每站每小時實際有多少人進出，
  雖然這個數字包含非轉乘旅客，但可以做兩件事：

  1. 量級校正（scale correction）
     estimated_total[station] = sum(estimated_trips) for all OD through station
     observed_total[station]  = sum(entry + exit) from ridership data
     scale_factor = observed_total / estimated_total
     corrected_trips = estimated_trips * scale_factor（per station）

  2. 上界截斷（cap）
     corrected_trips 不能超過該站該時段的 observed_total
     （轉乘人次不可能超過進出站總人次）

注意：
  - 若某站在進出人次資料中找不到對應名稱，scale_factor 設 1.0（不校正）
  - 校正後重新計算 transfer_ratio percentile rank（只對營運時段 hour >= 6）
  - 輸出欄位新增 scale_factor, pre_calibration_trips 方便 debug

進出人次 CSV 預期格式（自動偵測欄位）：
  有「日期」欄位：每日明細版
    → 欄位：日期, 站名, 進站人次, 出站人次
  有「年」或「年份」欄位：月份彙總版
    → 欄位：年份, 月份, 站名, 進站人次, 出站人次

  欄位名稱的空格/底線/繁簡差異會自動處理。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import re
import pandas as pd
import numpy as np
from pipeline.config import ANALYSIS_HOURS

_OPERATING_HOUR_MIN = 6

# 進出人次 CSV 的預設存放目錄
_DEFAULT_RIDERSHIP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'ridership_raw'
)

# ---------------------------------------------------------------------------
# 欄位偵測工具
# ---------------------------------------------------------------------------

def _normalize_col(name: str) -> str:
    """統一欄位名稱：去空白、底線、簡轉繁對照"""
    name = name.strip().replace('_', '').replace(' ', '')
    mapping = {
        '站名': '站名', '车站': '站名', '站別': '站名',
        '進站': '進站人次', '进站': '進站人次', '進站人次': '進站人次',
        '出站': '出站人次', '出站': '出站人次', '出站人次': '出站人次',
        '日期': '日期', '年份': '年份', '年': '年份',
        '月份': '月份', '月': '月份',
        '時段': '時段', '小時': '時段', '時': '時段', 'hour': '時段',
    }
    for k, v in mapping.items():
        if k in name:
            return v
    return name


def _detect_columns(df: pd.DataFrame) -> dict:
    """
    回傳欄位對照 dict，key 為語意名稱，value 為 df 的實際欄位名稱。
    必要欄位：站名, 進站人次, 出站人次
    選填欄位：日期 or (年份 + 月份), 時段
    """
    mapping = {}
    for col in df.columns:
        norm = _normalize_col(col)
        if norm not in mapping:
            mapping[norm] = col
    required = ['站名', '進站人次', '出站人次']
    missing = [r for r in required if r not in mapping]
    if missing:
        raise ValueError(
            f'進出人次 CSV 缺少必要欄位 {missing}。\n'
            f'實際欄位：{list(df.columns)}'
        )
    return mapping


# ---------------------------------------------------------------------------
# 讀取進出人次資料
# ---------------------------------------------------------------------------

def load_ridership(ridership_dir: str = _DEFAULT_RIDERSHIP_DIR) -> pd.DataFrame:
    """
    讀取 ridership_dir 下所有 CSV，合併成 (station_name, hour, total_passengers) 格式。
    hour 欄位：
      - 若原始資料有時段欄位 → 直接使用
      - 若只有日期/月份（無時段）→ 無法細分時段，total_passengers 為全日總和，
        hour 設 None，後續校正會用全日平均分配
    """
    csv_files = glob.glob(os.path.join(ridership_dir, '*.csv'))
    if not csv_files:
        print(f'  [Step3.5] 警告：{ridership_dir} 下無 CSV 檔案，跳過校正')
        return pd.DataFrame(columns=['station_name', 'hour', 'total_passengers'])

    dfs = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()
            col = _detect_columns(df)

            station_col = col['站名']
            entry_col   = col['進站人次']
            exit_col    = col['出站人次']
            hour_col    = col.get('時段', None)

            df[entry_col] = pd.to_numeric(df[entry_col], errors='coerce').fillna(0)
            df[exit_col]  = pd.to_numeric(df[exit_col],  errors='coerce').fillna(0)
            df['total_passengers'] = df[entry_col] + df[exit_col]

            if hour_col:
                df['hour'] = pd.to_numeric(df[hour_col], errors='coerce')
                result = df[[station_col, 'hour', 'total_passengers']].rename(
                    columns={station_col: 'station_name'}
                )
            else:
                # 無時段欄位：以全日平均分配給各小時
                daily = df.groupby(station_col)['total_passengers'].sum().reset_index()
                daily.columns = ['station_name', 'total_passengers']
                daily['total_passengers'] = daily['total_passengers'] / 24
                rows = []
                for _, row in daily.iterrows():
                    for h in range(24):
                        rows.append({
                            'station_name': row['station_name'],
                            'hour': h,
                            'total_passengers': row['total_passengers']
                        })
                result = pd.DataFrame(rows)

            dfs.append(result)
        except Exception as e:
            print(f'  [Step3.5] 跳過 {os.path.basename(path)}：{e}')

    if not dfs:
        return pd.DataFrame(columns=['station_name', 'hour', 'total_passengers'])

    merged = pd.concat(dfs, ignore_index=True)
    ridership = (
        merged.groupby(['station_name', 'hour'])['total_passengers']
        .mean()  # 多檔案取平均（典型日）
        .reset_index()
    )
    print(f'  [Step3.5] 進出人次資料：{len(ridership)} 筆，'
          f'{ridership["station_name"].nunique()} 站')
    return ridership


# ---------------------------------------------------------------------------
# 站名對齊
# ---------------------------------------------------------------------------

def _build_name_map(flow_stations: list, ridership_stations: list) -> dict:
    """
    嘗試將 flow 裡的站名對應到 ridership 資料的站名。
    策略（依優先順序）：
      1. 完全相符
      2. 去除「站」字後相符
      3. 任一方包含另一方
    回傳 {flow_station_name: ridership_station_name}
    """
    ridership_set = {s: s for s in ridership_stations}
    ridership_stripped = {re.sub(r'站$', '', s): s for s in ridership_stations}

    name_map = {}
    for fs in flow_stations:
        if fs in ridership_set:
            name_map[fs] = fs
            continue
        fs_stripped = re.sub(r'站$', '', fs)
        if fs_stripped in ridership_stripped:
            name_map[fs] = ridership_stripped[fs_stripped]
            continue
        # 包含關係
        candidates = [rs for rs in ridership_stations
                      if fs in rs or rs in fs or fs_stripped in rs]
        if candidates:
            name_map[fs] = candidates[0]

    return name_map


# ---------------------------------------------------------------------------
# 主校正函式
# ---------------------------------------------------------------------------

def calibrate(
    flow_df: pd.DataFrame,
    ridership_dir: str = _DEFAULT_RIDERSHIP_DIR,
) -> pd.DataFrame:
    """
    輸入：step3 輸出的 flow_df
      欄位：transfer_station, hour, estimated_trips, transfer_ratio, pressure_level

    輸出：校正後的 flow_df，新增欄位：
      pre_calibration_trips  — 校正前的 estimated_trips
      scale_factor           — observed_total / estimated_total（全日站級）
      estimated_trips        — 校正後（縮放 + cap）
      transfer_ratio         — 重新計算的 percentile rank
      pressure_level         — 重新計算的四分位
    """
    if flow_df.empty:
        return flow_df

    ridership = load_ridership(ridership_dir)

    if ridership.empty:
        print('  [Step3.5] 無進出人次資料，直接回傳原始 flow_df')
        flow_df['pre_calibration_trips'] = flow_df['estimated_trips']
        flow_df['scale_factor'] = 1.0
        return flow_df

    # 站名對齊
    flow_stations = flow_df['transfer_station'].unique().tolist()
    ridership_stations = ridership['station_name'].unique().tolist()
    name_map = _build_name_map(flow_stations, ridership_stations)

    unmatched = [s for s in flow_stations if s not in name_map]
    if unmatched:
        print(f'  [Step3.5] 找不到進出人次對應的轉乘站（{len(unmatched)} 個）：'
              f'{unmatched[:5]}... → scale_factor=1.0')

    # 計算每站全日 observed_total 和 estimated_total
    station_estimated = (
        flow_df.groupby('transfer_station')['estimated_trips'].sum()
        .reset_index()
        .rename(columns={'estimated_trips': 'estimated_total'})
    )
    station_observed = (
        ridership.groupby('station_name')['total_passengers'].sum()
        .reset_index()
        .rename(columns={'total_passengers': 'observed_total'})
    )

    # 合併，算 scale_factor
    station_estimated['ridership_name'] = station_estimated['transfer_station'].map(name_map)
    station_estimated = station_estimated.merge(
        station_observed,
        left_on='ridership_name', right_on='station_name',
        how='left'
    )
    station_estimated['scale_factor'] = (
        station_estimated['observed_total'] /
        station_estimated['estimated_total'].replace(0, np.nan)
    ).fillna(1.0).clip(lower=0.01, upper=100.0)  # 避免極端縮放

    scale_map = dict(zip(
        station_estimated['transfer_station'],
        station_estimated['scale_factor']
    ))

    # 備份原始值
    result = flow_df.copy()
    result['pre_calibration_trips'] = result['estimated_trips']
    result['scale_factor'] = result['transfer_station'].map(scale_map).fillna(1.0)

    # 縮放
    result['estimated_trips'] = (result['estimated_trips'] * result['scale_factor']).round(2)

    # 上界截斷：corrected_trips <= observed total 該站該時段
    ridership_hour = ridership.copy()
    ridership_hour['transfer_station'] = ridership_hour['station_name'].map(
        {v: k for k, v in name_map.items()}
    )
    ridership_hour = ridership_hour.dropna(subset=['transfer_station'])
    ridership_hour = ridership_hour.rename(columns={'total_passengers': 'obs_hour_total'})

    result = result.merge(
        ridership_hour[['transfer_station', 'hour', 'obs_hour_total']],
        on=['transfer_station', 'hour'],
        how='left'
    )
    has_obs = result['obs_hour_total'].notna()
    result.loc[has_obs, 'estimated_trips'] = result.loc[has_obs].apply(
        lambda r: min(r['estimated_trips'], r['obs_hour_total']), axis=1
    )
    result = result.drop(columns=['obs_hour_total'])

    # 重新計算 transfer_ratio（只對營運時段）
    result['transfer_ratio'] = 0.0
    is_operating = result['hour'] >= _OPERATING_HOUR_MIN
    if is_operating.any():
        operating_trips = result.loc[is_operating, 'estimated_trips']
        result.loc[is_operating, 'transfer_ratio'] = (
            operating_trips.rank(pct=True, method='average').round(4)
        )

    result['pressure_level'] = pd.cut(
        result['transfer_ratio'],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=['低', '中', '高', '極高'],
        include_lowest=True
    )

    matched_count = sum(1 for s in flow_stations if s in name_map)
    print(f'  [Step3.5] 站名對齊：{matched_count}/{len(flow_stations)} 站完成校正')
    print(f'  [Step3.5] scale_factor 範圍：'
          f'{result["scale_factor"].min():.3f} ~ {result["scale_factor"].max():.3f}')

    return result.sort_values(['hour', 'estimated_trips'], ascending=[True, False])


if __name__ == '__main__':
    # 快速測試：用假資料跑一遍
    import tempfile, json

    sample_flow = pd.DataFrame([
        {'transfer_station': '台北車站', 'hour': 8,  'estimated_trips': 500.0,
         'transfer_ratio': 0.9, 'pressure_level': '極高'},
        {'transfer_station': '台北車站', 'hour': 12, 'estimated_trips': 300.0,
         'transfer_ratio': 0.7, 'pressure_level': '高'},
        {'transfer_station': '七張',    'hour': 8,  'estimated_trips': 80.0,
         'transfer_ratio': 0.4, 'pressure_level': '中'},
        {'transfer_station': '古亭',    'hour': 9,  'estimated_trips': 120.0,
         'transfer_ratio': 0.5, 'pressure_level': '中'},
    ])

    # 用 tempdir 模擬有一個進出人次 CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        ridership_csv = os.path.join(tmpdir, 'ridership_sample.csv')
        ridership_data = pd.DataFrame([
            {'站名': '台北車站', '時段': 8,  '進站人次': 8000, '出站人次': 7000},
            {'站名': '台北車站', '時段': 12, '進站人次': 5000, '出站人次': 4500},
            {'站名': '七張',    '時段': 8,  '進站人次': 600,  '出站人次': 550},
            {'站名': '古亭',    '時段': 9,  '進站人次': 900,  '出站人次': 850},
        ])
        ridership_data.to_csv(ridership_csv, index=False, encoding='utf-8-sig')

        result = calibrate(sample_flow, ridership_dir=tmpdir)
        print('\n校正結果：')
        print(result[['transfer_station', 'hour', 'pre_calibration_trips',
                       'estimated_trips', 'scale_factor', 'transfer_ratio']].to_string())
