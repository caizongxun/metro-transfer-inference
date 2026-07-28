"""
step3_5_calibration.py
Step 3.5 — 用各站進出人次觀測值校正 estimated_trips

原理：
  OD 推算出的轉乘流量是模型估計值，沒有觀測依據。
  各站進出人次（公開資料）告訴我們每站每月/每年實際有多少人進出，
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

支援的進出人次 CSV 格式：
  1. 有時段欄位
     欄位：站名, 時段, 進站人次, 出站人次
  2. 無時段（月份/年份彙總版）
     欄位：統計期, 捷運站別, 進站人次, 出站人次[, 增減率欄位（自動忽略）]
     統計期為「85年」「113年1月」等民國年字串
     → 自動取最近年份資料，全日平均分配給各小時
  3. 一般日期版
     欄位：日期, 站名, 進站人次, 出站人次

  站名後綴（BR/R/G/O/BL/V/Y）和臺/台差異會自動處理。
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

_DEFAULT_RIDERSHIP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'ridership_raw'
)

# 北捷路線代碼後綴（站名可能帶著路線標記）
_LINE_SUFFIX_RE = re.compile(r'(BR|BL|[RGOVY])$')

# ---------------------------------------------------------------------------
# 欄位偵測
# ---------------------------------------------------------------------------

def _normalize_col(name: str) -> str:
    name = name.strip().replace('_', '').replace(' ', '')
    table = [
        # 站名相關
        ('捷運站別', '站名'), ('站名', '站名'), ('車站', '站名'), ('站別', '站名'),
        # 進站
        ('進站人次', '進站人次'), ('進站', '進站人次'), ('进站', '進站人次'),
        # 出站
        ('出站人次', '出站人次'), ('出站', '出站人次'), ('出站', '出站人次'),
        # 時段
        ('時段', '時段'), ('小時', '時段'), ('時', '時段'), ('hour', '時段'),
        # 日期
        ('日期', '日期'),
        # 統計期（民國年）— 記錄為特殊 key，load_ridership 會處理
        ('統計期', '統計期'),
        # 年月
        ('年份', '年份'), ('年', '年份'), ('月份', '月份'), ('月', '月份'),
    ]
    for keyword, semantic in table:
        if keyword in name:
            return semantic
    return name


def _detect_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    for col in df.columns:
        norm = _normalize_col(col)
        if norm not in mapping:
            mapping[norm] = col
    required = ['站名', '進站人次', '出站人次']
    missing = [r for r in required if r not in mapping]
    if missing:
        raise ValueError(
            f'進出人次 CSV 缺少必要欄位 {missing}。\n實際欄位：{list(df.columns)}'
        )
    return mapping


# ---------------------------------------------------------------------------
# 站名正規化（去後綴、臺/台統一）
# ---------------------------------------------------------------------------

def _normalize_station_name(name: str) -> str:
    """去掉路線代碼後綴，統一臺/台，去空白"""
    name = str(name).strip()
    name = name.replace('臺', '台')          # 繁簡/異體
    name = _LINE_SUFFIX_RE.sub('', name)     # 去 BR/BL/R/G/O/V/Y 後綴
    name = name.rstrip()                     # 去掉可能遺留的空白
    return name


# ---------------------------------------------------------------------------
# 讀取進出人次資料
# ---------------------------------------------------------------------------

def _parse_roc_year(s: str):
    """從 '113年1月' 或 '85年' 解析民國年數字，解析失敗回傳 None"""
    m = re.match(r'^(\d+)年', str(s).strip())
    return int(m.group(1)) if m else None


def load_ridership(ridership_dir: str = _DEFAULT_RIDERSHIP_DIR) -> pd.DataFrame:
    """
    讀取 ridership_dir 下所有 CSV，合併成 (station_name, hour, total_passengers)。
    station_name 已正規化（去後綴、台/臺統一）。
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

            # --- 偵測是否為「統計期 + 捷運站別」格式（北捷公開資料格式）---
            has_stat_period = any('統計期' in c for c in df.columns)
            has_metro_station = any('捷運站別' in c for c in df.columns)

            if has_stat_period and has_metro_station:
                result = _load_roc_format(df)
            else:
                col = _detect_columns(df)
                result = _load_generic_format(df, col)

            dfs.append(result)
        except Exception as e:
            print(f'  [Step3.5] 跳過 {os.path.basename(path)}：{e}')

    if not dfs:
        return pd.DataFrame(columns=['station_name', 'hour', 'total_passengers'])

    merged = pd.concat(dfs, ignore_index=True)
    ridership = (
        merged.groupby(['station_name', 'hour'])['total_passengers']
        .mean()
        .reset_index()
    )
    print(f'  [Step3.5] 進出人次資料：{len(ridership)} 筆，'
          f'{ridership["station_name"].nunique()} 站')
    return ridership


def _load_roc_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    處理北捷公開資料格式：
      統計期（民國年）, 捷運站別, 進站人次, 出站人次[, 增減率欄位...]
    取最近年份資料，全日平均分配給各小時。
    """
    stat_col    = next(c for c in df.columns if '統計期' in c)
    station_col = next(c for c in df.columns if '捷運站別' in c)
    entry_col   = next(c for c in df.columns if '進站人次' in c)
    exit_col    = next(c for c in df.columns if '出站人次' in c)

    df = df[[stat_col, station_col, entry_col, exit_col]].copy()
    df['_roc_year'] = df[stat_col].apply(_parse_roc_year)
    df = df.dropna(subset=['_roc_year'])

    # 取最近年份（最大民國年）
    latest_year = df['_roc_year'].max()
    df = df[df['_roc_year'] == latest_year].copy()
    print(f'  [Step3.5] 進出人次資料使用民國 {int(latest_year)} 年資料')

    df[entry_col] = pd.to_numeric(df[entry_col], errors='coerce').fillna(0)
    df[exit_col]  = pd.to_numeric(df[exit_col],  errors='coerce').fillna(0)
    df['total_passengers'] = df[entry_col] + df[exit_col]

    # 正規化站名
    df['station_name'] = df[station_col].apply(_normalize_station_name)

    # 全日加總後平均分配給 24 小時
    daily = df.groupby('station_name')['total_passengers'].sum().reset_index()
    rows = []
    for _, row in daily.iterrows():
        hourly = row['total_passengers'] / 24
        for h in range(24):
            rows.append({'station_name': row['station_name'],
                         'hour': h,
                         'total_passengers': hourly})
    return pd.DataFrame(rows)


def _load_generic_format(df: pd.DataFrame, col: dict) -> pd.DataFrame:
    """處理有時段欄位或日期欄位的一般格式"""
    station_col = col['站名']
    entry_col   = col['進站人次']
    exit_col    = col['出站人次']
    hour_col    = col.get('時段', None)

    df[entry_col] = pd.to_numeric(df[entry_col], errors='coerce').fillna(0)
    df[exit_col]  = pd.to_numeric(df[exit_col],  errors='coerce').fillna(0)
    df['total_passengers'] = df[entry_col] + df[exit_col]
    df['station_name'] = df[station_col].apply(_normalize_station_name)

    if hour_col:
        df['hour'] = pd.to_numeric(df[hour_col], errors='coerce')
        return df[['station_name', 'hour', 'total_passengers']].copy()
    else:
        daily = df.groupby('station_name')['total_passengers'].sum().reset_index()
        rows = []
        for _, row in daily.iterrows():
            hourly = row['total_passengers'] / 24
            for h in range(24):
                rows.append({'station_name': row['station_name'],
                             'hour': h,
                             'total_passengers': hourly})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 站名對齊
# ---------------------------------------------------------------------------

def _build_name_map(flow_stations: list, ridership_stations: list) -> dict:
    """
    flow 站名 → ridership 站名對照。
    ridership_stations 已經過 _normalize_station_name 正規化。
    flow_stations 也先做正規化再比對。
    策略（依優先順序）：
      1. 正規化後完全相符
      2. 去「站」字後相符
      3. 包含關係
    """
    # ridership 正規化 lookup
    rs_norm_map = {_normalize_station_name(s): s for s in ridership_stations}
    rs_nostop   = {re.sub(r'站$', '', _normalize_station_name(s)): s
                   for s in ridership_stations}

    name_map = {}
    for fs in flow_stations:
        fs_norm = _normalize_station_name(fs)
        fs_nostop = re.sub(r'站$', '', fs_norm)

        if fs_norm in rs_norm_map:
            name_map[fs] = rs_norm_map[fs_norm]
            continue
        if fs_nostop in rs_nostop:
            name_map[fs] = rs_nostop[fs_nostop]
            continue
        # 包含關係（正規化後）
        candidates = [s for s in ridership_stations
                      if fs_norm in _normalize_station_name(s)
                      or _normalize_station_name(s) in fs_norm]
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
    if flow_df.empty:
        return flow_df

    ridership = load_ridership(ridership_dir)

    if ridership.empty:
        print('  [Step3.5] 無進出人次資料，直接回傳原始 flow_df')
        flow_df = flow_df.copy()
        flow_df['pre_calibration_trips'] = flow_df['estimated_trips']
        flow_df['scale_factor'] = 1.0
        return flow_df

    flow_stations = flow_df['transfer_station'].unique().tolist()
    ridership_stations = ridership['station_name'].unique().tolist()
    name_map = _build_name_map(flow_stations, ridership_stations)

    unmatched = [s for s in flow_stations if s not in name_map]
    if unmatched:
        print(f'  [Step3.5] 未對齊站（{len(unmatched)} 個）：{unmatched[:8]} → scale_factor=1.0')

    station_estimated = (
        flow_df.groupby('transfer_station')['estimated_trips'].sum()
        .reset_index().rename(columns={'estimated_trips': 'estimated_total'})
    )
    station_observed = (
        ridership.groupby('station_name')['total_passengers'].sum()
        .reset_index().rename(columns={'total_passengers': 'observed_total'})
    )

    station_estimated['ridership_name'] = station_estimated['transfer_station'].map(name_map)
    station_estimated = station_estimated.merge(
        station_observed, left_on='ridership_name', right_on='station_name', how='left'
    )
    station_estimated['scale_factor'] = (
        station_estimated['observed_total'] /
        station_estimated['estimated_total'].replace(0, np.nan)
    ).fillna(1.0).clip(lower=0.01, upper=100.0)

    scale_map = dict(zip(
        station_estimated['transfer_station'],
        station_estimated['scale_factor']
    ))

    result = flow_df.copy()
    result['pre_calibration_trips'] = result['estimated_trips']
    result['scale_factor'] = result['transfer_station'].map(scale_map).fillna(1.0)
    result['estimated_trips'] = (result['estimated_trips'] * result['scale_factor']).round(2)

    # 上界截斷
    ridership_hour = ridership.copy()
    reverse_map = {}
    for fs, rs in name_map.items():
        reverse_map.setdefault(rs, fs)  # 取第一個對應
    ridership_hour['transfer_station'] = ridership_hour['station_name'].map(reverse_map)
    ridership_hour = ridership_hour.dropna(subset=['transfer_station'])
    ridership_hour = ridership_hour.rename(columns={'total_passengers': 'obs_hour_total'})

    result = result.merge(
        ridership_hour[['transfer_station', 'hour', 'obs_hour_total']],
        on=['transfer_station', 'hour'], how='left'
    )
    has_obs = result['obs_hour_total'].notna()
    result.loc[has_obs, 'estimated_trips'] = result.loc[has_obs].apply(
        lambda r: min(r['estimated_trips'], r['obs_hour_total']), axis=1
    )
    result = result.drop(columns=['obs_hour_total'])

    # 重算 transfer_ratio
    result['transfer_ratio'] = 0.0
    is_operating = result['hour'] >= _OPERATING_HOUR_MIN
    if is_operating.any():
        result.loc[is_operating, 'transfer_ratio'] = (
            result.loc[is_operating, 'estimated_trips']
            .rank(pct=True, method='average').round(4)
        )

    result['pressure_level'] = pd.cut(
        result['transfer_ratio'],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=['低', '中', '高', '極高'],
        include_lowest=True
    )

    matched = sum(1 for s in flow_stations if s in name_map)
    print(f'  [Step3.5] 站名對齊：{matched}/{len(flow_stations)} 站')
    print(f'  [Step3.5] scale_factor 範圍：'
          f'{result["scale_factor"].min():.3f} ~ {result["scale_factor"].max():.3f}')

    return result.sort_values(['hour', 'estimated_trips'], ascending=[True, False])


if __name__ == '__main__':
    import tempfile

    # 模擬北捷公開資料格式（統計期 + 捷運站別）
    sample_flow = pd.DataFrame([
        {'transfer_station': '台北車站', 'hour': 8,  'estimated_trips': 500.0,
         'transfer_ratio': 0.9, 'pressure_level': '極高'},
        {'transfer_station': '台北車站', 'hour': 12, 'estimated_trips': 300.0,
         'transfer_ratio': 0.7, 'pressure_level': '高'},
        {'transfer_station': '七張',    'hour': 8,  'estimated_trips': 80.0,
         'transfer_ratio': 0.4, 'pressure_level': '中'},
        {'transfer_station': '南京復興', 'hour': 9,  'estimated_trips': 120.0,
         'transfer_ratio': 0.5, 'pressure_level': '中'},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        ridership_csv = os.path.join(tmpdir, 'ridership.csv')
        # 模擬北捷公開格式（含 BR 後綴、民國年）
        pd.DataFrame([
            {'統計期': '113年', '捷運站別': '臺北車站',   '進站人次': 192000, '出站人次': 168000,
             '進站人次增減率[%]': 0, '出站人次增減率[%]': 0},
            {'統計期': '113年', '捷運站別': '七張',       '進站人次': 14400,  '出站人次': 13200,
             '進站人次增減率[%]': 0, '出站人次增減率[%]': 0},
            {'統計期': '113年', '捷運站別': '南京復興BR', '進站人次': 28800,  '出站人次': 24000,
             '進站人次增減率[%]': 0, '出站人次增減率[%]': 0},
        ]).to_csv(ridership_csv, index=False, encoding='utf-8-sig')

        result = calibrate(sample_flow, ridership_dir=tmpdir)
        print('\n校正結果：')
        print(result[['transfer_station', 'hour', 'pre_calibration_trips',
                       'estimated_trips', 'scale_factor', 'transfer_ratio']].to_string())
