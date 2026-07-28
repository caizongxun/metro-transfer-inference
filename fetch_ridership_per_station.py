"""
fetch_ridership_per_station.py
批次下載北捷官網「各站進出量統計」 ODS 檔案
來源： https://www.metro.taipei/cp.aspx?n=FF31501BEBDD0136

URL 規律：
  https://web.metro.taipei/RidershipPerStation/YYYYMM_cht.ods
  例： 202401_cht.ods = 民國 113 年 1 月
資料範圍： 2015/01 起（民國 104 年起增加各站進出資料）

ODS 檔内工作表：
  工作表 1（進站）：欄 = 車站名稱, 列 = 日期, 內容 = 進站人次
  工作表 2（出站）：欄 = 車站名稱, 列 = 日期, 內容 = 出站人次

使用方式：
  # 下載最近 12 個月
  python fetch_ridership_per_station.py --latest 12

  # 下載指定年份
  python fetch_ridership_per_station.py --year 2024

  # 下載年份範圍
  python fetch_ridership_per_station.py --year-range 2023 2024

  # 下載并轉成單一共用 CSV（就放進 data/ridership_raw/）
  python fetch_ridership_per_station.py --latest 12 --convert-csv

  # 只轉換已存在的 ODS （不重新下載）
  python fetch_ridership_per_station.py --convert-csv --no-download
"""

import argparse
import os
import sys
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

BASE_URL = 'https://web.metro.taipei/RidershipPerStation'
ODS_DIR  = Path('data/ridership_ods')      # ODS 原始檔存放目錄
CSV_OUT  = Path('data/ridership_raw')      # 轉換後 CSV 目錄（step3_5 讀取處）
FIRST_YEAR_MONTH = (2015, 1)              # 各站資料最早起始年月

# ---------------------------------------------------------------------------
# URL / 檔名工具
# ---------------------------------------------------------------------------

def build_url(year: int, month: int) -> str:
    return f'{BASE_URL}/{year:04d}{month:02d}_cht.ods'


def ods_filename(year: int, month: int) -> str:
    return f'{year:04d}{month:02d}_cht.ods'


def available_months(start_ym=None, end_ym=None):
    """回傳 (year, month) list，預設範圍 = FIRST_YEAR_MONTH 至上個月"""
    today = date.today()
    if end_ym is None:
        # 取上個月（当月尚未封檔）
        if today.month == 1:
            end_ym = (today.year - 1, 12)
        else:
            end_ym = (today.year, today.month - 1)
    if start_ym is None:
        start_ym = FIRST_YEAR_MONTH

    months = []
    y, m = start_ym
    ey, em = end_ym
    while (y, m) <= (ey, em):
        months.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return months


# ---------------------------------------------------------------------------
# 下載
# ---------------------------------------------------------------------------

def download_ods(year: int, month: int, out_dir: Path, skip_existing=True) -> bool:
    url  = build_url(year, month)
    dest = out_dir / ods_filename(year, month)
    if skip_existing and dest.exists():
        print(f'  [跳過] {dest.name} 已存在')
        return True
    print(f'  [下載] {dest.name}  <- {url}')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, 'wb') as f:
            f.write(resp.read())
        size_kb = dest.stat().st_size / 1024
        print(f'  [完成] {dest.name}  ({size_kb:.0f} KB)')
        return True
    except Exception as e:
        print(f'  [失敗] {dest.name}: {e}')
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# ODS -> CSV 轉換
# ---------------------------------------------------------------------------

def convert_ods_to_csv(ods_path: Path, csv_out_dir: Path) -> bool:
    """
    讀取 ODS 內兩張工作表（進站 / 出站），
    轉成長格式 CSV：日期, 站名, 進站人次, 出站人次
    輸出檔名與 ODS 同名，對應年月和 str -> .csv
    """
    try:
        import pandas as pd
    except ImportError:
        print('  [錯誤] 需要 pandas：pip install pandas odfpy')
        return False

    try:
        sheets = pd.read_excel(ods_path, sheet_name=None, engine='odf',
                               index_col=0, header=0)
    except Exception as e:
        print(f'  [讀取失敗] {ods_path.name}: {e}')
        return False

    sheet_names = list(sheets.keys())
    if len(sheet_names) < 2:
        print(f'  [警告] {ods_path.name} 工作表數 < 2，跳過')
        return False

    entry_df = sheets[sheet_names[0]]   # 進站
    exit_df  = sheets[sheet_names[1]]   # 出站

    def melt_sheet(df, col_name):
        df = df.copy()
        df.index.name = '日期'
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]
        long = df.reset_index().melt(id_vars='日期', var_name='站名', value_name=col_name)
        long[col_name] = pd.to_numeric(long[col_name], errors='coerce').fillna(0).astype(int)
        return long

    entry_long = melt_sheet(entry_df, '進站人次')
    exit_long  = melt_sheet(exit_df,  '出站人次')

    merged = entry_long.merge(exit_long, on=['日期', '站名'], how='outer')
    merged['進站人次'] = merged['進站人次'].fillna(0).astype(int)
    merged['出站人次'] = merged['出站人次'].fillna(0).astype(int)
    merged = merged.sort_values(['日期', '站名']).reset_index(drop=True)

    csv_out_dir.mkdir(parents=True, exist_ok=True)
    out_name = ods_path.stem + '.csv'  # e.g. 202401_cht.csv
    out_path = csv_out_dir / out_name
    merged.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'  [CSV] {out_path}  ({len(merged)} 列)')
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='北捷各站進出量批次下載器'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--latest', type=int, metavar='N',
                       help='下載最近 N 個月')
    group.add_argument('--year', type=int,
                       help='下載指定西元年（例: 2024）所有月份')
    group.add_argument('--year-range', type=int, nargs=2, metavar=('FROM', 'TO'),
                       help='下載年份範圍（西元），例: --year-range 2022 2024')
    parser.add_argument('--month', type=int, default=None,
                        help='配合 --year 使用，只下載指定月份')
    parser.add_argument('--convert-csv', action='store_true',
                        help='下載完後轉換 ODS -> CSV（就放入 data/ridership_raw/）')
    parser.add_argument('--no-download', action='store_true',
                        help='不下載，只對已存在的 ODS 做 --convert-csv')
    parser.add_argument('--ods-dir', default=str(ODS_DIR),
                        help=f'ODS 存放目錄（預設: {ODS_DIR}）')
    parser.add_argument('--csv-dir', default=str(CSV_OUT),
                        help=f'CSV 輸出目錄（預設: {CSV_OUT}）')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='兩次請求間隔秒數（預設 0.5）')
    return parser.parse_args()


def resolve_months(args):
    today = date.today()
    prev_month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)

    if args.no_download:
        return []  # 不下載時不需要 month list

    if args.latest:
        all_months = available_months(end_ym=prev_month)
        return all_months[-args.latest:]

    if args.year:
        year = args.year
        start = (year, args.month if args.month else 1)
        end   = (year, args.month if args.month else 12)
        # 不能超過上個月
        if end > prev_month:
            end = prev_month
        return available_months(start_ym=start, end_ym=end)

    if args.year_range:
        start = (args.year_range[0], 1)
        end   = (args.year_range[1], 12)
        if end > prev_month:
            end = prev_month
        return available_months(start_ym=start, end_ym=end)

    # 預設：最近 6 個月
    print('[INFO] 未指定範圍，預認下載最近 6 個月')
    return available_months(end_ym=prev_month)[-6:]


def main():
    args = parse_args()
    ods_dir = Path(args.ods_dir)
    csv_dir = Path(args.csv_dir)
    ods_dir.mkdir(parents=True, exist_ok=True)

    months = resolve_months(args)

    # --- 下載階段 ---
    if not args.no_download:
        print(f'\n[Step 1] 下載 {len(months)} 個月份 ODS -> {ods_dir}\n')
        ok_count, fail_count = 0, 0
        for y, m in months:
            ok = download_ods(y, m, ods_dir)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            time.sleep(args.delay)
        print(f'\n下載完成：成功 {ok_count}，失敗 {fail_count}')
    else:
        print('[Step 1] 跳過下載（--no-download）')

    # --- 轉換階段 ---
    if args.convert_csv:
        ods_files = sorted(ods_dir.glob('*_cht.ods'))
        print(f'\n[Step 2] 轉換 {len(ods_files)} 個 ODS -> CSV  ->  {csv_dir}\n')
        ok_count, fail_count = 0, 0
        for ods_path in ods_files:
            csv_path = csv_dir / (ods_path.stem + '.csv')
            if csv_path.exists():
                print(f'  [跳過] {csv_path.name} 已存在')
                ok_count += 1
                continue
            ok = convert_ods_to_csv(ods_path, csv_dir)
            ok_count += ok
            fail_count += not ok
        print(f'\n轉換完成：成功 {ok_count}，失敗 {fail_count}')
        print(f'CSV 檔案已放入 {csv_dir}，可直接被 step3_5_calibration.py 讀取')
    else:
        if not args.no_download:
            print(f'\n提示：ODS 已存入 {ods_dir}。加上 --convert-csv 可轉出 CSV。')


if __name__ == '__main__':
    main()
