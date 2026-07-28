"""
data_fetcher.py
從臺北捷運公開資料批次下載：
  1. 每日分時各站 OD 流量（--mode od）
  2. 各站進出人次（--mode ridership）
  3. 全部（--mode all）

使用方式：
    python data_fetcher.py --mode od --index 臺北捷運每日分時各站OD流量統計資料.csv
    python data_fetcher.py --mode ridership --index 臺北捷運各站進出人次.csv
    python data_fetcher.py --mode all \\
        --od-index 臺北捷運每日分時各站OD流量統計資料.csv \\
        --ridership-index 臺北捷運各站進出人次.csv
"""

import argparse
import os
import subprocess
import pandas as pd


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------

def load_index(index_path: str) -> pd.DataFrame:
    df = pd.read_csv(index_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    print(f'索引欄位：{list(df.columns)}')
    print(f'共 {len(df)} 筆紀錄')
    return df


def find_url_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        sample = str(df[col].dropna().iloc[0])
        if sample.startswith('http'):
            return col
    raise ValueError('找不到 URL 欄位，請確認索引 CSV 格式')


def find_year_month_columns(df: pd.DataFrame):
    year_col, month_col = None, None
    for col in df.columns:
        col_lower = col.strip().lower()
        if '年' in col or 'year' in col_lower:
            year_col = col
        if '月' in col or 'month' in col_lower:
            month_col = col
    return year_col, month_col


def download_with_wget(url: str, output_dir: str, filename: str = None) -> bool:
    url = url.strip()
    if not filename:
        filename = url.split('/')[-1]
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        print(f'  [跳過] 已存在：{filename}')
        return True
    print(f'  [下載] {filename}')
    cmd = ['wget', '-q', '-O', output_path, url]
    try:
        result = subprocess.run(cmd, timeout=120, capture_output=True, text=True)
        if result.returncode == 0:
            size = os.path.getsize(output_path)
            print(f'  [完成] {filename} ({size/1024:.1f} KB)')
            return True
        else:
            print(f'  [失敗] {filename}: {result.stderr.strip()}')
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    except subprocess.TimeoutExpired:
        print(f'  [逾時] {filename}')
        return False


def batch_download(df: pd.DataFrame, output_dir: str,
                   year: int = None, month: int = None, latest: int = None):
    os.makedirs(output_dir, exist_ok=True)
    url_col = find_url_column(df)
    year_col, month_col = find_year_month_columns(df)

    target = df.copy()
    if year and year_col:
        target = target[target[year_col].astype(str).str.strip() == str(year)]
        print(f'篩選年份 {year}：{len(target)} 筆')
    if month and month_col and year:
        target = target[target[month_col].astype(str).str.strip() == str(month)]
        print(f'篩選月份 {month}：{len(target)} 筆')
    if latest:
        target = target.tail(latest)
        print(f'取最新 {latest} 個月：{len(target)} 筆')

    if target.empty:
        print('沒有符合條件的資料')
        return

    print(f'\n開始下載 {len(target)} 筆 → {output_dir}/\n')
    success, fail = 0, 0
    for _, row in target.iterrows():
        url = str(row[url_col]).strip()
        if not url.startswith('http'):
            continue
        ok = download_with_wget(url, output_dir)
        success += ok
        fail += not ok

    print(f'\n完成：成功 {success} 個，失敗 {fail} 個')
    files = sorted(os.listdir(output_dir))
    if files:
        print(f'已下載檔案（{len(files)} 個）：')
        for f in files:
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f'  {f} ({size/1024:.1f} KB)')


# ---------------------------------------------------------------------------
# 進出人次資料格式說明
# ---------------------------------------------------------------------------
# 臺北捷運各站進出人次公開資料（data.gov.tw）欄位格式：
#   日期, 站名（中文）, 進站人次, 出站人次
# 月份彙總版（年月統計）欄位格式：
#   年份, 月份, 站名, 進站人次, 出站人次
#
# 注意：各來源的欄位名稱可能有差異（繁簡/空格/底線），
# step3_5_calibration.py 的 load_ridership() 會自動偵測欄位名稱。
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='北捷公開資料批次下載器')
    parser.add_argument('--mode', choices=['od', 'ridership', 'all'],
                        default='od', help='下載模式')
    # OD 資料參數
    parser.add_argument('--od-index', dest='od_index',
                        default=None, help='OD 索引 CSV 路徑')
    parser.add_argument('--od-output', dest='od_output',
                        default='data/od_raw', help='OD 下載目錄')
    # 進出人次參數
    parser.add_argument('--ridership-index', dest='ridership_index',
                        default=None, help='進出人次索引 CSV 路徑')
    parser.add_argument('--ridership-output', dest='ridership_output',
                        default='data/ridership_raw', help='進出人次下載目錄')
    # 舊版相容（--index 直接指 OD 索引）
    parser.add_argument('--index', default=None,
                        help='[相容] OD 索引 CSV 路徑，等同 --od-index')
    # 篩選條件
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--latest', type=int, default=None)
    args = parser.parse_args()

    # 舊版相容
    if args.index and not args.od_index:
        args.od_index = args.index

    if args.mode in ('od', 'all'):
        if not args.od_index:
            parser.error('--mode od 需要 --od-index 或 --index 參數')
        df = load_index(args.od_index)
        batch_download(df, args.od_output, args.year, args.month, args.latest)

    if args.mode in ('ridership', 'all'):
        if not args.ridership_index:
            parser.error('--mode ridership 需要 --ridership-index 參數')
        df = load_index(args.ridership_index)
        batch_download(df, args.ridership_output, args.year, args.month, args.latest)


if __name__ == '__main__':
    main()
