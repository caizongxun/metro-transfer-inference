"""
data_fetcher.py
從臺北捷運 OD 流量索引 CSV 讀取 URL，批次用 wget 下載月份資料

使用方式：
    python data_fetcher.py --index 臺北捷運每日分時各站OD流量統計資料.csv
    python data_fetcher.py --index 臺北捷運每日分時各站OD流量統計資料.csv --year 2024
    python data_fetcher.py --index 臺北捷運每日分時各站OD流量統計資料.csv --year 2024 --month 1
    python data_fetcher.py --index 臺北捷運每日分時各站OD流量統計資料.csv --latest 3
"""

import argparse
import os
import subprocess
import pandas as pd


def load_index(index_path: str) -> pd.DataFrame:
    """讀取索引 CSV，自動偵測欄位名稱"""
    df = pd.read_csv(index_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    print(f"索引欄位：{list(df.columns)}")
    print(f"共 {len(df)} 筆紀錄（月份）")
    return df


def find_url_column(df: pd.DataFrame) -> str:
    """自動找出 URL 欄位名稱"""
    for col in df.columns:
        sample = str(df[col].dropna().iloc[0])
        if sample.startswith("http"):
            return col
    raise ValueError("找不到 URL 欄位，請確認索引 CSV 格式")


def find_year_month_columns(df: pd.DataFrame):
    """自動找出年份與月份欄位"""
    year_col, month_col = None, None
    for col in df.columns:
        col_lower = col.strip().lower()
        if "年" in col or "year" in col_lower:
            year_col = col
        if "月" in col or "month" in col_lower:
            month_col = col
    return year_col, month_col


def download_with_wget(url: str, output_dir: str, filename: str = None):
    """用 wget 下載單一檔案"""
    url = url.strip()
    if not filename:
        filename = url.split("/")[-1]

    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"  [跳過] 已存在：{filename}")
        return True

    print(f"  [下載] {filename}")
    cmd = ["wget", "-q", "-O", output_path, url]

    try:
        result = subprocess.run(cmd, timeout=120, capture_output=True, text=True)
        if result.returncode == 0:
            size = os.path.getsize(output_path)
            print(f"  [完成] {filename} ({size/1024:.1f} KB)")
            return True
        else:
            print(f"  [失敗] {filename}: {result.stderr.strip()}")
            # 刪除空檔
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    except subprocess.TimeoutExpired:
        print(f"  [逾時] {filename}")
        return False


def main():
    parser = argparse.ArgumentParser(description="北捷 OD 資料批次下載器")
    parser.add_argument("--index", required=True, help="索引 CSV 路徑")
    parser.add_argument("--output", default="data/od_raw", help="下載目錄（預設 data/od_raw）")
    parser.add_argument("--year", type=int, default=None, help="只下載指定年份，例如 2024")
    parser.add_argument("--month", type=int, default=None, help="搭配 --year 指定月份")
    parser.add_argument("--latest", type=int, default=None, help="只下載最新 N 個月，例如 --latest 3")
    args = parser.parse_args()

    # 建立輸出目錄
    os.makedirs(args.output, exist_ok=True)

    # 讀取索引
    df = load_index(args.index)
    url_col = find_url_column(df)
    year_col, month_col = find_year_month_columns(df)

    print(f"\nURL 欄位：{url_col}")
    print(f"年份欄位：{year_col}，月份欄位：{month_col}")

    # 篩選條件
    target = df.copy()

    if args.year and year_col:
        target = target[target[year_col].astype(str).str.strip() == str(args.year)]
        print(f"\n篩選年份 {args.year}：{len(target)} 筆")

    if args.month and month_col and args.year:
        target = target[target[month_col].astype(str).str.strip() == str(args.month)]
        print(f"篩選月份 {args.month}：{len(target)} 筆")

    if args.latest:
        target = target.tail(args.latest)
        print(f"\n取最新 {args.latest} 個月：{len(target)} 筆")

    if target.empty:
        print("沒有符合條件的資料，請確認篩選參數")
        return

    # 批次下載
    print(f"\n開始下載 {len(target)} 個月的 OD 資料 → {args.output}/\n")
    success, fail = 0, 0

    for _, row in target.iterrows():
        url = str(row[url_col]).strip()
        if not url.startswith("http"):
            print(f"  [跳過] 非有效 URL：{url}")
            continue

        ok = download_with_wget(url, args.output)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\n完成：成功 {success} 個，失敗 {fail} 個")
    print(f"檔案存放於：{os.path.abspath(args.output)}")

    # 列出已下載檔案
    files = sorted(os.listdir(args.output))
    if files:
        print(f"\n已下載檔案（{len(files)} 個）：")
        for f in files:
            size = os.path.getsize(os.path.join(args.output, f))
            print(f"  {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
