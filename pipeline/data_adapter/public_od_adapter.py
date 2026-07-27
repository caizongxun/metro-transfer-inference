"""
public_od_adapter.py
效能關鍵：全向量化 groupby/sum，移除 apply(lambda)
checkpoint: 每處理完一個檔案存一次，不自動删除，需手動呼叫 clear_checkpoint()
"""

import os
import glob
import pickle
import pandas as pd
import numpy as np

CHECKPOINT_PATH = 'pipeline/data/checkpoint_acc.pkl'


def load_public_od(od_dir: str = None, od_path: str = None,
                   filter_years: list = None, filter_yearmonths: list = None,
                   chunk_size: int = 50_000,
                   resume: bool = True) -> pd.DataFrame:
    files = _select_files(od_dir, od_path, filter_yearmonths, filter_years)
    if not files:
        print('[Adapter] 找不到對應資料，使用內建模擬資料')
        return _generate_sample_od()

    trips_acc = {}
    dates_acc = {}
    done_files = set()
    total_rows = 0

    if resume and os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'rb') as f:
                ckpt = pickle.load(f)
            trips_acc  = ckpt.get('trips_acc', {})
            dates_acc  = ckpt.get('dates_acc', {})
            done_files = ckpt.get('done_files', set())
            total_rows = ckpt.get('total_rows', 0)
            print(f'[Adapter] 載入 checkpoint：已處理 {len(done_files)} 個檔案，'
                  f'{total_rows:,} 行，{len(trips_acc):,} 個 key')
        except Exception as e:
            print(f'[Adapter] checkpoint 讀取失敗 ({e})，重新開始')

    pending = [f for f in files if os.path.basename(f) not in done_files]
    print(f'[Adapter] 待處理 {len(pending)}/{len(files)} 個檔案 (chunk={chunk_size:,})')

    try:
        from tqdm import tqdm
        file_iter = tqdm(pending, desc='files', unit='file')
    except ImportError:
        file_iter = pending

    for fpath in file_iter:
        fname = os.path.basename(fpath)
        try:
            file_rows = 0
            for chunk in pd.read_csv(fpath, encoding='utf-8-sig',
                                     dtype={'時段': str}, chunksize=chunk_size):
                chunk = chunk.copy()
                chunk.columns = chunk.columns.str.strip()
                chunk = _rename_columns(chunk)
                needed = {'origin', 'destination', 'hour', 'trips'}
                if not needed.issubset(chunk.columns):
                    continue
                chunk['trips'] = pd.to_numeric(chunk['trips'], errors='coerce').fillna(0).astype('float32')
                chunk['hour']  = pd.to_numeric(chunk['hour'],  errors='coerce').fillna(0).astype('int8')
                if 'date' in chunk.columns:
                    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                else:
                    chunk['date'] = None
                grp_trips = chunk.groupby(['origin', 'destination', 'hour'], sort=False)['trips'].sum()
                for key, val in grp_trips.items():
                    trips_acc[key] = trips_acc.get(key, 0.0) + float(val)
                if chunk['date'].notna().any():
                    grp_dates = (
                        chunk[chunk['date'].notna()]
                        .groupby(['origin', 'destination', 'hour'], sort=False)['date']
                        .unique()
                    )
                    for key, arr in grp_dates.items():
                        if key in dates_acc:
                            dates_acc[key].update(arr.tolist())
                        else:
                            dates_acc[key] = set(arr.tolist())
                file_rows += len(chunk)

            total_rows += file_rows
            done_files.add(fname)
            print(f'  [{fname[-14:-4]}] {file_rows:,} 行 ✓')

            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            with open(CHECKPOINT_PATH, 'wb') as f:
                pickle.dump({'trips_acc': trips_acc, 'dates_acc': dates_acc,
                             'done_files': done_files, 'total_rows': total_rows}, f)
        except Exception as e:
            print(f'[Adapter] 跳過 {fname}: {e}')

    if not trips_acc:
        print('[Adapter] 彙整後無資料，使用模擬資料')
        return _generate_sample_od()

    keys   = list(trips_acc.keys())
    n_days = [len(dates_acc[k]) if k in dates_acc and dates_acc[k] else 1 for k in keys]
    df = pd.DataFrame({
        'origin':      [k[0] for k in keys],
        'destination': [k[1] for k in keys],
        'hour':        [k[2] for k in keys],
        'trips':       [round(trips_acc[k] / n_days[i], 2) for i, k in enumerate(keys)],
        'n_days':      n_days,
    })
    df = df[df['trips'] > 0]

    # 不自動删除 checkpoint，需手動呼叫 clear_checkpoint()
    print(f'[Adapter] 完成：{total_rows:,} 行 → {len(df):,} 筆典型日'
          f' ({df["origin"].nunique()} 個起站)')
    return df


def clear_checkpoint():
    """手動清除 step1 checkpoint，下次執行會從頭重跑"""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f'[Adapter] 已清除 {CHECKPOINT_PATH}')
    else:
        print('[Adapter] 無 checkpoint 可清除')


def aggregate_to_typical_day(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        return df[['origin', 'destination', 'hour', 'trips']]
    n_days = df['date'].nunique() or 1
    agg = df.groupby(['origin', 'destination', 'hour'])['trips'].sum().reset_index()
    agg['trips'] = (agg['trips'] / n_days).round(2)
    return agg[agg['trips'] > 0]


def _select_files(od_dir, od_path, filter_yearmonths, filter_years):
    files = []
    if od_dir and os.path.isdir(od_dir):
        all_files = sorted(glob.glob(os.path.join(od_dir, '*.csv')))
        if filter_yearmonths:
            files = [f for f in all_files if any(ym in os.path.basename(f) for ym in filter_yearmonths)]
            print(f'[Adapter] 年月篩選 {filter_yearmonths}: {len(files)}/{len(all_files)} 個檔案')
        elif filter_years:
            files = [f for f in all_files if any(str(y) in os.path.basename(f) for y in filter_years)]
            print(f'[Adapter] 年份篩選 {filter_years}: {len(files)}/{len(all_files)} 個檔案')
        else:
            files = all_files
            print(f'[Adapter] 全部 {len(files)} 個檔案')
    elif od_path and os.path.exists(od_path):
        files = [od_path]
    return files


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {'進站': 'origin', '起站': 'origin', '出站': 'destination', '迄站': 'destination',
               '時段': 'hour', '時間': 'hour', '人次': 'trips', '旅次': 'trips', '日期': 'date'}
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


def _generate_sample_od() -> pd.DataFrame:
    np.random.seed(42)
    od_pairs = [
        ('台北車站', '忠孝復興'), ('忠孝復興', '台北車站'),
        ('台北車站', '南京復興'), ('南京復興', '台北車站'),
        ('板橋', '台北車站'), ('台北車站', '板橋'),
        ('新店', '台北車站'), ('台北車站', '新店'),
        ('淡水', '台北車站'), ('台北車站', '淡水'),
    ]
    hour_weights = {6:0.03,7:0.10,8:0.13,9:0.09,10:0.05,11:0.04,12:0.05,
                    13:0.04,14:0.04,15:0.04,16:0.06,17:0.10,18:0.12,
                    19:0.08,20:0.05,21:0.04,22:0.03,23:0.01}
    rows = []
    for o, d in od_pairs:
        base = np.random.randint(800, 3000)
        for h, w in hour_weights.items():
            t = round(base * w * (1 + np.random.normal(0, 0.05)))
            rows.append({'origin': o, 'destination': d, 'hour': h, 'trips': max(0, t), 'n_days': 1})
    df = pd.DataFrame(rows)
    print(f'[Adapter] 使用模擬 OD 資料 ({len(od_pairs)} OD 對, {len(df)} 筆)')
    return df
