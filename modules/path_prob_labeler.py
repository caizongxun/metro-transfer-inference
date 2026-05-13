"""
path_prob_labeler.py  (OOM 優化版)
路徑機率標記器：
    對每個 (進站, 出站) OD 對，
    用 path_inference 算出各轉乘站的期望流量貢獻。

根本原因不能直接轉為行展開計算：
    1. path_inference 需要網路圖 G，個別展開体積大
    2. 每個 OD 對的候選路徑數量有限（K=5），結果可緩存
    3. 所以於相同 OD 對只計算一次，粘贴到全部時段
"""

import gc
import pandas as pd
import numpy as np
from collections import defaultdict
from modules.network_builder import load_network, build_graph
from modules.path_inference import infer_path_probabilities
from modules.station_mapper import load_mapping, name_to_id


def build_transfer_flow_labels(
    od_df: pd.DataFrame,
    G,
    mapping: dict,
) -> pd.DataFrame:
    """
    輸入：含 (日期, 時段, 進站, 出站, 人次) 的 DataFrame
    輸出：每個 (transfer_station, 日期, 時段) 的估算流量

    計算方式：
        expected_flow(T) = Σ_{OD} 人次(OD) × P(路徑經過T | OD)

    OOM 優化：
        - path_prob 對同一 OD 對只計算一次（cache by (origin, dest) 名稱）
        - 結果直接累積到 aggregation dict，不產生大型中間 DataFrame
        - 每處理 10k 筆就 gc.collect() 一次
    """
    path_prob_cache = {}  # {(origin_name, dest_name): {t_station: prob}}

    def get_cached_probs(origin_name, dest_name):
        key = (origin_name, dest_name)
        if key in path_prob_cache:
            return path_prob_cache[key]
        oid = name_to_id(origin_name, mapping)
        did = name_to_id(dest_name, mapping)
        if oid is None or did is None or oid == did:
            path_prob_cache[key] = {}
            return {}
        try:
            paths = infer_path_probabilities(
                origin=oid, destination=did,
                actual_time=None, G=G, k=5
            )
        except Exception:
            path_prob_cache[key] = {}
            return {}
        result = defaultdict(float)
        for p in paths:
            for (_, sta_b) in p.get("transfer_stations", []):
                result[sta_b] += p["prob"]
        path_prob_cache[key] = dict(result)
        return path_prob_cache[key]

    # 累積字典：{(date, hour, transfer_station): expected_flow}
    agg_dict = defaultdict(float)

    df = od_df[od_df["人次"] > 0].reset_index(drop=True)
    total = len(df)
    print(f"  共 {total:,} 筆 OD，開始產生標記...")

    for i, row in enumerate(df.itertuples(index=False), 1):
        if i % 20000 == 0:
            print(f"  進度：{i:,}/{total:,} ({i/total*100:.1f}%) "
                  f"| cache大小：{len(path_prob_cache)}")
            gc.collect()

        transfer_probs = get_cached_probs(row.進站, row.出站)
        if not transfer_probs:
            continue

        date_str = str(row.日期)[:10]
        hour     = int(row.時段)
        trips    = float(row.人次)

        for t_station, prob in transfer_probs.items():
            agg_dict[(date_str, hour, t_station)] += trips * prob

    # 轉成 DataFrame
    records = [
        {"日期": k[0], "hour": k[1], "transfer_station": k[2], "expected_flow": v}
        for k, v in agg_dict.items()
    ]
    del agg_dict, path_prob_cache
    gc.collect()

    return pd.DataFrame(records)
