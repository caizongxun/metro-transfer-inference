"""
path_prob_labeler.py
路徑機率標記器：
    對每個 (進站, 出站) OD 對，
    用 path_inference 算出各轉乘站的期望流量貢獻。

輸出格式（用於 ML label 生成）：
    transfer_station | hour | weekday | expected_flow

這個 label 就是 XGBoost 要學的目標值。
"""

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
    top_n_od: int = 50
) -> pd.DataFrame:
    """
    輸入：含 (日期, 時段, 進站, 出站, 人次) 的 DataFrame
    輸出：每個 (transfer_station, 日期, 時段) 的估算流量

    計算方式：
        expected_flow(T) = Σ_{OD} 人次(OD) × P(路徑經過T | OD)
    """
    # 先對 OD 對快取路徑機率（避免重複計算）
    path_prob_cache = {}

    def get_cached_probs(origin_name, dest_name):
        key = (origin_name, dest_name)
        if key in path_prob_cache:
            return path_prob_cache[key]
        oid = name_to_id(origin_name, mapping)
        did = name_to_id(dest_name, mapping)
        paths = infer_path_probabilities(
            origin=oid, destination=did,
            actual_time=None, G=G, k=5
        )
        # 整理成 {transfer_station: 機率} dict
        result = defaultdict(float)
        for p in paths:
            for (sta_a, sta_b) in p["transfer_stations"]:
                result[sta_b] += p["prob"]
        path_prob_cache[key] = dict(result)
        return path_prob_cache[key]

    # 只取有人次的 OD
    df = od_df[od_df["人次"] > 0].copy()

    rows = []
    total = len(df)
    print(f"  計算標記中，共 {total:,} 筆 OD 紀錄...")

    # 分組逐筆計算
    for i, row in enumerate(df.itertuples(index=False), 1):
        if i % 50000 == 0:
            print(f"  進度：{i:,}/{total:,} ({i/total*100:.1f}%)")

        origin = row.進站
        dest   = row.出站
        if origin == dest:
            continue

        transfer_probs = get_cached_probs(origin, dest)
        trip_count = row.人次

        for t_station, prob in transfer_probs.items():
            rows.append({
                "日期":              str(row.日期)[:10],
                "hour":             row.時段,
                "transfer_station": t_station,
                "origin":           origin,
                "destination":      dest,
                "trips":            trip_count,
                "path_prob":        prob,
                "expected_flow":    trip_count * prob,
            })

    label_df = pd.DataFrame(rows)

    # 聚合：同一轉乘站 + 時段 的總期望流量
    agg = (
        label_df
        .groupby(["日期", "hour", "transfer_station"])
        .agg(expected_flow=("expected_flow", "sum"))
        .reset_index()
    )
    return agg


if __name__ == "__main__":
    import glob
    from modules.afc_processor import load_od_csv, filter_by_timeslot

    network = load_network()
    G = build_graph(network)
    mapping = load_mapping()

    files = sorted(glob.glob("data/od_raw/*.csv"))
    if not files:
        print("請先執行 data_fetcher.py")
    else:
        df = load_od_csv(files[-1])  # 最新一個月
        peak = filter_by_timeslot(df, 7, 9)
        labels = build_transfer_flow_labels(peak, G, mapping)
        print("\n標記結果（前10行）：")
        print(labels.sort_values("expected_flow", ascending=False).head(10).to_string(index=False))
