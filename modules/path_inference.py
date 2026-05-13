"""
path_inference.py
核心推論模組：
1. 計算候選路徑的理論旅行時間分布
2. 根據實際 AFC 旅行時間計算路徑機率（Softmax MNL）
3. 加入擁擠度懲罰
4. 輸出每筆旅次的路徑機率
"""

import pandas as pd
import numpy as np
import networkx as nx
from modules.network_builder import get_k_shortest_paths, get_transfer_stations

# 懲罰係數

ALPHA_TIME     = 1.0   # 旅行時間權重
BETA_TRANSFER  = 2.0   # 轉乘次數懲罰（分鐘）
GAMMA_CROWDING = 1.5   # 擁擠度懲罰分數

CROWDING_MAP = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


def get_crowding_penalty(path: list, crowding_df, timestamp: str) -> float:
    """  計算路徑上的擁擠懲罰分數 """
    if crowding_df is None:
        return 0.0
    try:
        relevant = crowding_df[crowding_df["station_id"].isin(path)]
        if relevant.empty:
            return 0.0
        return float(relevant["crowding_score"].max()) * GAMMA_CROWDING
    except Exception:
        return 0.0


def compute_path_score(path_info: dict, actual_time,
                       G: nx.Graph, crowding_df,
                       timestamp: str = None) -> float:
    """
    計算單條候選路徑的效用分數（負値越低越好）

    有 actual_time：
        score = -ALPHA * |actual - theory|  - BETA * n_transfers - crowding
    無 actual_time（不分析單一旅次、只看 OD 對總量）：
        score = -BETA * n_transfers - crowding
        → 轉乘次數最少的路徑機率最高（符合旅客自然選擇行為）
    """
    path        = path_info["path"]
    theory_time = path_info["theory_time"]

    if actual_time is not None:
        time_penalty = ALPHA_TIME * abs(float(actual_time) - theory_time)
    else:
        time_penalty = 0.0

    transfers        = get_transfer_stations(G, path)
    transfer_penalty = BETA_TRANSFER * len(transfers)
    crowding_penalty = get_crowding_penalty(path, crowding_df, timestamp)

    return -(time_penalty + transfer_penalty + crowding_penalty)


def infer_path_probabilities(origin: str, destination: str,
                             actual_time,
                             G: nx.Graph,
                             crowding_df=None,
                             timestamp: str = None,
                             k: int = 5) -> list:
    """
    主推論函式：回傳候選路徑機率列表

    回傳格式：
    [
      {
        "path": [...],
        "theory_time": 20.5,
        "score": -1.2,
        "prob": 0.68,
        "transfer_stations": [...]
      }, ...
    ]
    """
    candidates = get_k_shortest_paths(G, origin, destination, k=k)
    if not candidates:
        return []

    for c in candidates:
        c["score"] = compute_path_score(c, actual_time, G, crowding_df, timestamp)
        c["transfer_stations"] = get_transfer_stations(G, c["path"])

    # Softmax 轉成機率
    scores = np.array([c["score"] for c in candidates])
    exp_s  = np.exp(scores - scores.max())
    probs  = exp_s / exp_s.sum()

    for c, p in zip(candidates, probs):
        c["prob"] = round(float(p), 4)

    candidates.sort(key=lambda x: x["prob"], reverse=True)
    return candidates


if __name__ == "__main__":
    from modules.network_builder import load_network, build_graph

    network = load_network()
    G = build_graph(network)
    print(f"路網：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊")

    # 有 actual_time 的對照
    print("\n模式 A（有實際旅行時間）")
    r = infer_path_probabilities("BR14", "BL17X", actual_time=30.0, G=G, k=5)
    for x in r:
        t = [f"{a}->{b}" for a, b in x["transfer_stations"]]
        print(f"  {' -> '.join(x['path'])} | {x['theory_time']:.1f}min | {x['prob']*100:.1f}% | 轉: {t or '直達'}")

    # 無 actual_time（OD 總量模式）
    print("\n模式 B（僅達 OD 總量，不知旅行時間）")
    r = infer_path_probabilities("G01", "BL17X", actual_time=None, G=G, k=5)
    for x in r:
        t = [f"{a}->{b}" for a, b in x["transfer_stations"]]
        print(f"  {' -> '.join(x['path'])} | {x['theory_time']:.1f}min | {x['prob']*100:.1f}% | 轉: {t or '直達'}")
