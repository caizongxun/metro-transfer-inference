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

# 懲罰係數（可調整）
ALPHA_TIME     = 1.0   # 旅行時間權重
BETA_TRANSFER  = 2.0   # 轉乘次數懲罰（分鐘）
GAMMA_CROWDING = 1.5   # 擁擠度懲罰（每級別的額外分鐘）

# 擁擠度對應分數
CROWDING_MAP = {
    "green":  0,
    "yellow": 1,
    "orange": 2,
    "red":    3
}


def get_crowding_penalty(path: list, crowding_df: pd.DataFrame, timestamp: str) -> float:
    """
    計算路徑上的擁擠懲罰分數
    簡化：取路徑中所有站點最大擁擠度 × GAMMA
    """
    if crowding_df is None or crowding_df.empty:
        return 0.0

    relevant = crowding_df[crowding_df["station_id"].isin(path)]
    if relevant.empty:
        return 0.0

    max_crowding = relevant["crowding_score"].max()
    return max_crowding * GAMMA_CROWDING


def compute_path_score(path_info: dict, actual_time: float,
                       G: nx.Graph, crowding_df: pd.DataFrame,
                       timestamp: str = None) -> float:
    """
    計算單條候選路徑的效用分數（負值越低越好）

    分數 = -ALPHA * time_diff^2 - BETA * transfer_count - crowding_penalty
    time_diff：實際旅時與理論旅時的差距（越小越好）
    """
    path = path_info["path"]
    theory_time = path_info["theory_time"]
    time_diff = abs(actual_time - theory_time)

    transfers = get_transfer_stations(G, path)
    transfer_penalty = BETA_TRANSFER * len(transfers)

    crowding_penalty = get_crowding_penalty(path, crowding_df, timestamp)

    score = -(ALPHA_TIME * time_diff + transfer_penalty + crowding_penalty)
    return score


def infer_path_probabilities(origin: str, destination: str,
                             actual_time: float,
                             G: nx.Graph,
                             crowding_df: pd.DataFrame = None,
                             timestamp: str = None,
                             k: int = 5) -> list:
    """
    主推論函式：回傳候選路徑機率列表

    回傳格式：
    [
      { "path": [...], "theory_time": 20.5, "score": -1.2, "prob": 0.68, "transfer_stations": [...] },
      ...
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
    scores_shifted = scores - scores.max()  # 數值穩定
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / exp_scores.sum()

    for c, p in zip(candidates, probs):
        c["prob"] = round(float(p), 4)

    # 依機率降序排列
    candidates.sort(key=lambda x: x["prob"], reverse=True)
    return candidates


if __name__ == "__main__":
    from modules.network_builder import load_network, build_graph
    import pandas as pd

    network = load_network()
    G = build_graph(network)
    crowding_df = pd.read_csv("data/sample_crowding.csv")

    result = infer_path_probabilities(
        origin="BL05",
        destination="R09",
        actual_time=25.75,
        G=G,
        crowding_df=crowding_df,
        k=3
    )

    print("路徑推論結果 (BL05 → R09)：")
    for r in result:
        transfers = [f"{a}->{b}" for a, b in r["transfer_stations"]]
        print(f"  路徑: {' -> '.join(r['path'])}")
        print(f"  理論時間: {r['theory_time']:.1f} min | 機率: {r['prob']*100:.1f}%")
        print(f"  轉乘: {transfers if transfers else '無'}")
        print()
