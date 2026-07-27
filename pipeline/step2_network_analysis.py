"""
step2_network_analysis.py
路網拓樸分析：給定 OD 對，找出必要/可能的轉乘站，並計算路徑比例。

輸出格式（DataFrame）：
  origin, destination, hour, trips,
  transfer_stations (list),   # 這條路徑經過的轉乘站
  path_prob                   # 這條路徑的機率（當有多條路徑時）

目前使用「必要轉乘站法」：
  - 若 OD 跨路線，計算所有可能路徑並分配比例
  - 單一路徑的 OD → prob=1.0
  - 多路徑 OD → 依路徑長度 softmax 分配（等私有資料後可換成旅行時間匹配）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import networkx as nx
from pipeline.config import NETWORK_JSON_PATH, STATION_MAPPING_PATH


def load_network_graph() -> nx.Graph:
    """載入路網圖"""
    with open(NETWORK_JSON_PATH, encoding='utf-8') as f:
        network = json.load(f)

    G = nx.Graph()
    for station in network.get('stations', []):
        G.add_node(station['id'], name=station.get('name', station['id']),
                   line=station.get('line', ''), is_transfer=station.get('is_transfer', False))

    for edge in network.get('edges', []):
        G.add_edge(edge['from'], edge['to'],
                   weight=edge.get('travel_time', 2),
                   line=edge.get('line', ''))
    return G


def load_station_mapping() -> dict:
    """站名 → 站 ID 對照表"""
    with open(STATION_MAPPING_PATH, encoding='utf-8') as f:
        return json.load(f)


def get_transfer_stations_on_path(path: list, G: nx.Graph) -> list:
    """回傳路徑上的轉乘站（is_transfer=True 的站，排除起訖站）"""
    transfers = []
    for node in path[1:-1]:
        if G.nodes[node].get('is_transfer', False):
            transfers.append(G.nodes[node].get('name', node))
    return transfers


def infer_paths_for_od(origin_id: str, dest_id: str, G: nx.Graph, k: int = 3) -> list:
    """
    找 k 條最短路徑，用路徑長度 softmax 分配機率。
    回傳 list of dict: {path, transfer_stations, path_prob}
    """
    if origin_id not in G or dest_id not in G:
        return []
    if origin_id == dest_id:
        return []

    try:
        paths = list(nx.shortest_simple_paths(G, origin_id, dest_id, weight='weight'))
        paths = paths[:k]
    except nx.NetworkXNoPath:
        return []

    # 路徑長度（加權）
    lengths = []
    for p in paths:
        length = sum(G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1))
        lengths.append(length)

    # softmax：越短的路徑機率越高（用負值）
    lengths_arr = np.array(lengths)
    scores = np.exp(-lengths_arr / np.mean(lengths_arr))
    probs = scores / scores.sum()

    result = []
    for path, prob in zip(paths, probs):
        transfers = get_transfer_stations_on_path(path, G)
        path_names = [G.nodes[n].get('name', n) for n in path]
        result.append({
            'path': path_names,
            'transfer_stations': transfers,
            'path_prob': round(float(prob), 4),
            'path_length_min': round(lengths[paths.index(path)], 1)
        })
    return result


def analyze_od_dataframe(od_df: pd.DataFrame, G: nx.Graph, mapping: dict) -> pd.DataFrame:
    """
    對整個 OD DataFrame 做路徑分析。
    輸出：每個 OD-時段-路徑 一筆，含 transfer_stations 和 path_prob。
    """
    name_to_id = {v: k for k, v in mapping.items()} if isinstance(list(mapping.values())[0], str) else mapping

    rows = []
    od_pairs = od_df.groupby(['origin', 'destination'])

    for (origin_name, dest_name), group in od_pairs:
        origin_id = name_to_id.get(origin_name, origin_name)
        dest_id = name_to_id.get(dest_name, dest_name)

        paths = infer_paths_for_od(origin_id, dest_id, G)

        if not paths:
            # 無法在路網中找到路徑（站名對照問題），保留原始資料
            for _, row in group.iterrows():
                rows.append({
                    'origin': origin_name, 'destination': dest_name,
                    'hour': row['hour'], 'trips': row['trips'],
                    'transfer_stations': [], 'path_prob': 1.0,
                    'path_length_min': None
                })
            continue

        for _, row in group.iterrows():
            for p in paths:
                rows.append({
                    'origin': origin_name, 'destination': dest_name,
                    'hour': row['hour'],
                    'trips': row['trips'] * p['path_prob'],  # 依比例分配旅次
                    'transfer_stations': p['transfer_stations'],
                    'path_prob': p['path_prob'],
                    'path_length_min': p['path_length_min']
                })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    G = load_network_graph()
    mapping = load_station_mapping()
    print(f"路網：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊")
    transfer_nodes = [n for n, d in G.nodes(data=True) if d.get('is_transfer')]
    print(f"轉乘站：{len(transfer_nodes)} 個")
