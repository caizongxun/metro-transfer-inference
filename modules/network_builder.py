"""
network_builder.py
建立北捷路網圖，產生候選路徑集合（K-Shortest Paths）
"""

import json
import networkx as nx
from itertools import islice

# 候選路徑最多允許超出最短路徑幾個站
# 避免 k-shortest 找出「跑回頭再繞過來」的無意義長路
MAX_EXTRA_STATIONS = 4


def load_network(network_path: str = "data/network.json") -> dict:
    """載入路網 JSON"""
    with open(network_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(network: dict) -> nx.Graph:
    """
    建立無向加權圖
    - 節點 = 站點 ID
    - 邊 = 相鄰站（同線）或轉乘關係
    - 權重 = 行駛時間（分鐘），轉乘加步行懲罰
    使用 line_order 確保算出正確的站間順序
    """
    G = nx.Graph()
    stations       = network["stations"]
    line_order     = network.get("line_order", {})
    transfer_times = network.get("transfer_time_minutes", {})

    station_map = {s["id"]: s for s in stations}

    for s in stations:
        G.add_node(s["id"], name=s.get("name", s["id"]), line=s["line"])

    DEFAULT_RIDE_TIME = 2.0
    for line, ordered_ids in line_order.items():
        for i in range(len(ordered_ids) - 1):
            a, b = ordered_ids[i], ordered_ids[i + 1]
            if a in station_map and b in station_map:
                G.add_edge(a, b, weight=DEFAULT_RIDE_TIME, edge_type="ride", line=line)

    for s in stations:
        if "transfer" in s:
            for t_id in s["transfer"]:
                if t_id not in station_map:
                    continue
                if G.has_edge(s["id"], t_id):
                    continue
                key1 = f"{s['id']}_{t_id}"
                key2 = f"{t_id}_{s['id']}"
                walk = transfer_times.get(key1) or transfer_times.get(key2, 3.0)
                G.add_edge(s["id"], t_id, weight=walk, edge_type="transfer")

    return G


def get_k_shortest_paths(G: nx.Graph, origin: str, destination: str, k: int = 5) -> list:
    """
    產生 K 條最短路徑（候選路徑集合）

    過濾條件：
    - 路徑站數不超過最短路徑站數 + MAX_EXTRA_STATIONS
      → 避免「繞一大圈」的無意義替代路出現在候選集中

    回傳: [{"path": [...], "theory_time": float}, ...]
    """
    if origin not in G:
        raise nx.NodeNotFound(f"origin node {origin} not in graph")
    if destination not in G:
        raise nx.NodeNotFound(f"target node {destination} not in graph")

    try:
        # 先撈比 k 更多條，再用站數過濾後取前 k 條
        raw_paths = list(islice(
            nx.shortest_simple_paths(G, origin, destination, weight="weight"),
            k * 4  # 先取 4x 候選數量，確保過濾後仍有足夠選擇
        ))
    except nx.NetworkXNoPath:
        return []

    if not raw_paths:
        return []

    min_stations = len(raw_paths[0])
    max_allowed  = min_stations + MAX_EXTRA_STATIONS

    result = []
    for path in raw_paths:
        if len(path) > max_allowed:
            continue
        total = sum(G[path[i]][path[i+1]]["weight"] for i in range(len(path)-1))
        result.append({"path": path, "theory_time": total})
        if len(result) >= k:
            break

    return result


def get_transfer_stations(G: nx.Graph, path: list) -> list:
    """找出路徑中的轉乘站"""
    transfers = []
    for i in range(len(path) - 1):
        edge = G[path[i]][path[i+1]]
        if edge.get("edge_type") == "transfer":
            transfers.append((path[i], path[i+1]))
    return transfers


if __name__ == "__main__":
    network = load_network()
    G = build_graph(network)
    print(f"路網建立完成：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊")

    test_paths = get_k_shortest_paths(G, "BR14", "BL17X", k=3)
    for i, p in enumerate(test_paths):
        print(f"路徑 {i+1}: {' -> '.join(p['path'])}，理論時間 {p['theory_time']:.1f} 分鐘")
