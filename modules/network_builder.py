"""
network_builder.py
建立北捷路網圖，產生候選路徑集合（K-Shortest Paths）
"""

import json
import networkx as nx
from itertools import islice


def load_network(network_path: str = "data/network.json") -> dict:
    """載入路網 JSON"""
    with open(network_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(network: dict) -> nx.Graph:
    """
    建立無向加權圖
    節點 = 站點 ID
    邊 = 相鄰站（同線）或轉乘關係
    權重 = 行駛時間（分鐘），轉乘加步行懲罰
    """
    G = nx.Graph()
    stations = network["stations"]
    transfer_times = network.get("transfer_time_minutes", {})

    # 同線站點依序連接（簡化：假設站間 2 分鐘行駛時間）
    line_stations = {}
    for s in stations:
        line = s["line"]
        line_stations.setdefault(line, []).append(s["id"])
        G.add_node(s["id"], name=s["name"], line=line)

    for line, sids in line_stations.items():
        for i in range(len(sids) - 1):
            G.add_edge(sids[i], sids[i + 1], weight=2.0, edge_type="ride")

    # 轉乘關係
    for s in stations:
        if "transfer" in s:
            for t_id in s["transfer"]:
                key1 = f"{s['id']}_{t_id}"
                key2 = f"{t_id}_{s['id']}"
                walk_time = transfer_times.get(key1) or transfer_times.get(key2, 3.0)
                if not G.has_edge(s["id"], t_id):
                    G.add_edge(s["id"], t_id, weight=walk_time, edge_type="transfer")

    return G


def get_k_shortest_paths(G: nx.Graph, origin: str, destination: str, k: int = 5) -> list:
    """
    產生 K 條最短路徑（候選路徑集合）
    回傳: [(path_node_list, total_weight), ...]
    """
    try:
        paths = list(islice(nx.shortest_simple_paths(G, origin, destination, weight="weight"), k))
    except nx.NetworkXNoPath:
        return []

    result = []
    for path in paths:
        total_weight = sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))
        result.append({"path": path, "theory_time": total_weight})

    return result


def get_transfer_stations(G: nx.Graph, path: list) -> list:
    """找出路徑中的轉乘站"""
    transfers = []
    for i in range(len(path) - 1):
        edge = G[path[i]][path[i + 1]]
        if edge.get("edge_type") == "transfer":
            transfers.append((path[i], path[i + 1]))
    return transfers


if __name__ == "__main__":
    network = load_network()
    G = build_graph(network)
    print(f"路網建立完成：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊")
    paths = get_k_shortest_paths(G, "BL05", "R09", k=3)
    for i, p in enumerate(paths):
        print(f"路徑 {i+1}: {' -> '.join(p['path'])}，理論時間 {p['theory_time']:.1f} 分鐘")
