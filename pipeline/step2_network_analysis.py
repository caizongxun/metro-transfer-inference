"""
step2_network_analysis.py
路網拓樸分析：給定 OD 對，找出必要/可能的轉乘站，並計算路徑比例。

修正紀錄：
  - station_mapping.json 的值是 dict（{"id":...,"line":...}），不是 string
    → name_to_id 改為直接從 mapping[name]['id'] 取值
  - network.json 的站名有前綴（"BL板橋", "Y板橋"），OD 資料只有「板橋」
    → build_name_to_ids() 建立模糊對照表，站名去前綴後 match
  - is_transfer 改從 network.json 的 transfer 欄位判斷（原本全為 False）
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
    with open(NETWORK_JSON_PATH, encoding='utf-8') as f:
        network = json.load(f)

    G = nx.Graph()

    # 建立 transfer edges 集合（用 id pair 判斷）
    transfer_pairs = set()
    for t_dict in network.get('transfer_time_minutes', {}).keys():
        # key 格式: "R10_BL11"
        parts = t_dict.split('_', 1)
        if len(parts) == 2:
            transfer_pairs.add((parts[0], parts[1]))

    for station in network.get('stations', []):
        sid = station['id']
        is_transfer = bool(station.get('transfer'))  # 有 transfer 欄位即為轉乘站
        G.add_node(sid,
                   name=station.get('name', sid),
                   line=station.get('line', ''),
                   is_transfer=is_transfer)

    # 先加路線內邊
    transfer_time = network.get('transfer_time_minutes', {})
    line_order = network.get('line_order', {})
    for line_name, order in line_order.items():
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            if G.has_node(a) and G.has_node(b):
                G.add_edge(a, b, weight=2.0, line=line_name, is_transfer_edge=False)

    # 加轉乘邊
    for key, t_min in transfer_time.items():
        parts = key.split('_', 1)
        if len(parts) == 2 and G.has_node(parts[0]) and G.has_node(parts[1]):
            G.add_edge(parts[0], parts[1],
                       weight=t_min, line='transfer', is_transfer_edge=True)

    return G


def load_station_mapping() -> dict:
    """回傳 {站名: id_string}，處理值為 dict 或 string 兩種格式"""
    with open(STATION_MAPPING_PATH, encoding='utf-8') as f:
        raw = json.load(f)
    result = {}
    for name, val in raw.items():
        if isinstance(val, dict):
            result[name] = val['id']
        else:
            result[name] = val
    return result


def build_name_to_ids(G: nx.Graph, mapping: dict) -> dict:
    """
    建立「OD 站名 → 路網節點 ID 列表」對照表。

    處理兩種情況：
    1. mapping 裡的站名直接對應（大部分站）
    2. network.json 中有前綴的站名（"BL板橋" → 「板橋」）
       → 一個 OD 站名對到多個節點 ID（實際上兩個板橋是不同站，
         但 OD 資料不區分，先各取一個權重 0.5）
    """
    name_to_ids = {}

    # 從 mapping 建基礎對照（站名 → id）
    for name, sid in mapping.items():
        clean = name  # mapping key 本來就是 OD 站名
        if sid in G:
            if clean not in name_to_ids:
                name_to_ids[clean] = []
            if sid not in name_to_ids[clean]:
                name_to_ids[clean].append(sid)

    # 補充：從 graph node name 去前綴後反查
    # e.g. node name="BL板橋" → strip prefix → "板橋"
    line_prefixes = ['BL', 'G', 'R', 'O', 'BR', 'V', 'Y']
    for nid, data in G.nodes(data=True):
        node_name = data.get('name', '')
        # 去前綴
        clean_name = node_name
        for pfx in line_prefixes:
            if node_name.startswith(pfx) and len(node_name) > len(pfx):
                candidate = node_name[len(pfx):]
                # 只有去掉後還是中文（非空）才算有效去前綴
                if candidate and not candidate[0].isascii():
                    clean_name = candidate
                    break

        if clean_name != node_name:  # 有去掉前綴
            if clean_name not in name_to_ids:
                name_to_ids[clean_name] = []
            if nid not in name_to_ids[clean_name]:
                name_to_ids[clean_name].append(nid)

    return name_to_ids


def get_transfer_stations_on_path(path: list, G: nx.Graph) -> list:
    """回傳路徑上的轉乘站名稱（排除起訖站）"""
    transfers = []
    for node in path[1:-1]:
        if G.nodes[node].get('is_transfer', False):
            transfers.append(G.nodes[node].get('name', node))
    return transfers


def infer_paths_for_od(origin_ids: list, dest_ids: list,
                       G: nx.Graph, k: int = 3) -> list:
    """
    對 (origin_ids x dest_ids) 所有組合找 k 條最短路徑，
    softmax 分配機率後合併回傳。
    """
    all_results = []
    for oid in origin_ids:
        for did in dest_ids:
            if oid not in G or did not in G or oid == did:
                continue
            try:
                paths = list(nx.shortest_simple_paths(G, oid, did, weight='weight'))[:k]
            except nx.NetworkXNoPath:
                continue

            lengths = [
                sum(G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1))
                for p in paths
            ]
            arr = np.array(lengths)
            scores = np.exp(-arr / (np.mean(arr) + 1e-9))
            probs = scores / scores.sum()

            for path, prob, length in zip(paths, probs, lengths):
                transfers = get_transfer_stations_on_path(path, G)
                path_names = [G.nodes[n].get('name', n) for n in path]
                all_results.append({
                    'path': path_names,
                    'transfer_stations': transfers,
                    'path_prob': round(float(prob) / (len(origin_ids) * len(dest_ids)), 6),
                    'path_length_min': round(length, 1)
                })

    # 重新正規化
    if all_results:
        total = sum(r['path_prob'] for r in all_results)
        for r in all_results:
            r['path_prob'] = round(r['path_prob'] / total, 6)
    return all_results


def analyze_od_dataframe(od_df: pd.DataFrame, G: nx.Graph, mapping: dict) -> pd.DataFrame:
    name_to_ids = build_name_to_ids(G, mapping)

    # 預先計算所有唯一 OD 對的路徑（避免重複計算）
    od_pairs = od_df[['origin', 'destination']].drop_duplicates()
    path_cache = {}

    hit, miss = 0, 0
    for _, row in od_pairs.iterrows():
        o, d = row['origin'], row['destination']
        o_ids = name_to_ids.get(o, [])
        d_ids = name_to_ids.get(d, [])
        if not o_ids or not d_ids:
            path_cache[(o, d)] = None
            miss += 1
            continue
        paths = infer_paths_for_od(o_ids, d_ids, G)
        path_cache[(o, d)] = paths if paths else None
        hit += 1

    print(f'  [Step2] 路徑解析：{hit} 對成功，{miss} 對找不到對應站（站名不在路網）')

    # 展開
    rows = []
    for (o, d), grp in od_df.groupby(['origin', 'destination']):
        paths = path_cache.get((o, d))
        if not paths:
            for _, r in grp.iterrows():
                rows.append({'origin': o, 'destination': d,
                             'hour': r['hour'], 'trips': r['trips'],
                             'transfer_stations': [], 'path_prob': 1.0,
                             'path_length_min': None})
            continue
        for _, r in grp.iterrows():
            for p in paths:
                rows.append({'origin': o, 'destination': d,
                             'hour': r['hour'],
                             'trips': round(r['trips'] * p['path_prob'], 4),
                             'transfer_stations': p['transfer_stations'],
                             'path_prob': p['path_prob'],
                             'path_length_min': p['path_length_min']})

    return pd.DataFrame(rows)


if __name__ == '__main__':
    G = load_network_graph()
    mapping = load_station_mapping()
    name_to_ids = build_name_to_ids(G, mapping)
    print(f'路網：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊')
    transfer_nodes = [n for n, d in G.nodes(data=True) if d.get('is_transfer')]
    print(f'轉乘站：{len(transfer_nodes)} 個 → {[G.nodes[n]["name"] for n in transfer_nodes]}')
    print(f'站名對照表：{len(name_to_ids)} 個 OD 站名可解析')
    # 測試板橋
    print(f'  板橋 → {name_to_ids.get("板橋", "NOT FOUND")}')
    print(f'  忠孝復興 → {name_to_ids.get("忠孝復興", "NOT FOUND")}')
