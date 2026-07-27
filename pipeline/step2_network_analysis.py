"""
step2_network_analysis.py
路網拓樸分析：給定 OD 對，找出對應轉乘站並計算路徑比例。

checkpoint: 路徑結果存 pickle，中斷後可跟過已計算的 OD pair
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pipeline.config import NETWORK_JSON_PATH, STATION_MAPPING_PATH

CHECKPOINT_PATH = 'pipeline/data/checkpoint_step2.pkl'


def load_network_graph() -> nx.Graph:
    with open(NETWORK_JSON_PATH, encoding='utf-8') as f:
        network = json.load(f)
    G = nx.Graph()
    for station in network.get('stations', []):
        G.add_node(station['id'],
                   name=station.get('name', station['id']),
                   line=station.get('line', ''),
                   is_transfer=bool(station.get('transfer')))
    transfer_time = network.get('transfer_time_minutes', {})
    line_order = network.get('line_order', {})
    for line_name, order in line_order.items():
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            if G.has_node(a) and G.has_node(b):
                G.add_edge(a, b, weight=2.0, line=line_name, is_transfer_edge=False)
    for key, t_min in transfer_time.items():
        parts = key.split('_', 1)
        if len(parts) == 2 and G.has_node(parts[0]) and G.has_node(parts[1]):
            G.add_edge(parts[0], parts[1], weight=t_min, line='transfer', is_transfer_edge=True)
    return G


def load_station_mapping() -> dict:
    with open(STATION_MAPPING_PATH, encoding='utf-8') as f:
        raw = json.load(f)
    return {name: (val['id'] if isinstance(val, dict) else val) for name, val in raw.items()}


def build_name_to_ids(G: nx.Graph, mapping: dict) -> dict:
    name_to_ids = {}
    for name, sid in mapping.items():
        if sid in G:
            name_to_ids.setdefault(name, []).append(sid)
    line_prefixes = ['BL', 'BR', 'G', 'R', 'O', 'V', 'Y']
    for nid, data in G.nodes(data=True):
        node_name = data.get('name', '')
        for pfx in line_prefixes:
            if node_name.startswith(pfx) and len(node_name) > len(pfx):
                candidate = node_name[len(pfx):]
                if candidate and not candidate[0].isascii():
                    name_to_ids.setdefault(candidate, [])
                    if nid not in name_to_ids[candidate]:
                        name_to_ids[candidate].append(nid)
                    break
    return name_to_ids


def get_transfer_stations_on_path(path: list, G: nx.Graph) -> list:
    return [
        G.nodes[node].get('name', node)
        for node in path[1:-1]
        if G.nodes[node].get('is_transfer', False)
    ]


def infer_paths_for_od(origin_ids: list, dest_ids: list, G: nx.Graph, k: int = 3) -> list:
    all_results = []
    for oid in origin_ids:
        for did in dest_ids:
            if oid not in G or did not in G or oid == did:
                continue
            try:
                paths = list(nx.shortest_simple_paths(G, oid, did, weight='weight'))[:k]
            except nx.NetworkXNoPath:
                continue
            lengths = [sum(G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)) for p in paths]
            arr = np.array(lengths, dtype=float)
            scores = np.exp(-arr / (arr.mean() + 1e-9))
            probs = scores / scores.sum()
            for path, prob, length in zip(paths, probs, lengths):
                all_results.append({
                    'path': [G.nodes[n].get('name', n) for n in path],
                    'transfer_stations': get_transfer_stations_on_path(path, G),
                    'path_prob': float(prob),
                    'path_length_min': round(length, 1)
                })
    if all_results:
        total = sum(r['path_prob'] for r in all_results)
        for r in all_results:
            r['path_prob'] = round(r['path_prob'] / total, 6)
    return all_results


def analyze_od_dataframe(od_df: pd.DataFrame, G: nx.Graph, mapping: dict,
                         resume: bool = True) -> pd.DataFrame:
    name_to_ids = build_name_to_ids(G, mapping)

    # 所有唯一 OD pair
    od_pairs = od_df[['origin', 'destination']].drop_duplicates().values.tolist()
    total_pairs = len(od_pairs)

    # 載入 checkpoint
    path_cache = {}
    if resume and os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'rb') as f:
                path_cache = pickle.load(f)
            print(f'  [Step2] 載入 checkpoint：{len(path_cache)} / {total_pairs} OD pair 已計算')
        except Exception as e:
            print(f'  [Step2] checkpoint 讀取失敗 ({e})，重新開始')
            path_cache = {}

    pending = [(o, d) for o, d in od_pairs if (o, d) not in path_cache]
    print(f'  [Step2] 待計算 {len(pending)} / {total_pairs} OD pair')

    # tqdm 進度條
    try:
        from tqdm import tqdm
        pair_iter = tqdm(pending, desc='OD pairs', unit='pair')
    except ImportError:
        pair_iter = pending

    hit, miss = 0, 0
    SAVE_INTERVAL = 500  # 每 500 pair 存一次

    for i, (o, d) in enumerate(pair_iter):
        o_ids = name_to_ids.get(o, [])
        d_ids = name_to_ids.get(d, [])
        if not o_ids or not d_ids:
            path_cache[(o, d)] = None
            miss += 1
        else:
            paths = infer_paths_for_od(o_ids, d_ids, G)
            path_cache[(o, d)] = paths if paths else None
            hit += 1

        if (i + 1) % SAVE_INTERVAL == 0:
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            with open(CHECKPOINT_PATH, 'wb') as f:
                pickle.dump(path_cache, f)

    # 最後存一次
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(path_cache, f)

    print(f'  [Step2] 路徑解析：{hit} 對成功，{miss} 對找不到對應站')

    # 展開成路徑 DataFrame
    rows = []
    for (o, d), grp in od_df.groupby(['origin', 'destination']):
        paths = path_cache.get((o, d))
        if not paths:
            for _, r in grp.iterrows():
                rows.append({'origin': o, 'destination': d, 'hour': r['hour'],
                             'trips': r['trips'], 'transfer_stations': [],
                             'path_prob': 1.0, 'path_length_min': None})
            continue
        for _, r in grp.iterrows():
            for p in paths:
                rows.append({'origin': o, 'destination': d, 'hour': r['hour'],
                             'trips': round(r['trips'] * p['path_prob'], 4),
                             'transfer_stations': p['transfer_stations'],
                             'path_prob': p['path_prob'],
                             'path_length_min': p['path_length_min']})
    return pd.DataFrame(rows)


def clear_checkpoint():
    """手動清除 step2 checkpoint"""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f'[Step2] 已清除 {CHECKPOINT_PATH}')
    else:
        print('[Step2] 無 checkpoint 可清除')


if __name__ == '__main__':
    G = load_network_graph()
    mapping = load_station_mapping()
    name_to_ids = build_name_to_ids(G, mapping)
    print(f'路網：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊')
    transfer_nodes = [n for n, d in G.nodes(data=True) if d.get('is_transfer')]
    print(f'轉乘站：{len(transfer_nodes)} 個 → {[G.nodes[n]["name"] for n in transfer_nodes]}')
    print(f'站名對照表：{len(name_to_ids)} 個 OD 站名可解析')
    print(f'  板橋 → {name_to_ids.get("板橋", "NOT FOUND")}')
    print(f'  忠孝復興 → {name_to_ids.get("忠孝復興", "NOT FOUND")}')
