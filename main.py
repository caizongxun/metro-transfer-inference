"""
main.py
主執行入口 - 北捷轉乘路徑推估引擎

執行步驟：
1. 載入路網、AFC、時刻表、擁擠度資料
2. 對每筆 AFC 旅次推論路徑機率
3. 聚合轉乘站流量
4. 輸出報告（CSV + 終端機顯示）
"""

import pandas as pd
from modules.network_builder import load_network, build_graph
from modules.afc_processor import load_afc, filter_by_timeslot
from modules.path_inference import infer_path_probabilities
from modules.output_formatter import (
    aggregate_transfer_flows,
    generate_pressure_report,
    generate_diversion_suggestion,
    format_od_report
)


def main():
    print("=" * 60)
    print("北捷轉乘路徑推估引擎")
    print("Metro Transfer Inference Engine")
    print("=" * 60)

    # ── 1. 載入資料 ──────────────────────────────────────────────
    print("\n[1/4] 載入資料...")
    network = load_network()
    G = build_graph(network)
    print(f"      路網：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊")

    afc_df = load_afc()
    crowding_df = pd.read_csv("data/sample_crowding.csv")
    print(f"      AFC 旅次：{len(afc_df)} 筆")

    # ── 2. 過濾早尖峰時段 ──────────────────────────────────────────
    print("\n[2/4] 過濾早尖峰時段 07:00-09:00...")
    peak_afc = filter_by_timeslot(afc_df, "07:00", "09:00")
    print(f"      早尖峰旅次：{len(peak_afc)} 筆")

    # ── 3. 對每筆旅次推論路徑 ────────────────────────────────────
    print("\n[3/4] 路徑推論中...")
    inference_results = []

    for _, row in peak_afc.iterrows():
        paths = infer_path_probabilities(
            origin=row["entry_station_id"],
            destination=row["exit_station_id"],
            actual_time=row["travel_time_min"],
            G=G,
            crowding_df=crowding_df,
            timestamp=str(row["entry_time"]),
            k=5
        )
        if paths:
            inference_results.append({
                "origin": row["entry_station_id"],
                "destination": row["exit_station_id"],
                "actual_time": row["travel_time_min"],
                "trip_count": 1,
                "paths": paths
            })

    print(f"      成功推論：{len(inference_results)} 筆旅次")

    # ── 4. 聚合輸出 ────────────────────────────────────────────
    print("\n[4/4] 聚合輸出...")

    # OD 路徑報告
    od_report = format_od_report(inference_results)
    od_report.to_csv("output_od_report.csv", index=False, encoding="utf-8-sig")
    print("\n📋 OD 路徑比例報告：")
    print(od_report.to_string(index=False))

    # 轉乘站流量
    flow_df = aggregate_transfer_flows(inference_results)
    pressure_df = generate_pressure_report(flow_df, threshold=3.0)
    pressure_df.to_csv("output_transfer_pressure.csv", index=False, encoding="utf-8-sig")
    print("\n🔥 轉乘站承壓報告：")
    print(pressure_df.to_string(index=False))

    # 分流建議
    suggestions = generate_diversion_suggestion(flow_df, network, threshold=3.0)
    if suggestions:
        print("\n🚦 分流建議：")
        for s in suggestions:
            print(f"  超載：{s['overloaded_pair']}")
            print(f"  建議替代：{s['alternative']}")
            print(f"  經由：{s['via_stations']}，額外時間：{s['extra_time']}")
    else:
        print("\n✅ 無超載警示")

    print("\n" + "=" * 60)
    print("完成！輸出檔案：output_od_report.csv, output_transfer_pressure.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
