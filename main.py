"""
main.py
主執行入口 - 北捷轉乘路徑推估引擎
使用真實 OD 資料，自動讀取 data/od_raw/ 下的所有 CSV

執行方式：
    python main.py
    python main.py --timeslot 7 9          # 早尖峰（預設）
    python main.py --timeslot 17 20        # 晩尖峰
    python main.py --top 20               # 取 Top 20 OD（預設 10）
    python main.py --weekday              # 只分析平日
"""

import argparse
import pandas as pd
from modules.network_builder import load_network, build_graph
from modules.afc_processor import (
    load_od_directory, filter_by_timeslot, filter_weekday, get_top_od_pairs
)
from modules.station_mapper import load_mapping, map_od_dataframe
from modules.path_inference import infer_path_probabilities
from modules.output_formatter import (
    aggregate_transfer_flows,
    generate_pressure_report,
    generate_diversion_suggestion,
    format_od_report
)


def main():
    parser = argparse.ArgumentParser(description="北捷轉乘路徑推估引擎")
    parser.add_argument("--timeslot", nargs=2, type=int, default=[7, 9],
                        metavar=("START", "END"), help="時段範圍（預設 7 9）")
    parser.add_argument("--top", type=int, default=10, help="取 Top N OD 對（預設 10）")
    parser.add_argument("--weekday", action="store_true", help="只分析平日")
    parser.add_argument("--od_dir", default="data/od_raw", help="OD CSV 目錄")
    parser.add_argument("--threshold", type=float, default=500.0,
                        help="轉乘承壓警示閾値（人次，預設 500）")
    args = parser.parse_args()

    print("=" * 60)
    print("北捷轉乘路徑推估引擎")
    print("Metro Transfer Inference Engine")
    print("=" * 60)

    # ── 1. 載入路網與對照表 ───────────────────────────────────
    print("\n[1/4] 載入路網與對照表...")
    network = load_network()
    G = build_graph(network)
    mapping = load_mapping()
    print(f"      路網：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊")
    print(f"      對照表：{len(mapping)} 站")

    # ── 2. 讀入真實 OD 資料 ──────────────────────────────────
    print(f"\n[2/4] 讀入 OD 資料（{args.od_dir}）...")
    df = load_od_directory(args.od_dir)

    if args.weekday:
        df = filter_weekday(df)
        print(f"      平日筆數：{len(df):,}")

    # 過濾時段
    start_h, end_h = args.timeslot
    peak_df = filter_by_timeslot(df, start_h, end_h)
    print(f"      時段 {start_h:02d}:00~{end_h:02d}:00 筆數：{len(peak_df):,}")

    # 取 Top N OD
    top_od = get_top_od_pairs(peak_df, top_n=args.top)
    print(f"\n      Top {args.top} OD 對（人次最多）：")
    print(top_od[["origin", "destination", "total_trips"]].to_string(index=False))

    # ── 3. 路徑推論 ──────────────────────────────────────────
    print(f"\n[3/4] 路徑推論中（{args.top} 對 OD）...")
    inference_results = []

    for _, row in top_od.iterrows():
        origin_name = row["origin"]
        dest_name   = row["destination"]
        trip_count  = int(row["total_trips"])

        # 中文站名 → 路網 ID
        from modules.station_mapper import name_to_id
        origin_id = name_to_id(origin_name, mapping)
        dest_id   = name_to_id(dest_name, mapping)

        # 用各時段平均人次作為代表性旅次數（不需要實際旅行時間）
        # 以理論最短時間作為推論基準
        paths = infer_path_probabilities(
            origin=origin_id,
            destination=dest_id,
            actual_time=None,   # 使用理論時間作為基準
            G=G,
            crowding_df=None,
            k=5
        )

        if paths:
            inference_results.append({
                "origin": origin_name,
                "origin_id": origin_id,
                "destination": dest_name,
                "destination_id": dest_id,
                "trip_count": trip_count,
                "paths": paths
            })
            top1 = paths[0]
            transfers = [f"{a}→{b}" for a, b in top1["transfer_stations"]]
            print(f"  {origin_name}→{dest_name} ({trip_count:,}人) │ "
                  f"路徑: {' -> '.join(top1['path'])} │ "
                  f"轉乘: {transfers if transfers else '直達'} │ "
                  f"機率: {top1['prob']*100:.1f}%")
        else:
            print(f"  {origin_name}→{dest_name}: 找不到路徑（路網尚未包含此站）")

    # ── 4. 聚合輸出 ────────────────────────────────────────────
    print("\n[4/4] 聚合輸出...")

    if not inference_results:
        print("沒有成功推論的旅次，請檢查路網與對照表")
        return

    # OD 路徑報告
    od_report = format_od_report(inference_results)
    od_report.to_csv("output_od_report.csv", index=False, encoding="utf-8-sig")
    print("\n📋 OD 路徑比例報告：")
    print(od_report.to_string(index=False))

    # 轉乘站承壓
    flow_df = aggregate_transfer_flows(inference_results)
    threshold = args.threshold
    pressure_df = generate_pressure_report(flow_df, threshold=threshold)
    pressure_df.to_csv("output_transfer_pressure.csv", index=False, encoding="utf-8-sig")
    print("\n🔥 轉乘站承壓報告：")
    print(pressure_df.to_string(index=False))

    # 分流建議
    suggestions = generate_diversion_suggestion(flow_df, network, threshold=threshold)
    if suggestions:
        print("\n🚦 分流建議：")
        for s in suggestions:
            print(f"  超載：{s['overloaded_pair']}")
            print(f"  建議替代：{s['alternative']}")
            print(f"  經由：{s['via_stations']}，額外時間：{s['extra_time']}")
    else:
        print(f"\n✅ 無超載警示（閾値 {threshold:.0f} 人次）")

    print("\n" + "=" * 60)
    print("完成！輸出檔案：output_od_report.csv, output_transfer_pressure.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
