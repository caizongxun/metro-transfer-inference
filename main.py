"""
main.py
主執行入口 - 北捐轉乘路徑推估引擎
使用真實 OD 資料，自動讀取 data/od_raw/ 下的所有 CSV

執行方式：
    python main.py
    python main.py --timeslot 7 9          # 早尖峰（預設）
    python main.py --timeslot 17 20        # 晩尖峰
    python main.py --top 20               # 取 Top 20 OD（預設 10）
    python main.py --weekday              # 只分析平日
    python main.py --ml                   # 啟用 ML 預測模組輔助承壓評估
    python main.py --ml --model_path models/xgb_transfer.json
"""

import os
import argparse
import pandas as pd
from modules.network_builder import load_network, build_graph
from modules.afc_processor import (
    load_od_directory, filter_by_timeslot, filter_weekday, get_top_od_pairs
)
from modules.station_mapper import load_mapping, name_to_id
from modules.path_inference import infer_path_probabilities
from modules.output_formatter import (
    aggregate_transfer_flows,
    generate_pressure_report,
    generate_diversion_suggestion,
    format_od_report
)


def load_ml_model(model_path: str):
    """
    載入 ML 模型。模型檔不存在時回傳 None 而不是丟出錯誤。
    """
    if not os.path.exists(model_path):
        print(f"  [ML] 模型檔不存在：{model_path}")
        print(f"  [ML] 請先執行： python train.py")
        return None
    try:
        from modules.ml_model import TransferFlowModel
        model = TransferFlowModel()
        model.load(model_path)
        print(f"  [ML] 模型載入成功：{model_path}")
        return model
    except Exception as e:
        print(f"  [ML] 模型載入失敗：{e}")
        return None


def augment_pressure_with_ml(
    pressure_df: pd.DataFrame,
    model,
    hour: int,
    weekday: int,
    month: int,
    is_holiday: int = 0,
) -> pd.DataFrame:
    """
    對找到的找乘站承壓表，加入 ML 預測的期望流量和差異まイ。
    回傳新增欄位： ml_expected_flow, ml_diff, ml_overload
    """
    if model is None or 'transfer_station' not in pressure_df.columns:
        return pressure_df

    rows = []
    for _, row in pressure_df.iterrows():
        station = row.get('transfer_station', row.get('station', ''))
        # 估算經過此站的總旅次數（用於 ML 輸入）
        total_trips_est = float(row.get('estimated_flow', row.get('flow', 5000)))
        ml_flow = model.predict_for_datetime(
            transfer_station=station,
            hour=hour,
            weekday=weekday,
            month=month,
            is_holiday=is_holiday,
            total_trips_estimate=total_trips_est,
        )
        rows.append(ml_flow)

    pressure_df = pressure_df.copy()
    pressure_df['ml_expected_flow'] = rows
    # 實際 vs ML 預測的差異ま（正數=超載，負數=尚有餘裕）
    actual_col = 'estimated_flow' if 'estimated_flow' in pressure_df.columns else 'flow'
    if actual_col in pressure_df.columns:
        pressure_df['ml_diff'] = (
            pressure_df[actual_col] - pressure_df['ml_expected_flow']
        ).round(1)
        pressure_df['ml_overload'] = pressure_df['ml_diff'] > 0
    return pressure_df


def main():
    parser = argparse.ArgumentParser(description="北捐轉乘路徑推估引擎")
    parser.add_argument("--timeslot", nargs=2, type=int, default=[7, 9],
                        metavar=("START", "END"), help="時段範圍（預設 7 9）")
    parser.add_argument("--top", type=int, default=10, help="取 Top N OD 對（預設 10）")
    parser.add_argument("--weekday", action="store_true", help="只分析平日")
    parser.add_argument("--od_dir", default="data/od_raw", help="OD CSV 目錄")
    parser.add_argument("--threshold", type=float, default=500.0,
                        help="轉乘承壓警示閾e値（人次，預設 500）")
    parser.add_argument("--ml", action="store_true",
                        help="啟用 ML 模型輔助承壓評估")
    parser.add_argument("--model_path", default="models/xgb_transfer.json",
                        help="ML 模型檔路徑（預設 models/xgb_transfer.json）")
    parser.add_argument("--is_holiday", type=int, default=0,
                        help="是否假日（0=平日 1=假日）")
    args = parser.parse_args()

    print("=" * 60)
    print("北捐轉乘路徑推估引擎")
    print("Metro Transfer Inference Engine")
    print("=" * 60)

    # ── 0. 載入 ML 模型（可選） ─────────────────────────────
    ml_model = None
    if args.ml:
        print("\n[0/5] 載入 ML 模型...")
        ml_model = load_ml_model(args.model_path)

    # ── 1. 載入路網與對照表 ────────────────────────────
    print("\n[1/5] 載入路網與對照表...")
    network = load_network()
    G = build_graph(network)
    mapping = load_mapping()
    print(f"      路網：{G.number_of_nodes()} 站，{G.number_of_edges()} 條邊")
    print(f"      對照表：{len(mapping)} 站")

    # ── 2. 讀入真實 OD 資料 ────────────────────────────
    print(f"\n[2/5] 讀入 OD 資料（{args.od_dir}）...")
    df = load_od_directory(args.od_dir)

    if args.weekday:
        df = filter_weekday(df)
        print(f"      平日筆數：{len(df):,}")

    start_h, end_h = args.timeslot
    peak_df = filter_by_timeslot(df, start_h, end_h)
    print(f"      時段 {start_h:02d}:00~{end_h:02d}:00 筆數：{len(peak_df):,}")

    top_od = get_top_od_pairs(peak_df, top_n=args.top)
    print(f"\n      Top {args.top} OD 對（人次最多）：")
    print(top_od[["origin", "destination", "total_trips"]].to_string(index=False))

    # ── 3. 路徑推論 ───────────────────────────────────
    print(f"\n[3/5] 路徑推論中（{args.top} 對 OD）...")
    inference_results = []

    for _, row in top_od.iterrows():
        origin_name = row["origin"]
        dest_name   = row["destination"]
        trip_count  = int(row["total_trips"])

        origin_id = name_to_id(origin_name, mapping)
        dest_id   = name_to_id(dest_name, mapping)

        paths = infer_path_probabilities(
            origin=origin_id,
            destination=dest_id,
            actual_time=None,
            G=G,
            crowding_df=None,
            k=5
        )

        if paths:
            inference_results.append({
                "origin":         origin_name,
                "origin_id":      origin_id,
                "destination":    dest_name,
                "destination_id": dest_id,
                "trip_count":     trip_count,
                "paths":          paths
            })
            top1 = paths[0]
            transfers = [f"{a}→{b}" for a, b in top1["transfer_stations"]]
            print(f"  {origin_name}→{dest_name} ({trip_count:,}人) │ "
                  f"路徑: {' -> '.join(top1['path'])} │ "
                  f"轉乘: {transfers if transfers else '直達'} │ "
                  f"機率: {top1['prob']*100:.1f}%")
        else:
            print(f"  {origin_name}→{dest_name}: 找不到路徑（路網尚未包含此站）")

    # ── 4. 聚合輸出 ───────────────────────────────────
    print("\n[4/5] 聚合輸出...")

    if not inference_results:
        print("沒有成功推論的旅次，請檢查路網與對照表")
        return

    od_report = format_od_report(inference_results)
    od_report.to_csv("output_od_report.csv", index=False, encoding="utf-8-sig")
    print("\n└─ OD 路徑比例報告：")
    print(od_report.to_string(index=False))

    flow_df = aggregate_transfer_flows(inference_results)
    threshold = args.threshold
    pressure_df = generate_pressure_report(flow_df, threshold=threshold)

    # ── 5. ML 輔助承壓評估 ───────────────────────────
    print("\n[5/5] 找乘站承壓報告...")
    if ml_model is not None:
        import datetime
        now = datetime.datetime.now()
        pressure_df = augment_pressure_with_ml(
            pressure_df,
            model=ml_model,
            hour=start_h,           # 以分析小時的起始點代入
            weekday=now.weekday(),
            month=now.month,
            is_holiday=args.is_holiday,
        )
        print("  [ML] 已將 ML 期望流量與差異ま加入承壓表")

    pressure_df.to_csv("output_transfer_pressure.csv", index=False, encoding="utf-8-sig")
    print("\n🔥 轉乘站承壓報告：")
    print(pressure_df.to_string(index=False))

    suggestions = generate_diversion_suggestion(flow_df, network, threshold=threshold)
    if suggestions:
        print("\n🚦 分流建議：")
        for s in suggestions:
            print(f"  超載：{s['overloaded_pair']}")
            print(f"  建議替代：{s['alternative']}")
            print(f"  經由：{s['via_stations']}，額外時間：{s['extra_time']}")
    else:
        print(f"\n✅ 無超載警示（閾e値 {threshold:.0f} 人次）")

    if ml_model is not None and 'ml_overload' in pressure_df.columns:
        ml_alert = pressure_df[pressure_df['ml_overload'] == True]
        if not ml_alert.empty:
            print(f"\n🤖 [ML預警] 以下轉乘站實際流量超過 ML 期望：")
            for _, r in ml_alert.iterrows():
                st = r.get('transfer_station', r.get('station', '?'))
                diff = r.get('ml_diff', 0)
                print(f"  {st} 超出 {diff:.0f} 人")
        else:
            print("\n🤖 [ML預警] 所有轉乘站流量均在 ML 期望範圍內")

    print("\n" + "=" * 60)
    print("完成！輸出檔案：output_od_report.csv, output_transfer_pressure.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
