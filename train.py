"""
train.py
完整 ML 訓練流程

執行方式：
    python train.py                         # 訓練全部資料
    python train.py --months 6              # 只用最近 6 個月
    python train.py --backend lightgbm      # 使用 LightGBM
    python train.py --timeslot 7 9          # 只訓練早尖峰

輸出：
    models/xgb_transfer.json        模型檔
    models/xgb_transfer_meta.json   特徵配置
    output_feature_importance.csv   特徵重要性
"""

import os
import glob
import argparse
import pandas as pd
from modules.afc_processor import load_od_csv, filter_by_timeslot, filter_weekday
from modules.network_builder import load_network, build_graph
from modules.station_mapper import load_mapping
from modules.path_prob_labeler import build_transfer_flow_labels
from modules.ml_model import TransferFlowModel
from modules.anomaly_detector import AnomalyDetector


def main():
    parser = argparse.ArgumentParser(description="ML 轉乘承壓預測訓練")
    parser.add_argument("--months",    type=int,   default=None,   help="使用最近 N 個月")
    parser.add_argument("--backend",   type=str,   default="xgboost",
                        choices=["xgboost", "lightgbm"])
    parser.add_argument("--timeslot",  nargs=2,    type=int, default=None,
                        metavar=("START", "END"), help="過濾時段，例如 7 9")
    parser.add_argument("--weekday",   action="store_true", help="只用平日")
    parser.add_argument("--od_dir",    default="data/od_raw")
    parser.add_argument("--model_out", default="models/xgb_transfer.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Metro Transfer ML 訓練")
    print("=" * 60)

    # ── 1. 載入路網 ───────────────────────────────────────────
    print("\n[1/5] 載入路網...")
    network = load_network()
    G = build_graph(network)
    mapping = load_mapping()
    print(f"      {G.number_of_nodes()} 站  {G.number_of_edges()} 邊")

    # ── 2. 讀入 OD 資料 ──────────────────────────────────────
    print(f"\n[2/5] 讀入 OD 資料（{args.od_dir}）...")
    files = sorted(glob.glob(os.path.join(args.od_dir, "*.csv")))
    if args.months:
        files = files[-args.months:]
    print(f"      使用 {len(files)} 個月份的資料")

    dfs = []
    for f in files:
        tmp = load_od_csv(f)
        if args.weekday:
            tmp = filter_weekday(tmp)
        if args.timeslot:
            tmp = filter_by_timeslot(tmp, args.timeslot[0], args.timeslot[1])
        dfs.append(tmp)
    od_df = pd.concat(dfs, ignore_index=True)
    od_df = od_df[od_df["人次"] > 0]
    print(f"      合併後：{len(od_df):,} 筆")

    # ── 3. 產生轉乘標記 ──────────────────────────────────────
    print("\n[3/5] 計算路徑機率 → 產生轉乘流量標記...")
    label_df = build_transfer_flow_labels(od_df, G, mapping)
    print(f"      標記筆數：{len(label_df):,}")
    print(f"      涵蓋轉乘站：{label_df['transfer_station'].nunique()} 個")
    label_df.to_csv("data/transfer_labels.csv", index=False, encoding="utf-8-sig")
    print("      已儲存：data/transfer_labels.csv")

    # ── 4. 訓練 XGBoost ──────────────────────────────────────
    print(f"\n[4/5] 訓練模型（{args.backend}）...")
    model = TransferFlowModel(backend=args.backend)
    metrics = model.train(od_df, label_df)
    model.save(args.model_out)

    if model.feature_importance_ is not None:
        fi = model.feature_importance_.reset_index()
        fi.columns = ["feature", "importance"]
        fi.to_csv("output_feature_importance.csv", index=False, encoding="utf-8-sig")
        print("\n特徵重要性 Top 10：")
        print(fi.head(10).to_string(index=False))

    # ── 5. 異常偵測 fit ──────────────────────────────────────
    print("\n[5/5] 訓練異常偵測器...")
    detector = AnomalyDetector(z_threshold=2.5)
    detector.fit(label_df)

    anomalies = detector.detect(label_df)
    anomalies.to_csv("output_anomalies.csv", index=False, encoding="utf-8-sig")
    print(f"      歷史異常事件：{len(anomalies)} 筆")
    print("\n風險轉乘站 Top 10：")
    print(detector.top_risk_stations(label_df, top_n=10).to_string(index=False))

    print("\n" + "=" * 60)
    print("訓練完成！")
    print(f"  模型：{args.model_out}")
    print("  特徵重要性：output_feature_importance.csv")
    print("  歷史異常：output_anomalies.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
