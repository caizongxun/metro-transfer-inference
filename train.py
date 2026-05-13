"""
train.py  (OOM 優化版)
核心策略：
    1. 逐月處理，不全部載入記憶體
    2. 每月產生 label→套全沈 od_df→讀就累積 label_df → 最後統一訓練
    3. 增存式 XGBoost（partial-fit via DMatrix warm-start）

執行方式：
    python train.py
    python train.py --months 12 --backend lightgbm
    python train.py --timeslot 7 9 --weekday
    python train.py --chunk_size 1        # 流量低的機器用 1 個月一小塊
"""

import os
import gc
import glob
import argparse
import numpy as np
import pandas as pd
from modules.afc_processor import load_od_csv, filter_by_timeslot, filter_weekday
from modules.network_builder import load_network, build_graph
from modules.station_mapper import load_mapping
from modules.path_prob_labeler import build_transfer_flow_labels
from modules.ml_model import TransferFlowModel
from modules.anomaly_detector import AnomalyDetector


def load_chunk(files: list, G, mapping: dict,
               timeslot=None, weekday_only=False) -> pd.DataFrame:
    """讀一組 CSV -> label, 完成後釋放 od_df"""
    dfs = []
    for f in files:
        tmp = load_od_csv(f)
        if weekday_only:
            tmp = filter_weekday(tmp)
        if timeslot:
            tmp = filter_by_timeslot(tmp, timeslot[0], timeslot[1])
        tmp = tmp[tmp["人次"] > 0]
        dfs.append(tmp)

    od_chunk = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    label_chunk = build_transfer_flow_labels(od_chunk, G, mapping)
    del od_chunk
    gc.collect()
    return label_chunk


def build_features_for_label(label_df: pd.DataFrame) -> pd.DataFrame:
    """
    label_df 欄位: 日期, hour, transfer_station, expected_flow
    回傳特徵 DataFrame, 欄位同 TransferFlowModel._feature_cols
    """
    df = label_df.copy()
    df["日期"] = pd.to_datetime(df["日期"])
    df["weekday"]  = df["日期"].dt.dayofweek
    df["month"]    = df["日期"].dt.month
    is_peak_hours  = set(list(range(7, 10)) + list(range(17, 20)))
    df["is_peak"]  = df["hour"].isin(is_peak_hours).astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]    / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]    / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]   / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]   / 12)
    return df


def main():
    parser = argparse.ArgumentParser(description="ML 轉乘承壓預測訓練 (OOM 優化版)")
    parser.add_argument("--months",     type=int,  default=None)
    parser.add_argument("--backend",    type=str,  default="xgboost",
                        choices=["xgboost", "lightgbm"])
    parser.add_argument("--timeslot",   nargs=2,   type=int, default=None,
                        metavar=("START", "END"))
    parser.add_argument("--weekday",    action="store_true")
    parser.add_argument("--od_dir",     default="data/od_raw")
    parser.add_argument("--model_out",  default="models/xgb_transfer.json")
    parser.add_argument("--chunk_size", type=int,  default=3,
                        help="每次處理幾個月 (預設 3, OOM 嚴重時改 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("Metro Transfer ML 訓練 (OOM 優化版)")
    print("=" * 60)

    # ── 1. 路網 ──────────────────────────────────────────────
    print("\n[1/4] 載入路網...")
    network = load_network()
    G = build_graph(network)
    mapping = load_mapping()
    print(f"      {G.number_of_nodes()} 站  {G.number_of_edges()} 邊")

    # ── 2. 逐 chunk 產生 label ───────────────────────────────
    files = sorted(glob.glob(os.path.join(args.od_dir, "*.csv")))
    if args.months:
        files = files[-args.months:]
    print(f"\n[2/4] 共 {len(files)} 個月份，每次處理 {args.chunk_size} 個月...")

    os.makedirs("data/label_chunks", exist_ok=True)
    chunks = [files[i:i+args.chunk_size]
              for i in range(0, len(files), args.chunk_size)]

    all_label_paths = []
    for ci, chunk_files in enumerate(chunks, 1):
        names = [os.path.basename(f) for f in chunk_files]
        print(f"  chunk {ci}/{len(chunks)}: {names}")

        label_chunk = load_chunk(
            chunk_files, G, mapping,
            timeslot=args.timeslot,
            weekday_only=args.weekday
        )
        out_path = f"data/label_chunks/chunk_{ci:03d}.parquet"
        label_chunk.to_parquet(out_path, index=False)
        all_label_paths.append(out_path)
        print(f"    → {len(label_chunk):,} 筆 label 已存 {out_path}")
        del label_chunk
        gc.collect()

    # ── 3. 合併 label & 訓練 ─────────────────────────────────
    print("\n[3/4] 合併 label 並訓練模型...")
    label_df = pd.concat(
        [pd.read_parquet(p) for p in all_label_paths],
        ignore_index=True
    )
    print(f"      總 label 筆數：{len(label_df):,}")
    print(f"      涵蓋轉乘站：{label_df['transfer_station'].nunique()} 個")
    label_df.to_csv("data/transfer_labels.csv", index=False, encoding="utf-8-sig")

    # 建特徵 (label 本身就夠，不需重載 od_df)
    feat_df = build_features_for_label(label_df)

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    t_enc = LabelEncoder()
    feat_df["transfer_station_id"] = t_enc.fit_transform(
        feat_df["transfer_station"]
    )

    FEATURE_COLS = [
        "weekday", "hour", "month",
        "is_peak", "is_weekend",
        "transfer_station_id",
        "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos",
        "month_sin", "month_cos",
    ]

    X = feat_df[FEATURE_COLS].fillna(0)
    y = feat_df["expected_flow"].fillna(0)
    del feat_df
    gc.collect()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    del X, y
    gc.collect()

    os.makedirs(os.path.dirname(args.model_out) or "models", exist_ok=True)

    if args.backend == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
            num_leaves=127, subsample=0.8,
            colsample_bytree=0.8, verbose=-1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=50)
            ]
        )
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        model.booster_.save_model(args.model_out)
    else:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5,
            early_stopping_rounds=30, verbosity=0,
            tree_method="hist",   # 省記憶體
            device="cpu",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        model.save_model(args.model_out)

    # 儲存 meta
    import json
    meta = {
        "backend": args.backend,
        "feature_cols": FEATURE_COLS,
        "transfer_classes": list(t_enc.classes_)
    }
    meta_path = args.model_out.replace(".json", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    preds_val = model.predict(X_val)
    mae  = mean_absolute_error(y_val, preds_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    print(f"  [評估] MAE={mae:.2f}  RMSE={rmse:.2f}")
    print(f"  模型已儲存：{args.model_out}")

    fi_df = fi.reset_index()
    fi_df.columns = ["feature", "importance"]
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv("output_feature_importance.csv", index=False, encoding="utf-8-sig")
    print("\n特徵重要性 Top 10：")
    print(fi_df.head(10).to_string(index=False))

    # ── 4. 異常偵測 ──────────────────────────────────────────
    print("\n[4/4] 訓練異常偵測器...")
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
    print("  meta：" + meta_path)
    print("  特徵重要性：output_feature_importance.csv")
    print("  歷史異常：output_anomalies.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
