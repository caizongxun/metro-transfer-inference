"""
anoMaly_detector.py
轉乘站異常流量偵測

方法：
    1. 比較 ML 預測值 vs 歷史統計值
    2. z-score > threshold → 異常警示
    3. 輸出異常紀錄 DataFrame

使用情境：
    - 即時監控：每小時跑一次，標記異常轉乘站
    - 歷史回溯：批次跑全部資料，找出歷史異常事件
"""

import pandas as pd
import numpy as np
from typing import Optional


class AnomalyDetector:
    """
    基於殘差 z-score 的異常偵測器
    """

    def __init__(self, z_threshold: float = 2.5, min_history: int = 30):
        """
        z_threshold : z-score 超過此值 → 異常
        min_history : 計算統計量所需的最少樣本數
        """
        self.z_threshold = z_threshold
        self.min_history = min_history
        self._stats: Optional[pd.DataFrame] = None  # 歷史統計表

    def fit(self, label_df: pd.DataFrame):
        """
        從歷史標記資料學習每個 (transfer_station, hour, weekday) 的
        流量均值與標準差

        label_df: 含 (日期, hour, transfer_station, expected_flow) 的 DataFrame
        """
        label_df = label_df.copy()
        label_df["日期"]    = pd.to_datetime(label_df["日期"])
        label_df["weekday"] = label_df["日期"].dt.dayofweek

        stats = (
            label_df
            .groupby(["transfer_station", "hour", "weekday"])
            .agg(
                hist_mean=("expected_flow", "mean"),
                hist_std=("expected_flow", "std"),
                count=("expected_flow", "count"),
            )
            .reset_index()
        )
        # 標準差 0 → 用 1（避免除以零）
        stats["hist_std"] = stats["hist_std"].fillna(1).clip(lower=1)
        self._stats = stats
        print(f"  [AnomalyDetector] fit 完成："
              f"{len(stats)} 個 (站, 時段, 星期) 組合")

    def detect(
        self,
        label_df: pd.DataFrame,
        pred_df:  Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        偵測異常

        label_df : 要偵測的資料，含 (日期, hour, transfer_station, expected_flow)
        pred_df  : ML 預測值 DataFrame，含同樣欄位但 expected_flow = predicted_flow
                   若 None，則直接用歷史統計做 z-score

        Returns: 異常紀錄 DataFrame
        """
        if self._stats is None:
            raise RuntimeError("請先 fit()")

        df = label_df.copy()
        df["日期"]    = pd.to_datetime(df["日期"])
        df["weekday"] = df["日期"].dt.dayofweek

        # 合併歷史統計
        df = df.merge(
            self._stats[["transfer_station", "hour", "weekday",
                          "hist_mean", "hist_std"]],
            on=["transfer_station", "hour", "weekday"],
            how="left"
        )
        df["hist_std"]  = df["hist_std"].fillna(1).clip(lower=1)
        df["hist_mean"] = df["hist_mean"].fillna(0)

        # 若有 ML 預測值，用殘差 (actual - predicted) 計算 z-score
        # 否則直接用 (actual - hist_mean) / hist_std
        if pred_df is not None:
            pred_map = (
                pred_df
                .set_index(["日期", "hour", "transfer_station"])["expected_flow"]
                .rename("predicted_flow")
            )
            df["日期"] = df["日期"].astype(str)
            df = df.join(
                pred_map,
                on=["日期", "hour", "transfer_station"]
            )
            df["predicted_flow"] = df["predicted_flow"].fillna(df["hist_mean"])
            df["residual"] = df["expected_flow"] - df["predicted_flow"]
        else:
            df["residual"] = df["expected_flow"] - df["hist_mean"]

        df["z_score"] = df["residual"] / df["hist_std"]
        df["is_anomaly"] = (df["z_score"].abs() > self.z_threshold).astype(int)
        df["anomaly_dir"] = df.apply(
            lambda r: "surge" if r["z_score"] > self.z_threshold
                      else ("drop" if r["z_score"] < -self.z_threshold else "normal"),
            axis=1
        )

        anomalies = df[df["is_anomaly"] == 1].copy()
        return anomalies.sort_values("z_score", ascending=False).reset_index(drop=True)

    def top_risk_stations(
        self,
        label_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        找出歷史上最常出現異常的轉乘站（風險排名）
        """
        anomalies = self.detect(label_df)
        risk = (
            anomalies
            .groupby("transfer_station")
            .agg(
                anomaly_count=("is_anomaly", "sum"),
                avg_z=("z_score", "mean"),
                max_z=("z_score", "max"),
                surge_count=("anomaly_dir",
                             lambda x: (x == "surge").sum()),
                drop_count=("anomaly_dir",
                            lambda x: (x == "drop").sum()),
            )
            .reset_index()
            .sort_values("anomaly_count", ascending=False)
            .head(top_n)
        )
        return risk


if __name__ == "__main__":
    # 快速測試
    np.random.seed(42)
    n = 1000
    fake_label = pd.DataFrame({
        "日期": pd.date_range("2025-01-01", periods=n, freq="h"),
        "hour": np.random.randint(0, 24, n),
        "transfer_station": np.random.choice(["忠孝復興","台北車站","忠孝新生","古亭"], n),
        "expected_flow": np.random.normal(500, 80, n).clip(0),
    })
    fake_label["日期"] = fake_label["日期"].dt.strftime("%Y-%m-%d")

    detector = AnomalyDetector(z_threshold=2.0)
    detector.fit(fake_label)
    anomalies = detector.detect(fake_label)
    print(f"偵測到異常：{len(anomalies)} 筆")
    print(anomalies[["日期","hour","transfer_station","expected_flow","z_score","anomaly_dir"]].head(10).to_string(index=False))

    print("\n風險站排名：")
    print(detector.top_risk_stations(fake_label).to_string(index=False))
