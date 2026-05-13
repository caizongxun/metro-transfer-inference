"""
ml_model.py
XGBoost 轉乘承壓預測模型

訓練目標：
    預測特定 (transfer_station, hour, weekday, month) 的期望承壓流量

使用方式：
    from modules.ml_model import TransferFlowModel
    model = TransferFlowModel()
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    model.save("models/xgb_transfer.json")
    model.load("models/xgb_transfer.json")
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[警告] xgboost 未安裝，請執行：pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from modules.feature_engineering import FEATURE_COLS, build_features
from modules.station_mapper import load_mapping


class TransferFlowModel:
    """
    轉乘站承壓流量預測模型
    支援 XGBoost（預設）或 LightGBM
    """

    def __init__(self, backend: str = "xgboost"):
        self.backend = backend
        self.model   = None
        self.encoders = None
        self.transfer_enc = LabelEncoder()
        self.feature_importance_ = None

    def _default_params(self) -> dict:
        if self.backend == "lightgbm":
            return {
                "objective":    "regression",
                "metric":       "rmse",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves":   127,
                "min_child_samples": 20,
                "subsample":    0.8,
                "colsample_bytree": 0.8,
                "verbose":     -1,
            }
        else:  # xgboost
            return {
                "objective":       "reg:squarederror",
                "eval_metric":     "rmse",
                "n_estimators":    500,
                "learning_rate":   0.05,
                "max_depth":       6,
                "subsample":       0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "early_stopping_rounds": 30,
                "verbosity":       0,
            }

    def build_training_data(
        self,
        od_df: pd.DataFrame,
        label_df: pd.DataFrame
    ) -> tuple:
        """
        合併 OD 特徵 + 轉乘站標記，組成訓練集

        od_df   : 原始 OD DataFrame（含 日期, 時段, 進站, 出站, 人次）
        label_df: build_transfer_flow_labels() 的輸出
                  (日期, hour, transfer_station, expected_flow)

        Returns: (X, y, encoders)
        """
        # 1. OD 特徵工程
        feat_df, self.encoders = build_features(od_df, fit_encoders=True)

        # 2. 聚合到 (日期, hour) 層級（label 是 per transfer_station）
        od_agg = (
            feat_df
            .groupby(["日期", "hour"])
            .agg(
                weekday=("weekday", "first"),
                month=("month", "first"),
                is_holiday=("is_holiday", "first"),
                is_peak=("is_peak", "first"),
                is_weekend=("is_weekend", "first"),
                hour_sin=("hour_sin", "first"),
                hour_cos=("hour_cos", "first"),
                weekday_sin=("weekday_sin", "first"),
                weekday_cos=("weekday_cos", "first"),
                month_sin=("month_sin", "first"),
                month_cos=("month_cos", "first"),
                total_trips=("人次", "sum"),
            )
            .reset_index()
        )

        # 3. 合併 label
        label_df["日期"] = label_df["日期"].astype(str)
        od_agg["日期"]   = od_agg["日期"].astype(str)

        merged = label_df.merge(od_agg, on=["日期", "hour"], how="left")

        # 4. 轉乘站 Label Encode
        self.transfer_enc.fit(merged["transfer_station"])
        merged["transfer_station_id"] = self.transfer_enc.transform(
            merged["transfer_station"]
        )

        # 5. 最終特徵集
        feature_cols = [
            "weekday", "hour", "month",
            "is_holiday", "is_peak", "is_weekend",
            "transfer_station_id",
            "total_trips",
            "hour_sin", "hour_cos",
            "weekday_sin", "weekday_cos",
            "month_sin", "month_cos",
        ]
        self._feature_cols = feature_cols

        X = merged[feature_cols].fillna(0)
        y = merged["expected_flow"].fillna(0)

        return X, y

    def train(
        self,
        od_df: pd.DataFrame,
        label_df: pd.DataFrame,
        test_size: float = 0.2,
        params: dict = None
    ):
        """
        訓練模型
        Returns: 評估指標 dict
        """
        X, y = self.build_training_data(od_df, label_df)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=False  # 時序資料不打亂
        )

        p = params or self._default_params()

        if self.backend == "lightgbm" and LGB_AVAILABLE:
            es = p.pop("early_stopping_rounds", 50)
            self.model = lgb.LGBMRegressor(**p)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(es, verbose=False),
                           lgb.log_evaluation(period=50)]
            )
        elif XGB_AVAILABLE:
            es = p.pop("early_stopping_rounds", 30)
            self.model = xgb.XGBRegressor(**p, early_stopping_rounds=es)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50
            )
        else:
            raise RuntimeError("xgboost 與 lightgbm 均未安裝")

        # 評估
        preds_val = self.model.predict(X_val)
        mae  = mean_absolute_error(y_val, preds_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds_val))

        # 特徵重要性
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=self._feature_cols
            ).sort_values(ascending=False)

        metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2),
                   "train_samples": len(X_train), "val_samples": len(X_val)}
        print(f"  [評估] MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"train={len(X_train):,}  val={len(X_val):,}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """預測轉乘流量"""
        if self.model is None:
            raise RuntimeError("請先 train() 或 load()")
        return self.model.predict(X[self._feature_cols].fillna(0))

    def predict_for_datetime(
        self,
        transfer_station: str,
        hour: int,
        weekday: int,
        month: int,
        is_holiday: int = 0,
        total_trips_estimate: float = 5000.0
    ) -> float:
        """
        快速預測單一情境
        """
        if transfer_station not in self.transfer_enc.classes_:
            return 0.0
        t_id = self.transfer_enc.transform([transfer_station])[0]
        x = pd.DataFrame([{
            "weekday":            weekday,
            "hour":               hour,
            "month":              month,
            "is_holiday":         is_holiday,
            "is_peak":            int(hour in set(list(range(7,10))+list(range(17,20)))),
            "is_weekend":         int(weekday >= 5),
            "transfer_station_id": t_id,
            "total_trips":        total_trips_estimate,
            "hour_sin":           np.sin(2*np.pi*hour/24),
            "hour_cos":           np.cos(2*np.pi*hour/24),
            "weekday_sin":        np.sin(2*np.pi*weekday/7),
            "weekday_cos":        np.cos(2*np.pi*weekday/7),
            "month_sin":          np.sin(2*np.pi*month/12),
            "month_cos":          np.cos(2*np.pi*month/12),
        }])
        return float(self.model.predict(x)[0])

    def save(self, path: str = "models/xgb_transfer.json"):
        """儲存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        meta = {
            "backend":       self.backend,
            "feature_cols":  self._feature_cols,
            "transfer_classes": list(self.transfer_enc.classes_)
        }
        with open(path.replace(".json", "_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"  模型已儲存：{path}")

    def load(self, path: str = "models/xgb_transfer.json"):
        """載入模型"""
        meta_path = path.replace(".json", "_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.backend      = meta["backend"]
        self._feature_cols = meta["feature_cols"]
        self.transfer_enc.classes_ = np.array(meta["transfer_classes"])

        if self.backend == "lightgbm" and LGB_AVAILABLE:
            self.model = lgb.Booster(model_file=path)
        elif XGB_AVAILABLE:
            self.model = xgb.XGBRegressor()
            self.model.load_model(path)
        print(f"  模型已載入：{path}")
