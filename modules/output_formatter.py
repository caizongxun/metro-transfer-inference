"""
output_formatter.py
聚合路徑推論結果，輸出三種格式：
1. OD 路徑機率分布
2. 轉乘站承壓排名
3. 分流建議
"""

import pandas as pd
from collections import defaultdict


def aggregate_transfer_flows(inference_results: list) -> pd.DataFrame:
    """
    聚合所有 OD 推論結果，計算每個轉乘站的流量

    inference_results 格式：
    [
      { "origin": "BL05", "destination": "R09", "paths": [infer_path_probabilities output] },
      ...
    ]
    """
    flow = defaultdict(float)

    for record in inference_results:
        paths = record.get("paths", [])
        trip_count = record.get("trip_count", 1)

        for p in paths:
            prob = p["prob"]
            for (from_st, to_st) in p["transfer_stations"]:
                key = f"{from_st}->{to_st}"
                flow[key] += prob * trip_count

    df = pd.DataFrame(
        [(k, v) for k, v in flow.items()],
        columns=["transfer_pair", "estimated_flow"]
    ).sort_values("estimated_flow", ascending=False).reset_index(drop=True)

    return df


def generate_pressure_report(flow_df: pd.DataFrame, threshold: float = 100.0) -> pd.DataFrame:
    """
    轉乘站承壓報告：超過 threshold 的轉乘對發出警示
    """
    df = flow_df.copy()
    df["alert"] = df["estimated_flow"].apply(
        lambda x: "🔴 高壓" if x >= threshold else ("🟡 中壓" if x >= threshold * 0.5 else "🟢 正常")
    )
    return df


def generate_diversion_suggestion(flow_df: pd.DataFrame,
                                   network: dict,
                                   threshold: float = 100.0) -> list:
    """
    分流建議：若主要轉乘站超載，回傳替代轉乘建議
    """
    high_pressure = flow_df[flow_df["estimated_flow"] >= threshold]["transfer_pair"].tolist()
    suggestions = []

    # 簡化：台北車站藍↔紅 超載 → 建議西門藍↔綠 + 中山綠↔紅
    DIVERSION_MAP = {
        "BL11->R10": [
            { "alt_path": "BL10→G09→G14→R09", "stations": "西門站 + 中山站", "time_cost": "+4 min" },
        ],
        "R10->BL11": [
            { "alt_path": "R09→G14→G09→BL10", "stations": "中山站 + 西門站", "time_cost": "+4 min" },
        ]
    }

    for pair in high_pressure:
        if pair in DIVERSION_MAP:
            for alt in DIVERSION_MAP[pair]:
                suggestions.append({
                    "overloaded_pair": pair,
                    "alternative": alt["alt_path"],
                    "via_stations": alt["stations"],
                    "extra_time": alt["time_cost"]
                })

    return suggestions


def format_od_report(inference_results: list) -> pd.DataFrame:
    """
    OD 路徑比例報告：每個 OD 的主要路徑機率
    """
    rows = []
    for record in inference_results:
        origin = record["origin"]
        destination = record["destination"]
        paths = record.get("paths", [])
        for p in paths[:2]:  # 取前兩條候選路徑
            transfers = [f"{a}→{b}" for a, b in p["transfer_stations"]]
            rows.append({
                "OD": f"{origin}→{destination}",
                "路徑": " -> ".join(p["path"]),
                "轉乘站": ", ".join(transfers) if transfers else "直達",
                "理論時間(min)": p["theory_time"],
                "機率": f"{p['prob']*100:.1f}%"
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mock_results = [
        {
            "origin": "BL05", "destination": "R09", "trip_count": 150,
            "paths": [
                { "path": ["BL05","BL10","BL11","R10","R09"], "theory_time": 22.5,
                  "prob": 0.68, "transfer_stations": [("BL11","R10")] },
                { "path": ["BL05","BL10","G09","G14","R09"], "theory_time": 26.5,
                  "prob": 0.24, "transfer_stations": [("BL10","G09"),("G14","R09")] },
            ]
        }
    ]

    flow_df = aggregate_transfer_flows(mock_results)
    print("轉乘站流量：")
    print(flow_df)

    report_df = generate_pressure_report(flow_df, threshold=50)
    print("\n承壓報告：")
    print(report_df)

    od_df = format_od_report(mock_results)
    print("\nOD 路徑報告：")
    print(od_df)
