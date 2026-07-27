"""
step5_export.py
輸出 JSON 供前端 dashboard 使用。

輸出格式（pipeline/dashboard/data.json）：
{
  "generated_at": "2026-07-27T18:00:00",
  "data_source": "public",  // 或 "private"
  "transfer_flow": [
    {"station": "台北車站", "hour": 8, "estimated_trips": 500,
     "transfer_ratio": 0.20, "pressure_level": "高"}, ...
  ],
  "heatmap_matrix": {
    "stations": [...],
    "hours": [6,7,...,23],
    "values": [[ratio_h6, ratio_h7, ...], ...]  // 每站一列
  },
  "headway_plan": [
    {"line": "板南線", "hour": 8, "suggested_headway_min": 3.2}, ...
  ],
  "summary": {
    "peak_hour": 8,
    "peak_station": "台北車站",
    "total_transfer_stations": 12
  }
}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
import pandas as pd
from pipeline.config import OUTPUT_JSON_PATH, DATA_SOURCE


def export_to_json(
    flow_df: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    headway_df: pd.DataFrame
) -> dict:
    """
    組合所有結果並輸出 JSON。
    """
    # transfer_flow
    flow_records = flow_df.to_dict(orient='records')
    for r in flow_records:
        r['pressure_level'] = str(r['pressure_level'])  # Categorical → str
        r['estimated_trips'] = round(float(r['estimated_trips']), 1)

    # heatmap matrix
    stations = list(heatmap_df.index)
    hours = [int(c) for c in heatmap_df.columns]
    values = heatmap_df.values.tolist()

    # headway plan
    headway_records = headway_df.to_dict(orient='records')

    # summary
    peak_row = flow_df.loc[flow_df['estimated_trips'].idxmax()] if len(flow_df) > 0 else None
    summary = {
        'peak_hour': int(peak_row['hour']) if peak_row is not None else None,
        'peak_station': str(peak_row['transfer_station']) if peak_row is not None else None,
        'total_transfer_stations': int(flow_df['transfer_station'].nunique()),
        'high_pressure_count': int((flow_df['pressure_level'] == '高').sum() +
                                    (flow_df['pressure_level'] == '極高').sum())
    }

    output = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_source': DATA_SOURCE,
        'transfer_flow': flow_records,
        'heatmap_matrix': {'stations': stations, 'hours': hours, 'values': values},
        'headway_plan': headway_records,
        'summary': summary
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Export] 已輸出至 {OUTPUT_JSON_PATH}")
    return output
