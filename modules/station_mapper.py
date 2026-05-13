"""
station_mapper.py
中文站名 ↔ 路網 ID 對照轉換模組
"""

import json
import os


def load_mapping(mapping_path: str = "data/station_mapping.json") -> dict:
    """讀取對照表"""
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def name_to_id(name: str, mapping: dict) -> str:
    """
    中文站名 → 路網 ID
    若該站為轉乘站，回傳主線 ID
    找不到則回傳原始名稱（防止崩潰）
    """
    entry = mapping.get(name)
    if entry:
        return entry["id"]
    return name  # fallback


def id_to_name(station_id: str, mapping: dict) -> str:
    """路網 ID → 中文站名"""
    for name, entry in mapping.items():
        if entry["id"] == station_id:
            return name
        if entry.get("also") == station_id:
            return name
    return station_id


def get_transfer_ids(name: str, mapping: dict) -> list:
    """
    取得轉乘站的所有 ID（主 ID + also ID）
    例： 台北車站 → ["R10", "BL11"]
    """
    entry = mapping.get(name)
    if not entry:
        return [name]
    ids = [entry["id"]]
    if "also" in entry:
        ids.append(entry["also"])
    return ids


def map_od_dataframe(df, mapping: dict):
    """
    將 OD DataFrame 的進站/出站中文名稱轉換為 ID
    回傳新增 origin_id, destination_id 欄位
    """
    df = df.copy()
    df["origin_id"]      = df["進站"].apply(lambda x: name_to_id(x, mapping))
    df["destination_id"] = df["出站"].apply(lambda x: name_to_id(x, mapping))
    return df


def unmapped_stations(df, mapping: dict) -> list:
    """列出尚未對照到的站名，方便补充對照表"""
    all_stations = set(df["進站"].unique()) | set(df["出站"].unique())
    missing = [s for s in sorted(all_stations) if s not in mapping]
    return missing


if __name__ == "__main__":
    import glob
    import pandas as pd
    from modules.afc_processor import load_od_csv, filter_by_timeslot

    mapping = load_mapping()
    print(f"對照表載入：{len(mapping)} 筆")

    files = sorted(glob.glob("data/od_raw/*.csv"))
    if files:
        df = load_od_csv(files[0])
        missing = unmapped_stations(df, mapping)
        if missing:
            print(f"\n尚未對照的站名：{len(missing)} 個")
            for s in missing:
                print(f"  - {s}")
        else:
            print("\n所有站名均已對照")

        # 測試轉換
        peak = filter_by_timeslot(df, 7, 9)
        peak_mapped = map_od_dataframe(peak, mapping)
        print("\n映射範例：")
        print(peak_mapped[["進站", "origin_id", "出站", "destination_id", "人次"]].head(10).to_string(index=False))
