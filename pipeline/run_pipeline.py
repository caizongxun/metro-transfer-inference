"""
run_pipeline.py
全流程執行入口。依序執行 step1 ~ step5，并在 step3 後插入 step3.5 / step3.6 校正，
以及在 step4 後插入 step4.5 跨線換乘同步。

Step 結構：
  Step 1   : 載入 OD 資料
  Step 2   : 路網拓撲分析
  Step 3   : 轉乘站流量估算
  Step 3.5 : 進出人次校正（ridership 校正）
  Step 3.6 : 列車到離站時間整合（actual_avg_wait_min）       ← 新增
  Step 4   : GA 班距最佳化
  Step 4.5 : 跨線換乘等待時間同步最佳化（offset_min）       ← 新增
  Step 5   : 匯出 JSON（熱力圖 + 趨勢 + 班距建議）
  Step 5T  : 完整時刻表生成（timetable_full.csv + JSON）    ← 新增

執行方式：
  cd <repo root>
  python pipeline/run_pipeline.py

切換資料來源：
  修改 pipeline/config.py 中的 DATA_SOURCE = 'private' 即可。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time


def main():
    print('=' * 60)
    print('北捷轉乘流量推估 + GA 班距最佳化 Pipeline')
    print('=' * 60)

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Step 1: 載入資料
    # ------------------------------------------------------------------ #
    print('\n[Step 1/7] 載入資料...')
    from pipeline.step1_load_data import run as step1_run
    step1_result = step1_run()
    od_df = step1_result['typical_od']
    print(f'  典型日 OD：{len(od_df):,} 筆，{od_df["origin"].nunique()} 個起站')
    print(f'  資料來源：{step1_result["data_source"]}')

    # ------------------------------------------------------------------ #
    # Step 2: 路網分析
    # ------------------------------------------------------------------ #
    print('\n[Step 2/7] 路網拓撲分析...')
    from pipeline.step2_network_analysis import load_network_graph, load_station_mapping, analyze_od_dataframe
    G = load_network_graph()
    mapping = load_station_mapping()
    path_df = analyze_od_dataframe(od_df, G, mapping)
    print(f'  路徑展開後：{len(path_df):,} 筆')

    # ------------------------------------------------------------------ #
    # Step 3: 流量估算
    # ------------------------------------------------------------------ #
    print('\n[Step 3/7] 轉乘站流量估算...')
    from pipeline.step3_flow_estimation import estimate_transfer_flow, build_heatmap_matrix
    flow_df = estimate_transfer_flow(path_df)
    print(f'  轉乘站數：{flow_df["transfer_station"].nunique()}')
    if len(flow_df) > 0:
        top_raw = flow_df.sort_values('estimated_trips', ascending=False).iloc[0]
        print(f'  校正前最高承壓：{top_raw["transfer_station"]} at {top_raw["hour"]}:00 '
              f'（比例 {top_raw["transfer_ratio"]*100:.1f}%）')

    # ------------------------------------------------------------------ #
    # Step 3.5: 進出人次校正
    # ------------------------------------------------------------------ #
    print('\n[Step 3.5/7] 進出人次校正（step3_5_calibration）...')
    from pipeline.step3_5_calibration import calibrate
    flow_df = calibrate(flow_df)
    if len(flow_df) > 0:
        top_cal = flow_df.sort_values('estimated_trips', ascending=False).iloc[0]
        print(f'  校正後最高承壓：{top_cal["transfer_station"]} at {top_cal["hour"]}:00 '
              f'（scale_factor={top_cal["scale_factor"]:.3f}）')

    # ------------------------------------------------------------------ #
    # Step 3.6: 列車到離站時間整合
    # ------------------------------------------------------------------ #
    print('\n[Step 3.6/7] 列車到離站時間整合...')
    from pipeline.step3_6_arrival_integration import integrate_arrival
    flow_df = integrate_arrival(flow_df)

    heatmap_df = build_heatmap_matrix(flow_df)

    # ------------------------------------------------------------------ #
    # Step 4: GA 最佳化
    # ------------------------------------------------------------------ #
    print('\n[Step 4/7] GA 班距最佳化（DEAP）...')
    from pipeline.step4_ga_optimizer import run_ga, headway_to_dataframe
    if len(flow_df) > 0:
        best_plan, logbook = run_ga(flow_df, verbose=False)
        headway_df = headway_to_dataframe(best_plan)
        print(f'  最佳化完成，建議班距範圍：'
              f'{headway_df["suggested_headway_min"].min():.1f} ~ '
              f'{headway_df["suggested_headway_min"].max():.1f} 分鐘')
    else:
        import pandas as pd
        headway_df = pd.DataFrame()
        print('  無轉乘流量資料，跳過 GA')

    # ------------------------------------------------------------------ #
    # Step 4.5: 跨線換乘同步最佳化
    # ------------------------------------------------------------------ #
    print('\n[Step 4.5/7] 跨線換乘等待時間同步最佳化...')
    from pipeline.step4_5_transfer_sync import optimize_transfer_sync
    sync_df = optimize_transfer_sync(headway_df, flow_df)

    # ------------------------------------------------------------------ #
    # Step 5: 匯出 JSON
    # ------------------------------------------------------------------ #
    print('\n[Step 5/7] 匯出 JSON...')
    from pipeline.step5_export import export_to_json
    export_to_json(flow_df, heatmap_df, headway_df)

    # ------------------------------------------------------------------ #
    # Step 5T: 完整時刻表生成
    # ------------------------------------------------------------------ #
    print('\n[Step 5T/7] 完整時刻表生成...')
    from pipeline.step5_timetable import generate_timetable, export_timetable
    timetable_df = generate_timetable(headway_df, sync_df)
    export_timetable(timetable_df, headway_df, sync_df)

    elapsed = time.time() - t0
    print(f'\n完成！總耗時 {elapsed:.1f} 秒')
    print('啟動視覺化：python pipeline/serve_dashboard.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
