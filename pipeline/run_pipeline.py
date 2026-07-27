"""
run_pipeline.py
全流程執行入口。依序執行 step1 ~ step5。

執行方式：
  cd <repo root>
  python pipeline/run_pipeline.py

切換資料來源：
  修改 pipeline/config.py 中的 DATA_SOURCE = 'private' 即可，其餘不需改動。
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

    # Step 1: 載入資料
    print('\n[Step 1/5] 載入資料...')
    from pipeline.step1_load_data import run as step1_run
    step1_result = step1_run()
    od_df = step1_result['typical_od']   # 典型日日平均，供後續步驟使用
    raw_od = step1_result['raw_od']      # 逐日原始資料（保留備用）
    print(f'  典型日 OD：{len(od_df):,} 筆，{od_df["origin"].nunique()} 個起站')
    print(f'  原始資料：{len(raw_od):,} 筆，浵蓋 {raw_od["date"].nunique()} 天')

    # Step 2: 路網分析
    print('\n[Step 2/5] 路網拓撲分析...')
    from pipeline.step2_network_analysis import load_network_graph, load_station_mapping, analyze_od_dataframe
    G = load_network_graph()
    mapping = load_station_mapping()
    path_df = analyze_od_dataframe(od_df, G, mapping)
    print(f'  路徑展開後：{len(path_df):,} 筆')

    # Step 3: 流量估算
    print('\n[Step 3/5] 轉乘站流量估算...')
    from pipeline.step3_flow_estimation import estimate_transfer_flow, build_heatmap_matrix
    flow_df = estimate_transfer_flow(path_df)
    heatmap_df = build_heatmap_matrix(flow_df)
    print(f'  轉乘站數：{flow_df["transfer_station"].nunique()}')
    if len(flow_df) > 0:
        top = flow_df.sort_values('estimated_trips', ascending=False).iloc[0]
        print(f'  最高承壓：{top["transfer_station"]} at {top["hour"]}:00 '
              f'（比例 {top["transfer_ratio"]*100:.1f}%）')

    # Step 4: GA 最佳化
    print('\n[Step 4/5] GA 班距最佳化（DEAP）...')
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

    # Step 5: 匙出
    print('\n[Step 5/5] 匙出 JSON...')
    from pipeline.step5_export import export_to_json
    export_to_json(flow_df, heatmap_df, headway_df)

    elapsed = time.time() - t0
    print(f'\n完成！總耗時 {elapsed:.1f} 秒')
    print('啟動視覺化：python pipeline/serve_dashboard.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
