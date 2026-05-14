"""
visualize_results.py
從訓練輸出檔建立視覚化圖表。
使用 matplotlib 輸出 PNG，不需要 Chrome / kaleido。

用途：
    讀取下列檔案並輸出 PNG / CSV：
      - data/transfer_labels.csv
      - output_feature_importance.csv
      - output_anomalies.csv

執行方式：
    python visualize_results.py
    python visualize_results.py --label data/transfer_labels.csv \
        --fi output_feature_importance.csv \
        --anom output_anomalies.csv
    python visualize_results.py --outdir output/viz

輸出圖表：
    - feature_importance.png    特徵重要性 Top 10
    - risk_stations.png         風險轉乘站排名
    - surge_drop.png            爆量 vs 導量事件比較
    - daily_transfer_trend.png  每日轉乘流量趨勢
    - hourly_profiles.png       Top 5 站小時平均流量曲線
    - weekday_weekend_pattern.png 平日 vs 假日小時檔
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非互動後端，不需要顯示器
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams

# 中文字型支持（若環境有安裝 NotoSansCJK）
rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'DejaVu Sans', 'sans-serif']
rcParams['axes.unicode_minus'] = False

COLOR_BLUE   = '#4E79A7'
COLOR_ORANGE = '#F28E2B'
COLOR_RED    = '#E15759'
COLOR_TEAL   = '#59A14F'


def parse_args():
    p = argparse.ArgumentParser(description='視覚化 Metro Transfer ML 輸出')
    p.add_argument('--label',  default='data/transfer_labels.csv')
    p.add_argument('--fi',     default='output_feature_importance.csv')
    p.add_argument('--anom',   default='output_anomalies.csv')
    p.add_argument('--outdir', default='output/viz')
    p.add_argument('--topk',   type=int, default=10)
    return p.parse_args()


def savefig(path: str, title: str = ''):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  完成：{path}')


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ── 讀入 CSV ─────────────────────────────────────────────
    label_df = pd.read_csv(args.label, encoding='utf-8-sig')
    fi_df    = pd.read_csv(args.fi,    encoding='utf-8-sig')
    anom_df  = pd.read_csv(args.anom,  encoding='utf-8-sig')

    label_df['日期'] = pd.to_datetime(label_df['日期'])
    label_df['weekday'] = label_df['日期'].dt.dayofweek

    if 'is_anomaly' not in anom_df.columns:
        anom_df['is_anomaly'] = 1

    # 風險站統計
    risk_df = (
        anom_df.groupby('transfer_station')
        .agg(anomaly_count=('is_anomaly', 'sum'))
        .reset_index()
        .sort_values('anomaly_count', ascending=False)
        .head(args.topk)
    )
    if 'anomaly_dir' in anom_df.columns:
        pivot = (
            anom_df[anom_df['transfer_station'].isin(risk_df['transfer_station'])]
            .groupby(['transfer_station', 'anomaly_dir'])
            .size().unstack(fill_value=0).reset_index()
        )
        for col in ['surge', 'drop']:
            if col not in pivot.columns:
                pivot[col] = 0
        risk_df = risk_df.merge(pivot[['transfer_station', 'surge', 'drop']],
                                on='transfer_station', how='left').fillna(0)
    else:
        risk_df['surge'] = 0
        risk_df['drop']  = 0

    # ── 1. 特徵重要性 Top 10 ───────────────────────────────
    fi_top = fi_df.sort_values('importance', ascending=False).head(args.topk)\
                  .sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(fi_top['feature'], fi_top['importance'], color=COLOR_TEAL)
    ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'特徵重要性 Top {args.topk}（XGBoost）', fontsize=14, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    savefig(os.path.join(args.outdir, 'feature_importance.png'))

    # ── 2. 風險轉乘站排名 ────────────────────────────────
    rd = risk_df.sort_values('anomaly_count', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(rd['transfer_station'], rd['anomaly_count'], color=COLOR_RED)
    ax.bar_label(bars, padding=2)
    ax.set_xlabel('Transfer Station')
    ax.set_ylabel('Anomaly Count')
    ax.set_title('風險轉乘站 Top 10（歷史異常次數）', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    ax.spines[['top', 'right']].set_visible(False)
    savefig(os.path.join(args.outdir, 'risk_stations.png'))

    # ── 3. 爆量 vs 導量 ───────────────────────────────────
    rd2 = risk_df.sort_values('anomaly_count', ascending=False)
    x = np.arange(len(rd2))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, rd2['surge'], width=w, label='Surge 爆量', color=COLOR_RED)
    ax.bar(x + w/2, rd2['drop'],  width=w, label='Drop 導量',  color=COLOR_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(rd2['transfer_station'], rotation=30, ha='right')
    ax.set_xlabel('Transfer Station')
    ax.set_ylabel('Events')
    ax.set_title('爆量 vs 導量事件比較（Top 10 站）', fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    savefig(os.path.join(args.outdir, 'surge_drop.png'))

    # ── 4. 每日轉乘流量趨勢 ──────────────────────────────
    trend_df = (
        label_df.groupby('日期', as_index=False)['expected_flow']
        .sum().sort_values('日期')
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(trend_df['日期'], trend_df['expected_flow'], color=COLOR_BLUE, linewidth=1.5)
    ax.fill_between(trend_df['日期'], trend_df['expected_flow'], alpha=0.15, color=COLOR_BLUE)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Flow')
    ax.set_title('每日轉乘流量趨勢（全系統所有轉乘站）', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k' if v >= 1000 else str(int(v))))
    ax.spines[['top', 'right']].set_visible(False)
    savefig(os.path.join(args.outdir, 'daily_transfer_trend.png'))

    # ── 5. Top 5 站小時平均流量曲線 ───────────────────────
    top_stations = (
        label_df.groupby('transfer_station')['expected_flow']
        .sum().sort_values(ascending=False)
        .head(min(5, label_df['transfer_station'].nunique()))
        .index.tolist()
    )
    profile_df = (
        label_df[label_df['transfer_station'].isin(top_stations)]
        .groupby(['transfer_station', 'hour'], as_index=False)['expected_flow']
        .mean()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for stn, grp in profile_df.groupby('transfer_station'):
        ax.plot(grp['hour'], grp['expected_flow'], marker='o', markersize=4, label=stn)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Avg Flow')
    ax.set_xticks(range(0, 24, 2))
    ax.set_title('高流量站小時平均轉乘流量曲線（Top 5 站）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    savefig(os.path.join(args.outdir, 'hourly_profiles.png'))

    # ── 6. 平日 vs 假日 ───────────────────────────────────
    label_df['day_type'] = np.where(label_df['weekday'] >= 5, 'Weekend 假日', 'Weekday 平日')
    wd_df = label_df.groupby(['day_type', 'hour'], as_index=False)['expected_flow'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Weekday 平日': COLOR_BLUE, 'Weekend 假日': COLOR_ORANGE}
    for dtype, grp in wd_df.groupby('day_type'):
        ax.plot(grp['hour'], grp['expected_flow'], marker='o', markersize=4,
                label=dtype, color=colors.get(dtype, COLOR_TEAL))
    ax.set_xlabel('Hour')
    ax.set_ylabel('Avg Flow')
    ax.set_xticks(range(0, 24, 2))
    ax.set_title('平日 vs 假日 轉乘小時檔', fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    savefig(os.path.join(args.outdir, 'weekday_weekend_pattern.png'))

    # ── 儲存 CSV ─────────────────────────────────────────────
    risk_df.to_csv(os.path.join(args.outdir, 'risk_stations.csv'), index=False, encoding='utf-8-sig')
    trend_df.to_csv(os.path.join(args.outdir, 'daily_transfer_trend.csv'), index=False, encoding='utf-8-sig')
    profile_df.to_csv(os.path.join(args.outdir, 'hourly_profiles.csv'), index=False, encoding='utf-8-sig')
    print(f'\n全部完成！圖表儲存於 {args.outdir}/')


if __name__ == '__main__':
    main()
