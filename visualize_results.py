"""
visualize_results.py
從訓練輸出檔建立視覚化圖表。

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
    - hourly_profiles.png       指定站各小時平均流量曲線
    - weekday_weekend_pattern.png 平日 vs 假日小時檔
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args():
    p = argparse.ArgumentParser(description='視覚化 Metro Transfer ML 輸出')
    p.add_argument('--label', default='data/transfer_labels.csv')
    p.add_argument('--fi',    default='output_feature_importance.csv')
    p.add_argument('--anom',  default='output_anomalies.csv')
    p.add_argument('--outdir', default='output/viz')
    p.add_argument('--topk',  type=int, default=10)
    return p.parse_args()


def save_fig(fig, path: str, caption: str, description: str = ''):
    fig.write_image(path)
    with open(path + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump({'caption': caption, 'description': description}, f, ensure_ascii=False)
    print(f'  完成：{path}')


def common_layout(fig, title: str, subtitle: str = ''):
    text = title
    if subtitle:
        text += f'<br><span style="font-size: 16px; font-weight: normal;">{subtitle}</span>'
    fig.update_layout(
        title={'text': text},
        font=dict(size=14),
        margin=dict(t=100, b=60, l=80, r=40),
    )
    return fig


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ── 讀入 ─────────────────────────────────────────────────
    label_df = pd.read_csv(args.label, encoding='utf-8-sig')
    fi_df    = pd.read_csv(args.fi,    encoding='utf-8-sig')
    anom_df  = pd.read_csv(args.anom,  encoding='utf-8-sig')

    label_df['日期'] = pd.to_datetime(label_df['日期'])
    label_df['weekday'] = label_df['日期'].dt.dayofweek
    label_df['date_str'] = label_df['日期'].dt.strftime('%Y-%m-%d')

    if 'is_anomaly' not in anom_df.columns:
        anom_df['is_anomaly'] = 1

    # 建立風險站統計表
    risk_df = (
        anom_df.groupby('transfer_station')
        .agg(
            anomaly_count=('is_anomaly', 'sum'),
            avg_z=('z_score', 'mean') if 'z_score' in anom_df.columns else ('is_anomaly', 'count'),
        )
        .reset_index()
        .sort_values('anomaly_count', ascending=False)
        .head(args.topk)
    )

    if 'anomaly_dir' in anom_df.columns:
        pivot = (
            anom_df[anom_df['transfer_station'].isin(risk_df['transfer_station'])]
            .groupby(['transfer_station', 'anomaly_dir'])
            .size().unstack(fill_value=0)
            .reset_index()
        )
        for col in ['surge', 'drop', 'normal']:
            if col not in pivot.columns:
                pivot[col] = 0
        risk_df = risk_df.merge(
            pivot[['transfer_station', 'surge', 'drop']],
            on='transfer_station', how='left'
        ).fillna(0)
    else:
        risk_df['surge'] = 0
        risk_df['drop']  = 0

    # ── 1. 特徵重要性 ───────────────────────────────────────
    fi_top = fi_df.sort_values('importance', ascending=True).tail(args.topk)
    fig = px.bar(
        fi_top, x='importance', y='feature', orientation='h',
        color='importance', color_continuous_scale='Teal',
        text=fi_top['importance'].map(lambda v: f'{v:.3f}')
    )
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig = common_layout(fig,
        '特徵重要性 Top 10（XGBoost）',
        'is_peak 與 transfer_station_id 對模型預測貢獻最大')
    fig.update_xaxes(title_text='Importance')
    fig.update_yaxes(title_text='Feature', automargin=True)
    save_fig(fig, os.path.join(args.outdir, 'feature_importance.png'),
             '特徵重要性 Top 10', 'XGBoost 模型最重要的 10 個特徵及其貢獻程度。')

    # ── 2. 風險轉乘站排行 ────────────────────────────────
    fig = px.bar(
        risk_df.sort_values('anomaly_count', ascending=False),
        x='transfer_station', y='anomaly_count',
        color='anomaly_count', color_continuous_scale='OrRd',
        text='anomaly_count'
    )
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig = common_layout(fig,
        '風險轉乘站 Top 10（歷史異常次數）',
        '歷史上最常出現異常流量的轉乘站')
    fig.update_xaxes(title_text='Transfer Station')
    fig.update_yaxes(title_text='Anomaly Count')
    save_fig(fig, os.path.join(args.outdir, 'risk_stations.png'),
             '風險轉乘站排名', '依異常次數排名的 Top 10 轉乘站。')

    # ── 3. 爆量 vs 導量 ────────────────────────────────────
    long_df = risk_df[['transfer_station', 'surge', 'drop']].melt(
        id_vars='transfer_station', var_name='type', value_name='count'
    )
    long_df['type'] = long_df['type'].map({'surge': '爆量 (Surge)', 'drop': '導量 (Drop)'})
    fig = px.bar(
        long_df, x='transfer_station', y='count',
        color='type', barmode='stack',
        color_discrete_map={'爆量 (Surge)': '#E15759', '導量 (Drop)': '#4E79A7'},
        text='count'
    )
    fig.update_traces(textposition='inside', cliponaxis=False)
    fig = common_layout(fig,
        '爆量 vs 導量事件比較（Top 10 站）',
        '紅色=流量爆表，藍色=低於期望的迷失客流')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
    fig.update_xaxes(title_text='Transfer Station')
    fig.update_yaxes(title_text='Events')
    save_fig(fig, os.path.join(args.outdir, 'surge_drop.png'),
             '爆量與導量事件', '堆疊条形圖比較各站爆量與導量異常次數。')

    # ── 4. 每日轉乘流量趨勢 ───────────────────────────────
    trend_df = (
        label_df.groupby('日期', as_index=False)['expected_flow']
        .sum()
        .sort_values('日期')
    )
    fig = px.line(trend_df, x='日期', y='expected_flow')
    fig.update_traces(fill='tozeroy', line_color='#4E79A7')
    fig = common_layout(fig,
        '每日轉乘流量趨勢',
        '全系統所有轉乘站当日總 expected_flow')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Total Flow')
    save_fig(fig, os.path.join(args.outdir, 'daily_transfer_trend.png'),
             '每日轉乘流量趨勢', '所有轉乘站每日總期望流量折線圖，可看尖峰/假日跑勢。')

    # ── 5. 高流量站小時檔 ───────────────────────────────
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
    fig = px.line(
        profile_df, x='hour', y='expected_flow',
        color='transfer_station',
        markers=True
    )
    fig = common_layout(fig,
        '指定站小時平均轉乘流量曲線（Top 5 站）',
        '平均小時 expected_flow，可鑑別尖峰二峰型態')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))
    fig.update_xaxes(title_text='Hour', tickmode='linear', tick0=0, dtick=2)
    fig.update_yaxes(title_text='Avg Flow')
    save_fig(fig, os.path.join(args.outdir, 'hourly_profiles.png'),
             '高流量站小時檔', 'Top 5 轉乘站按小時分組的平均流量折線圖。')

    # ── 6. 平日 vs 假日 ───────────────────────────────────
    label_df['day_type'] = np.where(label_df['weekday'] >= 5, 'Weekend 假日', 'Weekday 平日')
    wd_df = label_df.groupby(['day_type', 'hour'], as_index=False)['expected_flow'].mean()
    fig = px.line(
        wd_df, x='hour', y='expected_flow',
        color='day_type',
        color_discrete_map={'Weekday 平日': '#4E79A7', 'Weekend 假日': '#F28E2B'},
        markers=True
    )
    fig = common_layout(fig,
        '平日 vs 假日 轉乘小時檔',
        '平日尖峰明顔，假日流量曲線較平坦')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))
    fig.update_xaxes(title_text='Hour', tickmode='linear', tick0=0, dtick=2)
    fig.update_yaxes(title_text='Avg Flow')
    save_fig(fig, os.path.join(args.outdir, 'weekday_weekend_pattern.png'),
             '平日與假日模式', '平日與假日轉乘小時平均流量比較。')

    # ── 儲存 CSV ─────────────────────────────────────────────
    risk_df.to_csv(os.path.join(args.outdir, 'risk_stations.csv'), index=False, encoding='utf-8-sig')
    trend_df.to_csv(os.path.join(args.outdir, 'daily_transfer_trend.csv'), index=False, encoding='utf-8-sig')
    profile_df.to_csv(os.path.join(args.outdir, 'hourly_profiles.csv'), index=False, encoding='utf-8-sig')
    print(f'\n全部完成！圖表儲存於 {args.outdir}/')


if __name__ == '__main__':
    main()
