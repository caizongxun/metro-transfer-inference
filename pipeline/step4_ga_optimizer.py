"""
step4_ga_optimizer.py
GA 班距最佳化（使用 DEAP）。

問題定義：
  決策變數：各路線在各時段的班距（分鐘）
  目標函數（最小化）：全路網乘客加權等待時間總和
  約束：班距在 [HEADWAY_MIN, HEADWAY_MAX]，高壓轉乘站不得超載

個體編碼：
  [line1_h6, line1_h7, ..., line1_h23, line2_h6, ..., lineN_h23]
  共 N_lines × N_hours 個基因，每個基因為班距（float）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from pipeline.config import (
    GA_POPULATION, GA_GENERATIONS, GA_CROSSOVER_PROB, GA_MUTATION_PROB,
    HEADWAY_MIN, HEADWAY_MAX, ANALYSIS_HOURS, PRESSURE_THRESHOLD
)

# 相鄰時段班距變化的平滑懲罰係數
# 每分鐘的跨時段班距差異都會乘以此係數加入 fitness
# 調高此值可讓班距曲線更平滑，調低可允許更多彈性
SMOOTH_WEIGHT = 0.5

# 北捷主要路線
LINES = ['板南線', '淡水信義線', '中和新蘆線', '松山新店線', '文湖線']
N_HOURS = len(ANALYSIS_HOURS)

# network.json line code -> 中文路線名
LINE_CODE_MAP = {
    'R':  '淡水信義線',
    'BL': '板南線',
    'G':  '松山新店線',
    'O':  '中和新蘆線',
    'BR': '文湖線',
    'V':  '淡水信義線',
    'Y':  '板南線',
}

STATION_LINE_MAP = {
    '中山':     ['淡水信義線', '松山新店線'],
    '台北車站': ['淡水信義線', '板南線'],
    '民權西路': ['淡水信義線', '中和新蘆線'],
    '東門':     ['淡水信義線', '松山新店線'],
    '大安':     ['淡水信義線', '文湖線'],
    '忠孝新生': ['板南線',     '中和新蘆線'],
    '忠孝復興': ['板南線',     '文湖線'],
    '西門':     ['板南線',     '松山新店線'],
    '南港':     ['板南線'],
    '南港展覽館': ['板南線'],
    '南京復興': ['松山新店線', '文湖線'],
    '松江南京': ['松山新店線', '中和新蘆線'],
    '古亭':     ['松山新店線', '中和新蘆線'],
    '公館':     ['松山新店線', '板南線'],
    '七張':     ['松山新店線'],
    '新埔':     ['板南線'],
    '北投':     ['淡水信義線'],
}

LINE_CAPACITY = {
    '板南線': 1000, '淡水信義線': 1000, '中和新蘆線': 900,
    '松山新店線': 900, '文湖線': 400
}

_HOUR_MIN = min(ANALYSIS_HOURS)
_HOUR_MAX = max(ANALYSIS_HOURS)


def decode_individual(individual: list) -> dict:
    result = {}
    for i, line in enumerate(LINES):
        result[line] = {}
        for j, hour in enumerate(ANALYSIS_HOURS):
            val = individual[i * N_HOURS + j]
            result[line][int(hour)] = max(HEADWAY_MIN, min(HEADWAY_MAX, round(val, 1)))
    return result


def _lookup_headway(headway_plan: dict, line: str, hour) -> float:
    h = int(hour)
    h_clamped = max(_HOUR_MIN, min(_HOUR_MAX, h))
    return headway_plan[line][h_clamped]


def _station_avg_headway(headway_plan: dict, station: str, hour) -> float:
    lines = STATION_LINE_MAP.get(station, LINES)
    valid_lines = [l for l in lines if l in headway_plan]
    if not valid_lines:
        valid_lines = LINES
    return np.mean([_lookup_headway(headway_plan, line, hour) for line in valid_lines])


def fitness(individual: list, flow_df: pd.DataFrame):
    """
    適應度函數（最小化）：

    1. 加權等待時間：
       各轉乘站 × 時段的 estimated_trips × (station_avg_headway / 2)

    2. 超載懲罰（二次懲罰）：
       承壓比例超過閾值時，懲罰 = trips × excess² × HEADWAY_MAX × 10

    3. 平滑懲罰（新增）：
       各路線相鄰時段班距差的絕對值之和 × SMOOTH_WEIGHT
       防止班距在相鄰時段之間大幅跳動，使建議班距曲線更平滑
    """
    headway_plan = decode_individual(individual)
    total_wait = 0.0
    overload_penalty = 0.0

    for _, row in flow_df.iterrows():
        hour     = row['hour']
        trips    = row['estimated_trips']
        ratio    = row['transfer_ratio']
        station  = row.get('transfer_station', '')

        avg_headway = _station_avg_headway(headway_plan, station, hour)
        total_wait += trips * (avg_headway / 2)

        if ratio > PRESSURE_THRESHOLD:
            excess = ratio - PRESSURE_THRESHOLD
            overload_penalty += trips * (excess ** 2) * HEADWAY_MAX * 10

    # --- 平滑懲罰：相鄰時段班距差的 L1 norm ---
    smoothness_penalty = 0.0
    hours_sorted = sorted(ANALYSIS_HOURS)
    for line in LINES:
        for idx in range(len(hours_sorted) - 1):
            h_curr = hours_sorted[idx]
            h_next = hours_sorted[idx + 1]
            diff = abs(headway_plan[line][h_curr] - headway_plan[line][h_next])
            smoothness_penalty += diff
    smoothness_penalty *= SMOOTH_WEIGHT

    return (total_wait + overload_penalty + smoothness_penalty,)


def run_ga(flow_df: pd.DataFrame, verbose: bool = True) -> dict:
    n_genes = len(LINES) * N_HOURS

    if 'FitnessMin' in creator.__dict__:
        del creator.FitnessMin
    if 'Individual' in creator.__dict__:
        del creator.Individual

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('attr_headway', random.uniform, HEADWAY_MIN, HEADWAY_MAX)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.attr_headway, n=n_genes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', fitness, flow_df=flow_df)
    toolbox.register('mate', tools.cxBlend, alpha=0.3)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)

    pop = toolbox.population(n=GA_POPULATION)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min)
    stats.register('avg', np.mean)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=GA_CROSSOVER_PROB,
        mutpb=GA_MUTATION_PROB,
        ngen=GA_GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=verbose
    )

    best = decode_individual(hof[0])
    best_fitness = hof[0].fitness.values[0]
    if verbose:
        print(f"\n最佳適應度（加權等待時間）：{best_fitness:,.1f}")

    return best, logbook


def headway_to_dataframe(headway_plan: dict) -> pd.DataFrame:
    rows = []
    for line, hours in headway_plan.items():
        for hour, headway in hours.items():
            rows.append({'line': line, 'hour': hour, 'suggested_headway_min': headway})
    return pd.DataFrame(rows).sort_values(['line', 'hour'])


if __name__ == '__main__':
    sample_flow = pd.DataFrame([
        {'transfer_station': '台北車站', 'hour': 8,  'estimated_trips': 500, 'transfer_ratio': 0.20},
        {'transfer_station': '忠孝復興', 'hour': 8,  'estimated_trips': 300, 'transfer_ratio': 0.12},
        {'transfer_station': '台北車站', 'hour': 18, 'estimated_trips': 480, 'transfer_ratio': 0.19},
    ])
    best, log = run_ga(sample_flow, verbose=True)
    df = headway_to_dataframe(best)
    print(df)
