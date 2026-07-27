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

# 北捷主要路線
LINES = ['板南線', '淡水信義線', '中和新薊線', '松山新店線', '文湖線']
N_HOURS = len(ANALYSIS_HOURS)

# 各路線每班次容量（估算，滿載約1000人）
LINE_CAPACITY = {
    '板南線': 1000, '淡水信義線': 1000, '中和新薊線': 900,
    '松山新店線': 900, '文湖線': 400
}

# ANALYSIS_HOURS 的首尾，用於 clamp
_HOUR_MIN = min(ANALYSIS_HOURS)
_HOUR_MAX = max(ANALYSIS_HOURS)


def decode_individual(individual: list) -> dict:
    """
    將個體解碼為 {line: {hour: headway}} 的巢狀 dict。
    key 全部為 int。
    """
    result = {}
    for i, line in enumerate(LINES):
        result[line] = {}
        for j, hour in enumerate(ANALYSIS_HOURS):
            val = individual[i * N_HOURS + j]
            result[line][int(hour)] = max(HEADWAY_MIN, min(HEADWAY_MAX, round(val, 1)))
    return result


def _lookup_headway(headway_plan: dict, line: str, hour) -> float:
    """
    安全查詢班距：若 hour 不在 ANALYSIS_HOURS 內，
    clamp 到最近的已知時段。
    """
    h = int(hour)
    h_clamped = max(_HOUR_MIN, min(_HOUR_MAX, h))
    return headway_plan[line][h_clamped]


def fitness(individual: list, flow_df: pd.DataFrame):
    """
    適應度函數（最小化）：
    1. 加權等待時間：各轉乘站 × 時段的 estimated_trips × (headway/2)
    2. 超載懲罰：承壓比例超過閾値時加大懲罰
    """
    headway_plan = decode_individual(individual)
    total_wait = 0.0
    overload_penalty = 0.0

    for _, row in flow_df.iterrows():
        hour = row['hour']
        trips = row['estimated_trips']
        ratio = row['transfer_ratio']

        # 用安全查詢，避免 hour 超出 ANALYSIS_HOURS 範圍導致 KeyError
        avg_headway = np.mean([_lookup_headway(headway_plan, line, hour) for line in LINES])
        total_wait += trips * (avg_headway / 2)

        if ratio > PRESSURE_THRESHOLD:
            overload_penalty += trips * (ratio - PRESSURE_THRESHOLD) * 100

    return (total_wait + overload_penalty,)


def run_ga(flow_df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    執行 GA 最佳化，回傳最佳班距方案。
    """
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
    """將班距方案轉為 DataFrame 方便輸出"""
    rows = []
    for line, hours in headway_plan.items():
        for hour, headway in hours.items():
            rows.append({'line': line, 'hour': hour, 'suggested_headway_min': headway})
    return pd.DataFrame(rows).sort_values(['line', 'hour'])


if __name__ == '__main__':
    sample_flow = pd.DataFrame([
        {'transfer_station': '台北車站', 'hour': 8, 'estimated_trips': 500, 'transfer_ratio': 0.20},
        {'transfer_station': '忠孝復興', 'hour': 8, 'estimated_trips': 300, 'transfer_ratio': 0.12},
        {'transfer_station': '台北車站', 'hour': 18, 'estimated_trips': 480, 'transfer_ratio': 0.19},
    ])
    best, log = run_ga(sample_flow, verbose=True)
    df = headway_to_dataframe(best)
    print(df)
