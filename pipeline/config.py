# config.py — 全域設定

# 資料來源：'public' 使用開放資料，'private' 使用比賽私有資料
DATA_SOURCE = 'public'

# 公開資料路徑（放在 pipeline/data/ 下）
PUBLIC_OD_PATH = 'pipeline/data/od_hourly.csv'       # 分時 OD 資料
PUBLIC_INOUT_PATH = 'pipeline/data/station_inout.csv' # 各站進出人次

# 比賽私有資料路徑（取得後填入）
PRIVATE_AFC_PATH = 'pipeline/data/private/afc.csv'
PRIVATE_CROWDING_PATH = 'pipeline/data/private/crowding.csv'
PRIVATE_TIMETABLE_PATH = 'pipeline/data/private/timetable.csv'

# 路網設定
NETWORK_JSON_PATH = 'data/network.json'
STATION_MAPPING_PATH = 'data/station_mapping.json'

# GA 超參數
GA_POPULATION = 80
GA_GENERATIONS = 120
GA_CROSSOVER_PROB = 0.7
GA_MUTATION_PROB = 0.2

# 班距範圍（分鐘）
HEADWAY_MIN = 2
HEADWAY_MAX = 10

# 分析時段（小時，0~23）
ANALYSIS_HOURS = list(range(6, 24))  # 06:00 ~ 23:00

# 轉乘站承壓警示閾值（相對比例，0~1）
PRESSURE_THRESHOLD = 0.15  # 單站承接超過 15% 總轉乘量視為高壓

# 輸出路徑
OUTPUT_JSON_PATH = 'pipeline/dashboard/data.json'
