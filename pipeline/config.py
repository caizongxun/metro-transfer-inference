# config.py — 全域設定

# 資料來源：'public' 使用開放資料，'private' 使用比賽私有資料
DATA_SOURCE = 'public'

# 公開資料路徑
# 直接指向 data/od_raw/ 目錄，adapter 會自動合併所有月份 CSV
PUBLIC_OD_DIR = 'data/od_raw'          # 存放所有月份 CSV 的目錄
PUBLIC_OD_PATH = 'pipeline/data/od_hourly.csv'       # 備用：單一合併檔（若已預處理）
PUBLIC_INOUT_PATH = 'pipeline/data/station_inout.csv' # 各站進出人次（選用）

# 資料篩選：只取這些年月的資料（空 list 代表全部）
# 例如只取 2024 年：FILTER_YEARS = [2024]
# 例如只取最近半年：FILTER_YEARMONTHS = ['202501','202502','202503','202504','202505','202506']
FILTER_YEARS = []          # 空 = 全部年份
FILTER_YEARMONTHS = []     # 空 = 全部年月（比 FILTER_YEARS 更細）

# 比賽私有資料路徑（取得後填入，切換 DATA_SOURCE = 'private' 即可）
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
