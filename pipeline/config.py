# config.py — 全域設定

# 資料來源：'public' 使用開放資料，'private' 使用比賽私有資料
DATA_SOURCE = 'public'

# 公開資料路徑
PUBLIC_OD_DIR = 'data/od_raw'          # 存放所有月份 CSV 的目錄
PUBLIC_OD_PATH = 'pipeline/data/od_hourly.csv'       # 備用：單一合併檔（若已預處理）
PUBLIC_INOUT_PATH = 'pipeline/data/station_inout.csv'

# 資料篩選：控制要讀哪些月份（空 list = 全部）
# 建議一次取 3~6 個月防止 OOM，注釋原則不動
FILTER_YEARS = []          # 空 = 全部年份
FILTER_YEARMONTHS = ['202401', '202402', '202403',
                     '202404', '202405', '202406']  # 預設取 2024 上半年

# chunk 讀取行數（每次讀多少行再彙整）— 調小可降低峰値記憶體
CHUNK_SIZE = 50_000

# 比賽私有資料路徑
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
ANALYSIS_HOURS = list(range(6, 24))

# 轉乘站承壓警示閾值
PRESSURE_THRESHOLD = 0.15

# 輸出路徑
OUTPUT_JSON_PATH = 'pipeline/dashboard/data.json'
