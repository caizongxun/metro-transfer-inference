# Pipeline 模組說明

本資料夾為北捷轉乘流量推估 + GA 班距最佳化的完整 pipeline。

## 架構

```
pipeline/
├── step1_load_data.py        # 載入公開 OD 分時資料（或未來比賽私有資料）
├── step2_network_analysis.py # 路網拓樸分析：每個 OD 對 → 必要/可能轉乘站
├── step3_flow_estimation.py  # 轉乘站分時流量比例估算
├── step4_ga_optimizer.py     # GA 班距最佳化（DEAP）
├── step5_export.py           # 輸出 JSON 供前端使用
├── dashboard/
│   └── index.html            # 網頁視覺化（熱力圖 + 時間軸動畫）
├── data_adapter/
│   ├── public_od_adapter.py  # 公開資料介接器（現在用）
│   └── private_afc_adapter.py # 比賽私有票卡資料介接器（預留窗口）
└── config.py                 # 全域設定
```

## 快速執行

```bash
pip install -r ../requirements.txt
pip install deap flask

# 全流程跑一次
python run_pipeline.py

# 啟動視覺化網頁
python serve_dashboard.py
# 瀏覽器開 http://localhost:5050
```

## 資料窗口說明

- **現在**：`data_adapter/public_od_adapter.py` 讀取台北市開放資料，輸出標準格式
- **比賽後**：改用 `data_adapter/private_afc_adapter.py`，輸出同樣的標準格式
- `step1_load_data.py` 只認標準格式，不管資料來源，切換時**不需要改 step2~step5**

## 能計算什麼、不能計算什麼

**能算**：各轉乘站在各時段的相對承壓比例（例如台北車站在早尖峰承接約 18% 的總轉乘量）

**暫時不能算**：精確的絕對人次（需要票卡層級旅行時間，比賽私有資料才有）

**換入私有資料後能額外算**：個體層級路徑機率、精確候車時間、列車對齊
