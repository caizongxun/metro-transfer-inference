# 北捷轉乘路徑推估引擎

> Metro Transfer Inference Engine

利用 AFC 票卡交易紀錄、列車到離站時間、列車擁擠度三種資料，推估旅客在轉乘站之間的路徑選擇機率與人流分佈。

---

## 專案背景

北捷每日 700 萬人次，AFC 系統只記錄「進站 A、出站 C」，無法得知旅客在中間是否經過轉乘站 B。本專案透過機率推論模型，估計每個 OD（Origin-Destination）在不同時段下，旅客最可能走過的轉乘路徑，並聚合成可供營運決策的熱圖與警示。

---

## 模組說明

```
metro-transfer-inference/
├── data/
│   ├── sample_afc.csv              # AFC 票卡交易樣本
│   ├── sample_timetable.csv        # 列車到離站時間樣本
│   ├── sample_crowding.csv         # 列車擁擠度樣本
│   └── network.json                # 路網拓樸（站點/轉乘關係）
│
├── modules/
│   ├── network_builder.py          # 路網圖建立（NetworkX）
│   ├── afc_processor.py            # AFC 資料清洗與 OD 配對
│   ├── path_inference.py           # 候選路徑產生與機率推論
│   └── output_formatter.py         # 輸出熱圖與分流建議
│
├── main.py                         # 主執行入口
├── requirements.txt
└── README.md
```

---

## 資料來源

北捷提供（hackathon 期間）：
- 票卡交易紀錄（AFC）
- 列車到離站時間
- 列車擁擠度（綠/黃/橘/紅）

公開補充：
- 臺北捷運各站分時進出量統計 OD：https://data.taipei/dataset/detail?id=63f31c7e-7fc3-418b-bd82-b95158755b4d
- 臺北捷運各站進出人次：https://data.taipei/dataset/detail?id=178ebf06-0451-4ac1-bbba-c255ca1fdac6

---

## 快速開始

```bash
pip install -r requirements.txt
python main.py
```

---

## 輸出格式

系統輸出三層結果：

1. **OD 路徑比例**：某 OD 在某時段，各候選路徑的機率分佈。
2. **轉乘站承壓排名**：各轉乘站在尖峰時段承接轉乘人流的推估比例。
3. **分流建議**：若主要轉乘站擁擠過高，推薦替代轉乘站。

---

## 技術架構

```
AFC 資料
  └─→ OD 配對 & 旅行時間計算
           └─→ 候選路徑產生（K-Shortest Paths）
                    └─→ 理論時間 vs 實際時間符合度
                             └─→ 擁擠度加權（MNL 分數）
                                      └─→ 路徑機率聚合
                                               └─→ 輸出熱圖 / 警示 / 分流建議
```

---

## 國際參考案例

| 城市 | 方法 | 準確率 |
|------|------|--------|
| 倫敦 Underground（Oyster Card） | Bayesian Mixture Model | ~82% |
| 紐約 Subway（MetroCard） | Trip Chaining | ~95%（OD估計） |
| 新加坡 MRT（EZ-Link） | Multinomial Logit（MNL） | ~78% |
| 成都 Metro | MNL + Two-way Search | ~80% |
| 北京 Subway | 時變狀態機率模型 | ~85% |
