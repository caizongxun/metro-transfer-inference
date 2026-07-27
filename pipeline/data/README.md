# pipeline/data/

下載公開資料後放到這個目錄。

## 需要的檔案

1. **od_hourly.csv** — 臺北捷運各站分時進出量統計 OD
   - 下載：https://data.taipei/dataset/detail?id=63f31c7e-7fc3-418b-bd82-b95158755b4d
   - 欄位：進站、出站、時段、人次

2. **station_inout.csv** — 臺北捷運各站進出人次
   - 下載：https://data.taipei/dataset/detail?id=178ebf06-0451-4ac1-bbba-c255ca1fdac6
   - 欄位：站名、日期、進站人次、出站人次

## 尚未下載時

`public_od_adapter.py` 找不到檔案時會自動使用內建模擬資料，pipeline 仍可完整執行。

## 比賽私有資料（比賽開始後）

在 `private/` 子目錄放入：
- `private/afc.csv`
- `private/crowding.csv`
- `private/timetable.csv`

然後修改 `pipeline/config.py`：
```python
DATA_SOURCE = 'private'
```
