"""
export_static.py
把 data.json 與 timetable_summary.json inline 進 index.html，
產出完全自包含的靜態 HTML。

執行：
  python pipeline/export_static.py
  產出： pipeline/dashboard/dashboard_output.html
"""

import os
import json

DASHBOARD_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')
DATA_PATH       = os.path.join(DASHBOARD_DIR, 'data.json')
TIMETABLE_PATH  = os.path.join('data', 'output', 'timetable_summary.json')
TEMPLATE_PATH   = os.path.join(DASHBOARD_DIR, 'index.html')
OUTPUT_PATH     = os.path.join(DASHBOARD_DIR, 'dashboard_output.html')


def build_static_html():
    if not os.path.exists(DATA_PATH):
        print(f'[Export] 找不到 {DATA_PATH}')
        print('[Export] 請先執行 python pipeline/run_pipeline.py')
        return

    if not os.path.exists(TEMPLATE_PATH):
        print(f'[Export] 找不到 {TEMPLATE_PATH}')
        return

    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)

    # 讀入時刻表 JSON（若不存在則為空 dict）
    timetable = {}
    if os.path.exists(TIMETABLE_PATH):
        with open(TIMETABLE_PATH, encoding='utf-8') as f:
            timetable = json.load(f)
        print(f'[Export] 時刻表資料：{TIMETABLE_PATH} ({len(timetable)} 條路線)')
    else:
        print(f'[Export] 找不到 {TIMETABLE_PATH}，時刻表頁籤將顯示空狀態')

    with open(TEMPLATE_PATH, encoding='utf-8') as f:
        html = f.read()

    # 同時 inline 兩份資料
    inline_script = (
        f'<script>\n'
        f'let _INLINE_DATA = {json.dumps(data, ensure_ascii=False)};\n'
        f'let _INLINE_TIMETABLE = {json.dumps(timetable, ensure_ascii=False)};\n'
        f'</script>'
    )

    html = html.replace('</head>', f'{inline_script}\n</head>', 1)

    # 改寫 loadData()
    old_load = '''async function loadData() {
  try {
    const res = await fetch('./data.json');
    if (!res.ok) throw new Error('data.json not found');
    DATA = await res.json();
    initDashboard();
  } catch(e) {
    showEmpty('尚無資料', '請先執行 python pipeline/run_pipeline.py 產生 data.json');
  }
}'''

    new_load = '''async function loadData() {
  if (typeof _INLINE_DATA !== 'undefined') {
    DATA = _INLINE_DATA;
    TIMETABLE = (typeof _INLINE_TIMETABLE !== 'undefined') ? _INLINE_TIMETABLE : {};
    initDashboard();
    return;
  }
  try {
    const res = await fetch('./data.json');
    if (!res.ok) throw new Error('data.json not found');
    DATA = await res.json();
    try {
      const tr = await fetch('../data/output/timetable_summary.json');
      TIMETABLE = tr.ok ? await tr.json() : {};
    } catch(_) { TIMETABLE = {}; }
    initDashboard();
  } catch(e) {
    showEmpty('尚無資料', '請先執行 python pipeline/run_pipeline.py 產生 data.json');
  }
}'''

    if old_load in html:
        html = html.replace(old_load, new_load)
    else:
        print('[Export] Warning: loadData() 樣式不匹配，改用 fallback 插入')
        html = html.replace(
            'loadData();',
            'if(typeof _INLINE_DATA!=="undefined"){DATA=_INLINE_DATA;TIMETABLE=(typeof _INLINE_TIMETABLE!=="undefined")?_INLINE_TIMETABLE:{};initDashboard();}else{loadData();}',
            1
        )

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f'[Export] 產出：{OUTPUT_PATH}')
    print(f'[Export] 檔案大小：{size_kb:.0f} KB（可直接用瀏覽器開啟）')


if __name__ == '__main__':
    build_static_html()
