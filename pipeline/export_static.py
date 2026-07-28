"""
export_static.py
把 data.json inline 進 index.html，產出完全自包含的靜態 HTML。
適用於無法開 HTTP server 的環境（Lightning AI、Colab、等）。

執行：
  python pipeline/export_static.py
  產出： pipeline/dashboard/dashboard_output.html
"""

import os
import json

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')
DATA_PATH     = os.path.join(DASHBOARD_DIR, 'data.json')
TEMPLATE_PATH = os.path.join(DASHBOARD_DIR, 'index.html')
OUTPUT_PATH   = os.path.join(DASHBOARD_DIR, 'dashboard_output.html')


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

    with open(TEMPLATE_PATH, encoding='utf-8') as f:
        html = f.read()

    # 將 fetch('./data.json') 的非同步載入改為直接嵌入資料
    # 找到 loadData() 函數，整個换採
    inline_script = f'<script>\nlet _INLINE_DATA = {json.dumps(data, ensure_ascii=False)};\n</script>'

    # 在 </head> 前插入 inline data
    html = html.replace('</head>', f'{inline_script}\n</head>', 1)

    # 改寫 loadData() ：直接用 _INLINE_DATA 而不是 fetch
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
    initDashboard();
    return;
  }
  try {
    const res = await fetch('./data.json');
    if (!res.ok) throw new Error('data.json not found');
    DATA = await res.json();
    initDashboard();
  } catch(e) {
    showEmpty('尚無資料', '請先執行 python pipeline/run_pipeline.py 產生 data.json');
  }
}'''

    if old_load in html:
        html = html.replace(old_load, new_load)
    else:
        # fallback: template 版本已變，用注訋區塊替換
        print('[Export] Warning: loadData() 樣式不匹配，改用 fallback 插入')
        html = html.replace(
            'loadData();',
            'if(typeof _INLINE_DATA!=="undefined"){DATA=_INLINE_DATA;initDashboard();}else{loadData();}',
            1
        )

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f'[Export] 產出：{OUTPUT_PATH}')
    print(f'[Export] 檔案大小：{size_kb:.0f} KB（可直接用瀏覽器開啟）')


if __name__ == '__main__':
    build_static_html()
