"""
serve_dashboard.py
啟動本地 HTTP server 供前端 dashboard 使用。

執行：
  python pipeline/serve_dashboard.py
  瀏覽器開 http://localhost:5050
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 5050
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def log_message(self, format, *args):
        pass  # 關閉 access log


def main():
    data_path = os.path.join(DASHBOARD_DIR, 'data.json')
    if not os.path.exists(data_path):
        print('[Warning] data.json 不存在，請先執行 python pipeline/run_pipeline.py')
        print('[Warning] 繼續啟動 server，dashboard 將顯示空白狀態...')

    print(f'Dashboard 啟動中：http://localhost:{PORT}')
    print('停止：Ctrl+C')
    server = HTTPServer(('0.0.0.0', PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer 已停止')


if __name__ == '__main__':
    main()
