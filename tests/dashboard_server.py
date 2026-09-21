"""
dashboard_server.py
====================
A tiny live-status website for the human-eval ratings, no extra
dependencies (stdlib http.server + pandas, both already used elsewhere
in this project). Serves:

    GET /            -- an HTML page that polls /api/stats every 5s
    GET /api/stats   -- JSON: total/valid/excluded counts, per-annotator
                         breakdown, and the 5 most recent submissions

Usage:
    python dashboard_server.py [port]   # default port 8502
"""

import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

RESULTS_PATH = "/data/sriparna/swagata/iclr/human/human_evaluation/human_evaluation_results.csv"

PRE_FIX_IDS = {"1", "2", "12", "239492", "2204", "2001", "2002", "7564995"}
# guest_898a8ec3: 116 rows in 16.5 minutes, median 5s apart -- confirmed
# bot/spam. See human_eval_streamlit/human_eval.py's rate-limit fix.
SUSPECTED_BOT_IDS = {"guest_898a8ec3"}
EXCLUDE_IDS = PRE_FIX_IDS | SUSPECTED_BOT_IDS

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8502

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Human Eval -- Live Status</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117; color: #e5e7eb; margin: 0; padding: 24px;
  }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .sub { color: #9ca3af; font-size: 0.85rem; margin-bottom: 24px; }
  .cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
  .card {
    background: #1a1d27; border-radius: 10px; padding: 18px 22px;
    min-width: 140px; border: 1px solid #2a2e3a;
  }
  .card .n { font-size: 2.1rem; font-weight: 700; }
  .card .label { color: #9ca3af; font-size: 0.8rem; margin-top: 2px; }
  .good .n { color: #34d399; }
  .bad .n { color: #f87171; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 28px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #2a2e3a; font-size: 0.9rem; }
  th { color: #9ca3af; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #34d399; margin-right: 8px; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  #updated { color: #6b7280; font-size: 0.78rem; }
</style>
</head>
<body>
  <h1><span class="dot"></span>Human Evaluation -- Live Status</h1>
  <div class="sub">SD3 / PixArt / FLUX text-to-image ratings, auto-refreshing every 5s</div>

  <div class="cards" id="cards"></div>

  <h3>By annotator</h3>
  <table id="byAnnotator"><thead><tr><th>Annotator</th><th>Rows</th></tr></thead><tbody></tbody></table>

  <h3>Most recent</h3>
  <table id="recent"><thead><tr><th>Time</th><th>Annotator</th><th>Item</th><th>Score</th></tr></thead><tbody></tbody></table>

  <div id="updated"></div>

<script>
async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();

    document.getElementById('cards').innerHTML = `
      <div class="card good"><div class="n">${d.valid}</div><div class="label">Valid ratings</div></div>
      <div class="card"><div class="n">${d.annotators}</div><div class="label">Annotators</div></div>
      <div class="card bad"><div class="n">${d.excluded}</div><div class="label">Excluded (bot / pre-fix)</div></div>
      <div class="card"><div class="n">${d.total}</div><div class="label">Total rows</div></div>
    `;

    document.querySelector('#byAnnotator tbody').innerHTML = d.by_annotator
      .map(a => `<tr><td>${a.id}</td><td>${a.count}</td></tr>`).join('');

    document.querySelector('#recent tbody').innerHTML = d.recent
      .map(r => `<tr><td>${r.timestamp}</td><td>${r.annotator_id}</td><td>${r.item_id}</td><td>${r.human_overall}</td></tr>`).join('');

    document.getElementById('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('updated').textContent = 'connection lost, retrying...';
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def compute_stats():
    df = pd.read_csv(RESULTS_PATH)
    df["annotator_id"] = df["annotator_id"].astype(str)
    valid = df[~df["annotator_id"].isin(EXCLUDE_IDS)].sort_values("timestamp")

    by_annotator = (
        valid["annotator_id"].value_counts()
        .rename_axis("id").reset_index(name="count")
        .to_dict("records")
    )
    recent = valid.tail(5).iloc[::-1][
        ["timestamp", "annotator_id", "item_id", "human_overall"]
    ].to_dict("records")

    return {
        "total": int(len(df)),
        "valid": int(len(valid)),
        "excluded": int(len(df) - len(valid)),
        "annotators": int(valid["annotator_id"].nunique()),
        "by_annotator": by_annotator,
        "recent": recent,
        "generated_at": datetime.now().isoformat(),
    }


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # quiet -- avoid spamming stdout for every poll

    def do_GET(self):
        if self.path == "/api/stats":
            try:
                body = json.dumps(compute_stats()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"dashboard serving on :{PORT}")
    server.serve_forever()
