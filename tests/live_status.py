import pandas as pd
from datetime import datetime

PATH = "/data/sriparna/swagata/iclr/human/human_evaluation/human_evaluation_results.csv"
PRE_FIX_IDS = {"1", "2", "12", "239492", "2204", "2001", "2002", "7564995"}
# guest_898a8ec3: 116 rows in 16.5 minutes, median 5s apart -- confirmed
# bot/spam (see human_eval_streamlit/human_eval.py's rate-limit fix).
SUSPECTED_BOT_IDS = {"guest_898a8ec3"}
EXCLUDE_IDS = PRE_FIX_IDS | SUSPECTED_BOT_IDS

df = pd.read_csv(PATH)
df["annotator_id"] = df["annotator_id"].astype(str)
valid = df[~df["annotator_id"].isin(EXCLUDE_IDS)]

print(f"=== live human-eval status @ {datetime.now().strftime('%H:%M:%S')} ===")
print(f"total rows: {len(df)}   valid (post-fix): {len(valid)}   stale (pre-fix, excluded): {len(df) - len(valid)}")
print(f"valid annotators: {valid['annotator_id'].nunique()}")
print()
print(valid["annotator_id"].value_counts().rename("rows").to_string())
print()
last = valid.sort_values("timestamp").tail(3)
print("most recent submissions:")
print(last[["timestamp", "annotator_id", "item_id", "human_overall"]].to_string(index=False))
