"""
Lấy mỗi loài 2 file từ test.parquet và lưu ra thư mục data/samples/
"""

from pathlib import Path
import pandas as pd

PARQUET_PATH = "/home/ducpham/workspace/PTIT-CSDLDPT/data/test.parquet"
OUTPUT_DIR   = Path("/home/ducpham/workspace/PTIT-CSDLDPT/data/samples")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} rows, labels: {df['label'].nunique()} species")

saved = 0
for label, group in df.groupby("label"):
  for i, (_, row) in enumerate(group.head(2).iterrows()):
    audio_bytes = row["audio"]["bytes"]
    # Dùng tên loài làm tên file (thay space → underscore)
    safe_name = label.replace(" ", "_")
    out_path = OUTPUT_DIR / f"{safe_name}_{i+1}.wav"
    out_path.write_bytes(audio_bytes)
    saved += 1

print(f"Saved {saved} files to {OUTPUT_DIR}")
