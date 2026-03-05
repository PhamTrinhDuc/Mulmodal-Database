import pandas as pd


df = pd.read_parquet("/home/ducpham/workspace/PTIT-CSDLDPT/data/0000.parquet")

print(df.columns)