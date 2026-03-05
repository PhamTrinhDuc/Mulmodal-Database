"""
indexing.py — Offline pipeline: insert data và build vector index.

Supported index modes:
  - ivfflat : IVFFlat (inverted file, approximate, good for large datasets)
  - hnsw    : HNSW (Hierarchical Navigable Small World, fast query, higher memory)

Usage (CLI):
    python indexing.py --data 0000.parquet --mode ivfflat
    python indexing.py --data 0000.parquet --mode hnsw
"""

import argparse
import pickle

import pandas as pd
from psycopg2.extras import execute_values

from database import create_index, create_tables, drop_index, get_connection
from feature_extraction import build_vector, normalize_feature_columns, process_dataset

SKIP_META_COLS = {"label", "file_id", "duration_s", "sample_rate"}


def insert_records(conn, records: list[dict], feature_cols: list[str]):
    """
    Bulk-insert vào 4 bảng theo schema mới.

    records: list of dict với keys:
        file_id    (int)    — index trong parquet
        label      (str)    — tên loài chim (species_name)
        duration_s (float)  — độ dài audio sau preprocess (giây)
        sample_rate(int)    — sample rate sau preprocess
        embedding  (np.ndarray float32) — L2-normalized vector
        features   (dict)   — raw scalar values {col: float} trước normalize
    """
    # ── Step 1: Upsert birds (theo species_name) ─────────────────────────────
    unique_species = list({r["label"] for r in records})
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO birds (species_name) VALUES %s ON CONFLICT (species_name) DO NOTHING",
            [(s,) for s in unique_species],
        )
        cur.execute(
            "SELECT id, species_name FROM birds WHERE species_name = ANY(%s)",
            (unique_species,),
        )
        species_id_map = {row[1]: row[0] for row in cur.fetchall()}

    # ── Step 2: Insert audio_files, lấy lại id ───────────────────────────────
    audio_rows = [
        (
            species_id_map[r["label"]],
            f"parquet/{r['file_id']}",   # dùng để tra cứu lại row trong parquet
            r["sample_rate"],
            r["duration_s"],
        )
        for r in records
    ]
    with conn.cursor() as cur:
        audio_id_rows = execute_values(
            cur,
            """
            INSERT INTO audio_files (bird_id, file_path, sample_rate, duration_s)
            VALUES %s
            RETURNING id
            """,
            audio_rows,
            fetch=True,
        )
    audio_ids = [row[0] for row in audio_id_rows]

    # ── Step 3: Insert acoustic_features (raw scalar, trước normalize) ───────
    col_names = ", ".join(f'"{c}"' for c in feature_cols)
    value_placeholders = ", ".join(["%s"] * len(feature_cols))
    feature_rows = [
        [audio_id] + [float(r["features"][col]) for col in feature_cols]
        for audio_id, r in zip(audio_ids, records)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            f'INSERT INTO acoustic_features (audio_id, {col_names}) VALUES %s',
            feature_rows,
            template=f"(%s, {value_placeholders})",
        )

    # ── Step 4: Insert embeddings (L2-normalized vector) ─────────────────────
    embedding_rows = [(audio_id, r["embedding"]) for audio_id, r in zip(audio_ids, records)]
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO embeddings (audio_id, embedding) VALUES %s",
            embedding_rows,
            template="(%s, %s::vector)",
        )
        conn.commit()

    print(f"Inserted {len(records)} records into birds / audio_files / acoustic_features / embeddings.")

# ---------------------------------------------------------------------------
# End-to-end offline pipeline
# ---------------------------------------------------------------------------

def run_indexing(
    data_path: str,
    mode: str = "ivfflat",
    stats_path: str = "feature_stats.pkl",
    # IVFFlat
    lists: int = 40,
    # HNSW
    m: int = 16,
    ef_construction: int = 64,
):
    """
    Full offline pipeline:
      1. Load parquet
      2. Extract features
      3. Normalize (StandardScaler) + save stats
      4. Build L2-normalized vectors
      5. Connect DB → create table → drop old index → create new index → insert
    """
    # ----- 1. Load -----
    print(f"\n[1/5] Loading data from '{data_path}' ...")
    df = pd.read_parquet(data_path)
    print(f"      Loaded {len(df)} records.")

    # ----- 2. Extract features -----
    print("[2/5] Extracting features ...")
    feature_df = process_dataset(df)
    print(f"      Extracted {len(feature_df)} records, {len(feature_df.columns)} columns.")

    # ----- 3. Normalize -----
    print("[3/5] Normalizing feature columns (StandardScaler) ...")
    feature_df_norm, stats = normalize_feature_columns(feature_df)
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"      Stats saved to '{stats_path}'.")

    # ----- 4. Build vectors -----
    print("[4/5] Building L2-normalized embedding vectors ...")
    feature_cols = [c for c in feature_df_norm.columns if c not in SKIP_META_COLS]

    records = []
    for (_, raw_row), (_, norm_row) in zip(feature_df.iterrows(), feature_df_norm.iterrows()):
        vec = build_vector(norm_row, feature_cols)
        records.append({
            "file_id":     int(raw_row["file_id"]),
            "label":       raw_row["label"],
            "duration_s":  float(raw_row.get("duration_s", 0.0)),
            "sample_rate": int(raw_row.get("sample_rate", 22050)),
            "embedding":   vec,
            "features":    {col: float(raw_row[col]) for col in feature_cols},  # raw, trước normalize
        })

    print(f"      Vector dim    : {records[0]['embedding'].shape[0]}")
    print(f"      Feature cols  : {len(feature_cols)}")
    print(f"      Sample duration_s: {records[0]['duration_s']:.3f}s")

    # ----- 5. Database -----
    print(f"[5/5] Connecting to database and indexing (mode={mode}) ...")
    conn = get_connection()
    try:
        create_tables(conn, feature_cols)
        drop_index(conn)
        insert_records(conn, records, feature_cols)
        create_index(conn, mode=mode, lists=lists, m=m, ef_construction=ef_construction)
    finally:
        conn.close()

    print("\nIndexing complete.")

if __name__ == "__main__":

    DATA_PATH = "/home/ducpham/workspace/PTIT-CSDLDPT/data/index.parquet"
    STATS_PATH = "/home/ducpham/workspace/PTIT-CSDLDPT/data/feature_stats.pkl"

    parser = argparse.ArgumentParser(description="Offline indexing pipeline for bird_sounds.")
    parser.add_argument("--data", default=DATA_PATH, help="Path to parquet file")
    parser.add_argument(
        "--mode",
        default="ivfflat",
        choices=["ivfflat", "hnsw"],
        help="Vector index type (default: ivfflat)",
    )
    parser.add_argument("--stats", default=STATS_PATH, help="Output path for feature stats")
    # IVFFlat
    parser.add_argument("--lists", type=int, default=40, help="[IVFFlat] number of lists (clusters)")
    # HNSW
    parser.add_argument("--m", type=int, default=16, help="[HNSW] max connections per node")
    parser.add_argument("--ef-construction", type=int, default=64, help="[HNSW] ef_construction")

    args = parser.parse_args()

    run_indexing(
        data_path=args.data,
        mode=args.mode,
        stats_path=args.stats,
        lists=args.lists,
        m=args.m,
        ef_construction=args.ef_construction,
    )
