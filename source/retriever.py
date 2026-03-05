"""
retriever.py — Online retrieval pipeline: query top-K similar embeddings.

Supported index modes (must match the index built by indexing.py):
  - ivfflat : uses SET ivfflat.probes  (controls recall vs. speed trade-off)
  - hnsw    : uses SET hnsw.ef_search  (controls recall vs. speed trade-off)

Usage (CLI):
    python retriever.py --file-id 0 --mode ivfflat --top-k 5
    python retriever.py --file-id 0 --mode hnsw --top-k 5 --ef-search 100
    python retriever.py --file-id 0 --mode ivfflat --label "Hirundo rustica"
"""

import argparse
import pickle

import numpy as np
import pandas as pd

from database import get_connection
from feature_extraction import process_single_audio

def search_similar(
    conn,
    query_vector: np.ndarray,
    top_k: int = 5,
    filter_label: str = None,
    mode: str = "ivfflat",
    # IVFFlat runtime param
    probes: int = 20,
    # HNSW runtime param
    ef_search: int = 64,
) -> list[dict]:
    """
    Tìm top_k embeddings giống nhất với query_vector.

    Cosine distance (<=>): range [0, 2], càng nhỏ càng giống.
    Cosine similarity = 1 - distance: range [-1, 1], càng lớn càng giống.

    Args:
        conn         : psycopg2 connection (đã register_vector)
        query_vector : np.ndarray float32, đã qua StandardScaler + L2 normalize
        top_k        : số kết quả trả về
        filter_label : giới hạn tìm kiếm trong một loài cụ thể (optional)
        mode         : "ivfflat" hoặc "hnsw" — phải khớp với index đã build
        probes       : [IVFFlat] số lists được scan; cao hơn → recall tốt hơn / chậm hơn
        ef_search    : [HNSW] beam size lúc query; cao hơn → recall tốt hơn / chậm hơn

    Returns:
        list of dict: id, file_id, label, distance, similarity
    """
    mode = mode.lower()
    if mode not in ("ivfflat", "hnsw"):
        raise ValueError(f"Unsupported mode: '{mode}'. Choose 'ivfflat' or 'hnsw'.")

    with conn.cursor() as cur:
        # --- Set runtime search param phù hợp với index type ---
        if mode == "ivfflat":
            cur.execute("SET ivfflat.probes = %s;", (probes,))
        else:  # hnsw
            cur.execute("SET hnsw.ef_search = %s;", (ef_search,))

        # --- Run ANN query ---
        if filter_label:
            cur.execute(
                """
                SELECT e.id, af.id, b.species_name,
                       e.embedding <=> %s::vector AS distance
                FROM embeddings e
                JOIN audio_files af ON af.id = e.audio_id
                JOIN birds b ON b.id = af.bird_id
                WHERE b.species_name = %s
                ORDER BY distance
                LIMIT %s;
                """,
                (query_vector, filter_label, top_k),
            )
        else:
            cur.execute(
                """
                SELECT e.id, af.id, b.species_name,
                       e.embedding <=> %s::vector AS distance
                FROM embeddings e
                JOIN audio_files af ON af.id = e.audio_id
                JOIN birds b ON b.id = af.bird_id
                ORDER BY distance
                LIMIT %s;
                """,
                (query_vector, top_k),
            )

        rows = cur.fetchall()

    return [
        {
            "embedding_id": r[0],
            "audio_id": r[1],  # audio_files.id
            "label": r[2],
            "distance": round(float(r[3]), 6),
            "similarity": round(1 - float(r[3]), 6),
        }
        for r in rows
    ]

def run_query(
    query_vector: np.ndarray,
    top_k: int = 5,
    filter_label: str = None,
    mode: str = "ivfflat",
    probes: int = 20,
    ef_search: int = 64,
) -> list[dict]:
    """
    Convenience wrapper: mở connection, query, đóng connection.

    Returns: list of result dicts (id, file_id, label, distance, similarity).
    """
    conn = get_connection()
    try:
        results = search_similar(
            conn,
            query_vector,
            top_k=top_k,
            filter_label=filter_label,
            mode=mode,
            probes=probes,
            ef_search=ef_search,
        )
    finally:
        conn.close()
    return results

if __name__ == "__main__":
    TEST_PATH = "/home/ducpham/workspace/PTIT-CSDLDPT/data/test.parquet"
    STATS_PATH = "/home/ducpham/workspace/PTIT-CSDLDPT/data/feature_stats.pkl"

    parser = argparse.ArgumentParser(description="Online retrieval for bird_sounds.")
    parser.add_argument("--data", default=TEST_PATH, help="Path to parquet file (for query sample)")
    parser.add_argument("--file-id", type=int, default=0, help="Row index in parquet to use as query")
    parser.add_argument("--stats", default=STATS_PATH, help="Path to feature_stats.pkl")
    parser.add_argument(
        "--mode",
        default="ivfflat",
        choices=["ivfflat", "hnsw"],
        help="Index mode to use at query time (default: ivfflat)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--label", default=None, help="Filter results to this label (optional)")
    # IVFFlat
    parser.add_argument("--probes", type=int, default=20, help="[IVFFlat] ivfflat.probes")
    # HNSW
    parser.add_argument("--ef-search", type=int, default=64, help="[HNSW] hnsw.ef_search")

    args = parser.parse_args()

    # Load stats
    with open(args.stats, "rb") as f:
        stats = pickle.load(f)

    # Build query vector from chosen row in parquet
    df = pd.read_parquet(args.data)
    row = df.iloc[args.file_id]
    audio_bytes = row["audio"]["bytes"]
    query_vec = process_single_audio(audio_bytes, stats)  # [ONLINE] feature_extraction pipeline

    print(f"\nQuery  file_id={args.file_id}  label={row['label']}")
    print(f"Mode   : {args.mode}")
    print(f"Top-K  : {args.top_k}")
    if args.label:
        print(f"Filter : label='{args.label}'")
    print()

    results = run_query(
        query_vec,
        top_k=args.top_k,
        filter_label=args.label,
        mode=args.mode,
        probes=args.probes,
        ef_search=args.ef_search,
    )

    print(f"{'rank':>4}  {'file_id':>7}  {'label':<25}  {'similarity':>10}  {'distance':>10}")
    print("-" * 65)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:>4}  {r['file_id']:>7}  {r['label']:<25}  {r['similarity']:>10.4f}  {r['distance']:>10.4f}"
        )
