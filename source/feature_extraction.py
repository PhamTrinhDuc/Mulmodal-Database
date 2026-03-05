import librosa
import numpy as np
import pandas as pd
import io

TARGET_SR = 22050  # Chuẩn hóa sample rate cho toàn bộ dataset

def preprocess_audio(audio_data, sr):
    """
    Tiền xử lý audio trước khi trích xuất đặc trưng:
    1. Resample về TARGET_SR để đảm bảo tính nhất quán
    2. Trim silence ở đầu/cuối
    3. Normalize amplitude về [-1, 1]
    """
    # 1. Resample nếu sample rate khác TARGET_SR
    if sr != TARGET_SR:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)

    # 2. Trim silence (top_db=20: ngưỡng cắt lặng, đủ nhạy cho tiếng chim)
    audio_data, _ = librosa.effects.trim(audio_data, top_db=20)

    # 3. Normalize amplitude về [-1, 1]
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    return audio_data, TARGET_SR


def extract_features(audio_data, sr):
    """
    Trích xuất đặc trưng từ audio.
    Giả định audio đã được tiền xử lý (preprocess_audio).

    Tổng số chiều vector: 20+20+20+20 + 1+1+1+1+1+1 + 12 + 1+1 = 101 chiều
    """
    features = {}

    if audio_data.size == 0:
        print("Warning: Received empty audio data for feature extraction.")
        return {}

    # 1. MFCC (mean + std) — đặc trưng quan trọng nhất
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)   # 20 chiều
    features['mfcc_std'] = np.std(mfccs, axis=1)     # 20 chiều

    # MFCC delta — thay đổi theo thời gian (âm điệu chim nhanh/chậm)
    mfcc_delta = librosa.feature.delta(mfccs)
    features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)  # 20 chiều

    # MFCC delta-delta — gia tốc thay đổi (pattern phức tạp hơn)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    features['mfcc_delta2_mean'] = np.mean(mfcc_delta2, axis=1)  # 20 chiều

    # 2. Spectral Features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)

    # Spectral contrast — phân biệt dải tần cao/thấp, rất hiệu quả với tiếng chim
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)  # 7 chiều

    # Spectral flatness — phân biệt âm có tính nhạc (tiếng chim) vs noise
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)

    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # 4. Chroma Features (12 chiều)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1)

    # 5. RMS Energy
    rms = librosa.feature.rms(y=audio_data)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    # NOTE: Bỏ tempo — không ổn định với audio ngắn/tiếng chim, hay ra NaN

    return features


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def flatten_features(raw_features: dict) -> dict:
    """
    Flatten array-valued features thành scalar columns.
    Ví dụ: {'mfcc_mean': np.array([v1, v2, ...])} → {'mfcc_mean_1': v1, 'mfcc_mean_2': v2, ...}

    Dùng chung cho cả offline (process_dataset) và online (process_single_audio).
    """
    flat = {}
    for key, value in raw_features.items():
        if isinstance(value, np.ndarray):
            for i, val in enumerate(value.tolist()):
                flat[f"{key}_{i + 1}"] = val
        else:
            flat[key] = value
    return flat


# ---------------------------------------------------------------------------
# Offline — batch dataset
# ---------------------------------------------------------------------------

def process_dataset(df) -> pd.DataFrame:
    """
    [OFFLINE] Trích xuất feature cho toàn bộ DataFrame parquet.

    Input : DataFrame với cột 'audio' (bytes) và 'label'
    Output: DataFrame feature phẳng (scalar columns)
    """
    feature_list = []
    for idx, row in df.iterrows():
        audio_bytes = row['audio']['bytes']
        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            audio_data, sr = preprocess_audio(audio_data, sr)
            raw_features = extract_features(audio_data, sr)

            if raw_features:
                current_features = {
                    'label': row['label'],
                    'file_id': idx,
                    'duration_s': round(len(audio_data) / sr, 4),  # độ dài thực sau trim
                    'sample_rate': sr,                              # sr sau preprocess (=TARGET_SR)
                }
                current_features.update(flatten_features(raw_features))
                feature_list.append(current_features)
            else:
                print(f"Skipping index {idx} due to empty audio data or no features extracted.")

        except Exception as e:
            print(f"Error processing audio at index {idx} (label: {row['label']}): {e}")
            continue

    return pd.DataFrame(feature_list)


def normalize_feature_columns(df_features):
    """
    Chuẩn hóa các cột feature về mean=0, std=1 (StandardScaler).
    Cần thiết trước khi lưu vector vào pgvector để cosine/L2 similarity có ý nghĩa.
    Trả về: df đã normalize + thống kê (mean, std) để dùng khi query.
    """
    skip_cols = ['label', 'file_id', 'duration_s', 'sample_rate']
    feature_cols = [c for c in df_features.columns if c not in skip_cols]

    stats = {}
    df_norm = df_features.copy()
    for col in feature_cols:
        mean = df_features[col].mean()
        std = df_features[col].std()
        std = std if std > 0 else 1.0  # tránh chia 0
        df_norm[col] = (df_features[col] - mean) / std
        stats[col] = {'mean': mean, 'std': std}

    return df_norm, stats


def build_vector(row, feature_cols):
    """
    Flatten một hàng feature thành 1D numpy array để lưu vào pgvector.
    Pipeline:
        1. Feature scaling (StandardScaler) đã làm trước đó — cân bằng scale giữa các features
        2. Concat tất cả features thành 1 vector
        3. L2 normalize vector về độ dài = 1 — để cosine similarity chính xác nhất
    """
    vec = np.array([row[col] for col in feature_cols], dtype=np.float32)

    # L2 normalize: đưa vector về unit length (||vec|| = 1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


# ---------------------------------------------------------------------------
# Online — single audio file
# ---------------------------------------------------------------------------

def process_single_audio(audio_bytes: bytes, stats: dict, verbose: bool = False):
    """
    [ONLINE] Trích xuất embedding từ một file âm thanh (bytes) dùng để query.

    Args:
        audio_bytes : raw audio file content (bytes)
        stats       : dict đọc từ feature_stats.pkl
        verbose     : nếu True, trả về dict với kết quả trung gian từng bước;
                      nếu False (mặc định), trả về np.ndarray embedding.
    """
    _MAX_PTS = 2000  # giới hạn điểm waveform để JSON không quá lớn

    # ── 1. Load raw ──────────────────────────────────────────────────────────
    audio_raw, sr_raw = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    duration_before = len(audio_raw) / sr_raw

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    audio_data, sr = preprocess_audio(audio_raw, sr_raw)
    duration_after = len(audio_data) / sr

    # ── 3. Feature extraction ────────────────────────────────────────────────
    raw_features = extract_features(audio_data, sr)
    if not raw_features:
        raise ValueError("Feature extraction returned empty result for the given audio.")
    flat = flatten_features(raw_features)

    # ── 4. Normalize (StandardScaler) ────────────────────────────────────────
    feature_cols = [c for c in stats.keys() if c in flat]
    flat_normalized = {}
    for col in feature_cols:
        mean = stats[col]["mean"]
        std = stats[col]["std"] if stats[col]["std"] > 0 else 1.0
        flat_normalized[col] = (flat[col] - mean) / std

    # ── 5. Build L2-normalized embedding ─────────────────────────────────────
    embedding = build_vector(flat_normalized, feature_cols)

    if not verbose:
        return embedding

    # ── Verbose: trả về dict kết quả trung gian ──────────────────────────────
    def _down(arr):
        step = max(1, len(arr) // _MAX_PTS)
        return arr[::step].tolist()

    return {
        "embedding": embedding.tolist(),
        "steps": {
            "preprocessing": {
                "duration_before_s": round(duration_before, 4),
                "duration_after_s": round(duration_after, 4),
                "sample_rate": sr,
                "waveform_before": _down(audio_raw),
                "waveform_after": _down(audio_data),
            },
            "features_raw": {k: round(float(flat[k]), 6) for k in feature_cols},
            "features_normalized": {k: round(float(flat_normalized[k]), 6) for k in feature_cols},
            "embedding_norm": round(float(np.linalg.norm(embedding)), 6),
            "embedding": embedding.tolist(),
        },
    }


if __name__ == "__main__":
    import pickle
    data_path = '/home/ducpham/workspace/PTIT-CSDLDPT/0000.parquet'

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records")

    # ----------------------------------------------------------------
    # OFFLINE: xử lý batch → normalize → build vectors → lưu stats
    # ----------------------------------------------------------------
    print("\n--- OFFLINE mode ---")
    feature_df = process_dataset(df.iloc[0:10])
    print(f"Extracted features: {len(feature_df)} records, {len(feature_df.columns)} columns")

    feature_df_norm, stats = normalize_feature_columns(feature_df)
    with open("feature_stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    print("Saved feature stats to feature_stats.pkl")

    skip_cols = ['label', 'file_id']
    feature_cols = [c for c in feature_df_norm.columns if c not in skip_cols]
    sample_vector = build_vector(feature_df_norm.iloc[0], feature_cols)
    print(f"Offline vector shape : {sample_vector.shape}")  # Kỳ vọng: (108,)

    # ----------------------------------------------------------------
    # ONLINE: 1 file âm thanh → dùng stats đã lưu → query vector
    # ----------------------------------------------------------------
    print("\n--- ONLINE mode ---")
    with open("feature_stats.pkl", "rb") as f:
        loaded_stats = pickle.load(f)

    sample_audio_bytes = df.iloc[0]['audio']['bytes']
    online_vector = process_single_audio(sample_audio_bytes, loaded_stats)
    print(f"Online  vector shape : {online_vector.shape}")  # Kỳ vọng: (108,)
    print(f"L2 norm (should ≈ 1): {np.linalg.norm(online_vector):.6f}")