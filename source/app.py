"""
app.py — Streamlit frontend cho hệ thống tìm kiếm tiếng chim hót.

Usage:
    streamlit run app.py
"""

import io

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bird Sound Retrieval",
    page_icon="🐦",
    layout="centered",
)

st.title("🐦 Tìm kiếm tiếng chim hót")
st.caption("Upload một file âm thanh để tìm các tiếng chim tương tự nhất trong CSDL.")

# ---------------------------------------------------------------------------
# Sidebar — tùy chọn tìm kiếm
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Tùy chọn tìm kiếm")

    top_k = st.slider("Số kết quả trả về (Top-K)", min_value=1, max_value=20, value=5)

    mode = st.selectbox("Kiểu index", options=["ivfflat", "hnsw"], index=0)

    show_intermediate = st.checkbox("Hiển thị kết quả trung gian", value=False)

    # Lấy danh sách loài từ API để filter (optional)
    filter_species = None
    try:
        resp = requests.get(f"{API_URL}/birds", timeout=3)
        if resp.ok:
            species_list = ["(Tất cả)"] + [b["species_name"] for b in resp.json()]
            selected = st.selectbox("Lọc theo loài (optional)", species_list)
            if selected != "(Tất cả)":
                filter_species = selected
    except requests.exceptions.ConnectionError:
        st.warning("Không kết nối được đến API để lấy danh sách loài.")

    st.divider()
    st.markdown("**API endpoint:** `http://localhost:8000`")

# ---------------------------------------------------------------------------
# Main — upload + search
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Chọn file âm thanh",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    label_visibility="collapsed",
)

if uploaded is not None:
    st.audio(uploaded, format=uploaded.type)
    st.divider()

    if st.button("🔍 Tìm kiếm", use_container_width=True, type="primary"):
        params = {"top_k": top_k, "mode": mode, "verbose": show_intermediate}
        if filter_species:
            params["filter_species"] = filter_species

        with st.spinner("Đang xử lý..."):
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    params=params,
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error("Không kết nối được đến API. Hãy chắc chắn `api.py` đang chạy.")
                st.stop()

        if not response.ok:
            st.error(f"API lỗi {response.status_code}: {response.json().get('detail', '')}")
            st.stop()

        resp_json = response.json()
        results = resp_json["results"]
        intermediate = resp_json.get("intermediate")

        # ── Kết quả trung gian ───────────────────────────────────────────────
        if show_intermediate and intermediate:
            st.subheader("🔬 Kết quả trung gian")

            with st.expander("▶ Bước 1 — Tiền xử lý (Preprocessing)", expanded=True):
                pre = intermediate["preprocessing"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Duration gốc", f"{pre['duration_before_s']:.3f}s")
                c2.metric("Duration sau trim", f"{pre['duration_after_s']:.3f}s")
                c3.metric("Sample rate", f"{pre['sample_rate']} Hz")
                st.caption("Waveform trước xử lý:")
                st.line_chart(pre["waveform_before"], height=120)
                st.caption("Waveform sau trim & normalize amplitude:")
                st.line_chart(pre["waveform_after"], height=120)

            with st.expander("▶ Bước 2 & 3 — Feature Extraction + Normalization"):
                raw = intermediate["features_raw"]
                norm = intermediate["features_normalized"]
                feat_df = pd.DataFrame({
                    "Feature": list(raw.keys()),
                    "Raw value": [raw[k] for k in raw],
                    "Normalized": [norm[k] for k in raw],
                })
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

            with st.expander("▶ Bước 4 — Embedding Vector (L2-normalized)"):
                emb = intermediate["embedding"]
                st.caption(f"||v|| = {intermediate['embedding_norm']} (phải = 1.0) — số chiều: {len(emb)}")
                st.bar_chart(pd.Series(emb, name="value"), height=200)

            st.divider()

        # ── Top-K kết quả ─────────────────────────────────────────────────────
        st.subheader(f"Top-{top_k} kết quả tương tự")

        for r in results:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**#{r['rank']} — {r['species_name']}**")
                    st.caption(f"Audio ID: {r['audio_id']}  |  distance: {r['distance']:.4f}")
                with col2:
                    st.metric("Similarity", f"{r['similarity'] * 100:.1f}%")
                st.progress(min(r["similarity"], 1.0))

                # Audio player: fetch từ API
                try:
                    audio_resp = requests.get(
                        f"{API_URL}/audio/{r['audio_id']}", timeout=10
                    )
                    if audio_resp.ok:
                        st.audio(io.BytesIO(audio_resp.content), format="audio/wav")
                except requests.exceptions.RequestException:
                    st.caption("⚠️ Không tải được audio.")

# streamlit run app.py