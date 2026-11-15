import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE / "src"))
from features import extract_features

MODEL_FILES = {
    "Random Forest": BASE / "models" / "model_rf.joblib",
    "XGBoost": BASE / "models" / "model_xgb.joblib",
}

st.set_page_config(page_title="Detektor Phishing URL", page_icon="🛡️")
st.title("🛡️ Detektor Phishing URL")
st.write("Masukkan sebuah URL untuk dicek apakah **phishing** atau **aman**.")
st.caption("Catatan: aplikasi hanya membaca teks URL, tidak membuka halamannya.")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def prediksi_dummy(fitur):
    """Dipakai sementara kalau model asli belum ada."""
    f = fitur.iloc[0]
    skor = 0.1
    skor += 0.3 * f["has_at"]
    skor += 0.3 * f["has_ip"]
    skor += 0.1 * (f["num_hyphens"] > 2)
    skor += 0.1 * (f["url_length"] > 75)
    skor += 0.1 * (f["is_https"] == 0)
    return min(skor, 0.99)


pilihan = st.selectbox("Pilih model", list(MODEL_FILES.keys()))
url = st.text_input("URL", placeholder="https://contoh.com/login")

if st.button("Cek"):
    if not url.strip():
        st.warning("Isi URL dulu ya.")
    else:
        fitur = pd.DataFrame([extract_features(url)])
        path = MODEL_FILES[pilihan]

        if path.exists():
            paket = load_model(path)
            fitur = fitur[paket["columns"]]
            prob = paket["model"].predict_proba(fitur)[0][1]
        else:
            st.info("Model belum ada, memakai hasil sementara (dummy).")
            prob = prediksi_dummy(fitur)

        if prob >= 0.5:
            st.error(f"⚠️ PHISHING (peluang phishing {prob:.1%})")
        else:
            st.success(f"✅ AMAN (peluang phishing {prob:.1%})")

        with st.expander("Lihat fitur yang dihitung"):
            st.dataframe(fitur.T, use_container_width=True)