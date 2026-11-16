"""
Detektor Phishing URL: Perbandingan Random Forest dan XGBoost
Final project Keamanan Informasi, Universitas Siliwangi.

Aplikasi hanya membaca teks URL (analisis leksikal pasif). Aplikasi tidak
membuka, mengunduh, atau meminta isi URL yang dimasukkan.
"""

from datetime import datetime
from html import escape
from pathlib import Path
import sys
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st
import tldextract

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE / "src"))

try:
    from features import extract_features
except ImportError:
    st.error("Modul 'features.py' tidak ditemukan di folder 'src'. Periksa struktur folder proyek.")
    st.stop()

_ext_offline = tldextract.TLDExtract(suffix_list_urls=())

MODEL_FILES = {
    "Random Forest": BASE / "models" / "model_rf.joblib",
    "XGBoost": BASE / "models" / "model_xgb.joblib",
}

FITUR_INFO = {
    "url_length": ("Panjang URL", "Total karakter seluruh URL"),
    "host_length": ("Panjang Host", "Jumlah karakter nama host"),
    "path_length": ("Panjang Path", "Jumlah karakter jalur setelah host"),
    "num_dots": ("Jumlah Titik", "Frekuensi karakter titik"),
    "num_hyphens": ("Jumlah Tanda Hubung", "Frekuensi tanda hubung, sering muncul pada domain tiruan"),
    "num_underscores": ("Jumlah Underscore", "Frekuensi garis bawah"),
    "num_slashes": ("Jumlah Garis Miring", "Frekuensi pemisah segmen jalur"),
    "num_digits": ("Jumlah Digit", "Total karakter angka"),
    "digit_ratio": ("Rasio Digit", "Proporsi angka terhadap panjang URL"),
    "num_special": ("Karakter Khusus", "Karakter selain huruf, angka, dan garis bawah"),
    "has_at": ("Mengandung @", "1 jika ada simbol @ yang bisa menyamarkan tujuan"),
    "has_ip": ("Host Alamat IP", "1 jika host berupa alamat IP, bukan nama domain"),
    "num_subdomains": ("Jumlah Subdomain", "Tingkat subdomain pada host"),
    "is_https": ("Protokol HTTPS", "1 jika memakai https://"),
    "has_query": ("Ada Query", "1 jika memuat parameter setelah tanda tanya"),
    "num_params": ("Jumlah Parameter", "Banyaknya parameter yang dipisah tanda &"),
}

PRESET = {
    "Situs kampus UNSIL": "https://unsil.ac.id",
    "Mesin pencari Google": "https://www.google.com",
    "Alamat IP mentah": "http://192.168.1.1/admin",
    "Tiruan bank (HTTP)": "http://login-bca-klik-verifikasi.com",
    "Hosting gratis": "https://verifynew--verifynewnew.repl.co",
    "Contoh false positive": "https://www.bmvc2020-conference.com",
}

st.set_page_config(page_title="Detektor Phishing URL", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# Tema: seluruh warna dikelola lewat variabel CSS agar kontras terjaga
# ----------------------------------------------------------------------------
TEMA = {
    "Mode Gelap": {
        "scheme": "dark",
        "bg": "radial-gradient(circle at 15% 10%, #10294a 0%, #0a192f 45%, #050f1e 100%)",
        "card": "rgba(10, 25, 47, 0.92)", "box": "rgba(15, 30, 55, 0.95)",
        "border": "rgba(56, 189, 248, 0.32)", "text": "#f1f5f9", "text2": "#a8b5c7",
        "input": "#08162b", "popup": "#0a192f", "dot": "rgba(56, 189, 248, 0.10)",
        "accent": "#38bdf8", "ok": "#22c55e", "warn": "#f59e0b",
        "danger": "#f87171", "purple": "#c084fc",
        "btn": "linear-gradient(135deg, #0369a1 0%, #0284c7 100%)",
        "nav_bg": "#38bdf8", "nav_text": "#04101f", "chip": "rgba(56, 189, 248, 0.14)",
    },
    "Mode Terang": {
        "scheme": "light",
        "bg": "linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 40%, #e2e8f0 100%)",
        "card": "rgba(255, 255, 255, 0.96)", "box": "#f1f5f9",
        "border": "rgba(100, 116, 139, 0.45)", "text": "#0f172a", "text2": "#334155",
        "input": "#ffffff", "popup": "#ffffff", "dot": "rgba(3, 105, 161, 0.08)",
        "accent": "#0369a1", "ok": "#15803d", "warn": "#b45309",
        "danger": "#b91c1c", "purple": "#7e22ce",
        "btn": "linear-gradient(135deg, #075985 0%, #0369a1 100%)",
        "nav_bg": "#0369a1", "nav_text": "#ffffff", "chip": "rgba(3, 105, 161, 0.10)",
    },
}

CSS_STATIS = """
#MainMenu, footer { visibility: hidden; }

/* Header diberi latar solid sesuai tema, agar indikator Running selalu terbaca */
[data-testid="stHeader"] {
    background: var(--popup) !important;
    border-bottom: 1px solid var(--border);
}
[data-testid="stHeader"] *,
[data-testid="stToolbar"] *,
[data-testid="stStatusWidget"], [data-testid="stStatusWidget"] *,
[class*="StatusWidget"], [class*="StatusWidget"] * {
    color: var(--text) !important;
    background-color: transparent !important;
}
[data-testid="stHeader"] svg, [data-testid="stStatusWidget"] svg,
[class*="StatusWidget"] svg { fill: var(--text) !important; }
[data-testid="stStatusWidget"] code, [class*="StatusWidget"] code {
    background-color: var(--box) !important; color: var(--accent) !important;
}

/* Sembunyikan tombol Deploy */
[data-testid="stAppDeployButton"], .stDeployButton, .stAppDeployButton { display: none !important; }

:root { color-scheme: var(--scheme); }

.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    background-attachment: fixed;
    color: var(--text);
}
[data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; inset: 0; pointer-events: none;
    background-image: radial-gradient(var(--dot) 1px, transparent 1px);
    background-size: 26px 26px;
}
[data-testid="stSidebar"] {
    background: var(--card); border-right: 1px solid var(--border);
}

/* Teks Streamlit bawaan. Elemen HTML kustom mewarisi dan boleh punya warna sendiri */
[data-testid="stMarkdownContainer"] { color: var(--text); }
[data-testid="stMarkdownContainer"] > p,
[data-testid="stMarkdownContainer"] > ul > li,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
    color: var(--text);
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--text2); }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {
    color: var(--text); font-weight: 600;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stAlert"], [data-testid="stAlert"] * { color: var(--text); }
[data-testid="stHeader"] svg, [data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stSidebarCollapsedControl"] svg {
    fill: var(--text); color: var(--text);
}

/* Input dan dropdown */
input[type="text"] {
    background-color: var(--input) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: "JetBrains Mono", Consolas, monospace !important; font-size: 13.5px !important;
}
input[type="text"]:focus { border-color: var(--accent) !important; }
input::placeholder { color: var(--text2) !important; opacity: 1 !important; }
[data-baseweb="select"] > div {
    background-color: var(--input) !important; border-color: var(--border) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] * { color: var(--text) !important; }
[data-baseweb="select"] svg { fill: var(--text) !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background-color: var(--popup) !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: var(--text) !important; }
[data-baseweb="popover"] li:hover { background-color: var(--chip) !important; }

/* Tombol */
button[kind="primary"] {
    background: var(--btn) !important; border: 1px solid var(--accent) !important;
    border-radius: 8px !important; font-weight: 700 !important;
}
button[kind="primary"], button[kind="primary"] p { color: #ffffff !important; }
button[kind="secondary"] {
    background-color: var(--chip) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
button[kind="secondary"], button[kind="secondary"] p {
    color: var(--text) !important; font-size: 12.5px; font-weight: 600;
}
button[kind="secondary"]:hover { border-color: var(--accent) !important; }

/* Navigasi utama */
.stRadio div[role="radiogroup"] {
    background: var(--box); border: 1px solid var(--border);
    border-radius: 10px; padding: 4px; gap: 6px;
}
.stRadio label[data-baseweb="radio"] { border-radius: 7px; padding: 6px 14px; }
.stRadio label[data-baseweb="radio"] > div:first-child { display: none; }
.stRadio label[data-baseweb="radio"] p { font-weight: 700; font-size: 13px; }
.stRadio label[data-baseweb="radio"]:has(input:checked) { background: var(--nav_bg); }
.stRadio label[data-baseweb="radio"]:has(input:checked) p { color: var(--nav_text) !important; }

/* Wadah panel */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card); border: 1px solid var(--border) !important;
    border-radius: 12px; box-shadow: 0 4px 18px rgba(0, 0, 0, 0.18);
}

code, [data-testid="stMarkdownContainer"] code {
    background: var(--box) !important; color: var(--accent) !important;
    border: 1px solid var(--border); border-radius: 4px; padding: 1px 6px;
    font-family: "JetBrains Mono", Consolas, monospace;
}

/* Komponen kustom */
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
        padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 4px 18px rgba(0,0,0,0.18); }
.mono { font-family: "JetBrains Mono", Consolas, monospace; }
.kicker { font-size: 11px; font-weight: 800; letter-spacing: 0.8px; color: var(--accent);
          font-family: "JetBrains Mono", Consolas, monospace; margin-bottom: 4px; }
.judul { font-size: 18px; font-weight: 800; margin-bottom: 6px; }
.sub { font-size: 13px; color: var(--text2); line-height: 1.6; }
.hdr { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
.hdr-title { font-size: 20px; font-weight: 800; }
.pill { background: var(--box); border: 1px solid var(--border); border-radius: 6px;
        padding: 4px 10px; font-size: 11px; font-weight: 700; color: var(--accent);
        font-family: "JetBrains Mono", Consolas, monospace; margin-left: 6px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 11px;
         font-weight: 800; letter-spacing: 0.5px; border: 1px solid currentColor;
         font-family: "JetBrains Mono", Consolas, monospace; }
.b-safe { color: var(--ok); background: color-mix(in srgb, var(--ok) 14%, transparent); }
.b-warn { color: var(--warn); background: color-mix(in srgb, var(--warn) 14%, transparent); }
.b-danger { color: var(--danger); background: color-mix(in srgb, var(--danger) 14%, transparent); }
.b-info { color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }

.gauge { background: var(--box); border: 1px solid var(--border); border-radius: 10px;
         padding: 16px 18px; margin-bottom: 14px; }
.gauge-track { background: rgba(148,163,184,0.30); border-radius: 8px; height: 14px; margin: 10px 0 6px; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 8px; }
.g-safe { background: var(--ok); } .g-warn { background: var(--warn); } .g-danger { background: var(--danger); }
.gauge-scale { display: flex; justify-content: space-between; font-size: 10px; color: var(--text2);
               font-family: "JetBrains Mono", Consolas, monospace; }

.tile { background: var(--box); border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px; }
.tile:hover { border-color: var(--accent); }
.tile-label { font-size: 10.5px; font-weight: 700; color: var(--text2); text-transform: uppercase; margin-bottom: 2px; }
.tile-val { font-size: 15px; font-weight: 800; font-family: "JetBrains Mono", Consolas, monospace; }
.tile-note { font-size: 11px; color: var(--text2); margin-top: 3px; line-height: 1.4; }
.grid16 { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin-top: 8px; }
.t-ok { color: var(--ok); font-size: 10px; font-weight: 700; }
.t-warn { color: var(--warn); font-size: 10px; font-weight: 700; }
.t-danger { color: var(--danger); font-size: 10px; font-weight: 700; }

.dissect { background: var(--box); border: 1px solid var(--border); border-radius: 8px;
           padding: 12px 14px; margin: 12px 0; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; font-family: "JetBrains Mono", Consolas, monospace; font-size: 13px; }
.chip { padding: 3px 9px; border-radius: 5px; border: 1px solid currentColor; }
.chip small { font-size: 9.5px; font-weight: 800; opacity: 0.85; margin-right: 5px; }
.c-scheme { color: var(--accent); } .c-sub { color: var(--purple); } .c-dom { color: var(--ok); font-weight: 700; }
.c-tld { color: var(--warn); } .c-path { color: var(--text2); } .c-query { color: var(--danger); }

.finding { background: var(--box); border: 1px solid var(--border); border-left: 4px solid var(--warn);
           border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.finding.high { border-left-color: var(--danger); }
.finding-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.finding-body { font-size: 12px; color: var(--text2); line-height: 1.45; margin-top: 3px; }

.note { background: color-mix(in srgb, var(--accent) 12%, transparent); border: 1px solid var(--border);
        border-left: 4px solid var(--accent); border-radius: 8px; padding: 11px 15px; margin: 12px 0;
        font-size: 13px; line-height: 1.6; }
.note.warn { border-left-color: var(--warn); background: color-mix(in srgb, var(--warn) 12%, transparent); }

.cm { width: 100%; border-collapse: separate; border-spacing: 6px; }
.cm td { background: var(--box); border: 1px solid var(--border); border-radius: 8px; padding: 12px;
         text-align: center; font-family: "JetBrains Mono", Consolas, monospace; font-size: 12px; font-weight: 700; }
.cm td small { display: block; font-size: 10px; font-weight: 500; color: var(--text2); }
.cm td.good { color: var(--ok); background: color-mix(in srgb, var(--ok) 14%, transparent); font-size: 15px; }
.cm td.bad { color: var(--danger); background: color-mix(in srgb, var(--danger) 14%, transparent); font-size: 15px; }

.fi { margin-bottom: 10px; }
.fi-head { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px; }
.fi-track { background: rgba(148,163,184,0.30); border-radius: 4px; height: 8px; overflow: hidden; }
.fi-fill { height: 100%; background: var(--accent); border-radius: 4px; }

.standby { text-align: center; padding: 50px 16px; }
.ring { width: 76px; height: 76px; border: 2px dashed var(--accent); border-radius: 50%;
        margin: 0 auto 18px; display: flex; align-items: center; justify-content: center;
        color: var(--accent); font-weight: 800; font-size: 11px; }
.step-num { font-size: 10.5px; font-weight: 800; color: var(--accent);
            font-family: "JetBrains Mono", Consolas, monospace; }

/* Tampilan responsif: tablet dan HP */
.mono, code, .chips, .finding-body, .tile-note { overflow-wrap: anywhere; }
.cm { table-layout: fixed; }

@media (max-width: 992px) {
    .grid16 { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
    .block-container { padding: 3.5rem 0.8rem 2rem 0.8rem !important; max-width: 100% !important; }
    .card { padding: 14px; margin-bottom: 12px; }
    .hdr-title { font-size: 17px; }
    .judul { font-size: 16px; }
    .pill { display: inline-block; margin: 6px 6px 0 0; font-size: 10px; }

    /* menu utama: dua kolom, tidak terpotong */
    .stRadio div[role="radiogroup"] { flex-wrap: wrap; width: 100%; }
    .stRadio label[data-baseweb="radio"] {
        flex: 1 1 45%; justify-content: center; text-align: center; padding: 8px 6px;
    }

    .grid16 { grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .tile-val { font-size: 14px; }
    .gauge { padding: 12px; }
    .gauge-scale { font-size: 8.5px; }
    .gauge-scale span:nth-child(2), .gauge-scale span:nth-child(3) { display: none; }
    .chips { font-size: 12px; }
    .finding-top { flex-wrap: wrap; }
    .fi-head { font-size: 11px; }

    /* confusion matrix muat di layar sempit */
    .cm { border-spacing: 3px; }
    .cm td { padding: 8px 3px; font-size: 10px; }
    .cm td.good, .cm td.bad { font-size: 13px; }
    .cm td small { font-size: 8.5px; }

    /* tombol dan input lebih mudah disentuh */
    button[kind="primary"], button[kind="secondary"] { min-height: 42px; }
    input[type="text"] { font-size: 16px !important; }
}
@media (max-width: 400px) {
    .grid16 { grid-template-columns: 1fr 1fr; }
    .hdr-title { font-size: 15px; }
}
"""

with st.sidebar:
    st.markdown("### Tampilan")
    tema_pilihan = st.radio("Tema antarmuka", list(TEMA.keys()), index=0)
    st.markdown("---")
    st.markdown("**Ringkasan sistem**")
    st.caption("Dataset uji: 47.074 sampel")
    st.caption("Model: Random Forest dan XGBoost")
    st.caption("Metode: analisis leksikal pasif, tanpa koneksi ke URL target")

_t = TEMA[tema_pilihan]
_vars = ":root{" + "".join(f"--{k}:{v};" for k, v in _t.items()) + "}"
st.markdown(f"<style>{_vars}{CSS_STATIS}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Logika
# ----------------------------------------------------------------------------
@st.cache_resource
def muat_model(path_file: Path):
    return joblib.load(path_file)


def prediksi_dummy(fitur_df: pd.DataFrame) -> float:
    """Prediksi sementara berbasis aturan jika file model tidak tersedia."""
    f = fitur_df.iloc[0]
    skor = 0.10
    skor += 0.35 * (f.get("has_ip", 0) == 1)
    skor += 0.30 * (f.get("has_at", 0) == 1)
    skor += 0.15 * (f.get("is_https", 1) == 0)
    skor += 0.15 * (f.get("url_length", 0) > 75)
    skor += 0.15 * (f.get("num_hyphens", 0) >= 3)
    skor += 0.10 * (f.get("num_subdomains", 0) >= 3)
    skor += 0.10 * (f.get("digit_ratio", 0) > 0.20)
    return float(min(max(skor, 0.01), 0.99))


def inferensi(nama_model: str, fitur_df: pd.DataFrame):
    """Mengembalikan (peluang phishing, apakah memakai hasil sementara)."""
    path = MODEL_FILES[nama_model]
    if path.exists():
        try:
            paket = muat_model(path)
            kolom = paket.get("columns", list(fitur_df.columns))
            return float(paket["model"].predict_proba(fitur_df[kolom])[0][1]), False
        except Exception:
            pass
    return prediksi_dummy(fitur_df), True


def tentukan_status(prob: float):
    if prob < 0.35:
        return "AMAN", "b-safe", "g-safe", "var(--ok)"
    if prob <= 0.65:
        return "WASPADA", "b-warn", "g-warn", "var(--warn)"
    return "PHISHING", "b-danger", "g-danger", "var(--danger)"


def hitung_temuan(f: dict) -> list:
    """Penjelasan heuristik dari struktur URL. Bukan dasar keputusan model."""
    t = []
    if f.get("has_ip", 0) == 1:
        t.append(("Tinggi", "Host berupa alamat IP", "URL memakai alamat IP sebagai pengganti nama domain, sehingga identitas situs sulit dikenali."))
    if f.get("has_at", 0) == 1:
        t.append(("Tinggi", "Mengandung karakter @", "Teks sebelum @ diabaikan peramban, sehingga tujuan sebenarnya bisa disamarkan."))
    if f.get("is_https", 1) == 0:
        t.append(("Sedang", "Tidak memakai HTTPS", "Koneksi tidak terenkripsi. Perlu diingat, HTTPS saja tidak menjamin sebuah situs aman."))
    if f.get("url_length", 0) > 75:
        t.append(("Sedang", f"URL panjang ({f['url_length']} karakter)", "URL yang sangat panjang kerap dipakai untuk menyembunyikan domain asli."))
    if f.get("num_hyphens", 0) >= 3:
        t.append(("Sedang", f"Banyak tanda hubung ({f['num_hyphens']})", "Pola umum pada domain tiruan yang meniru nama layanan resmi."))
    if f.get("num_subdomains", 0) >= 3:
        t.append(("Sedang", f"Subdomain berlapis ({f['num_subdomains']} tingkat)", "Subdomain bertingkat dapat menyamarkan domain utama sebenarnya."))
    if f.get("digit_ratio", 0) > 0.20 or f.get("num_digits", 0) >= 10:
        t.append(("Sedang", f"Proporsi angka tinggi ({f['digit_ratio']:.0%})", "Banyak angka sering muncul pada nama domain yang dibuat otomatis."))
    if f.get("num_params", 0) >= 3:
        t.append(("Rendah", f"Parameter banyak ({f['num_params']})", "Parameter yang banyak bisa dipakai untuk pelacakan atau pengalihan."))
    if f.get("num_special", 0) > 10:
        t.append(("Rendah", f"Karakter khusus banyak ({f['num_special']})", "Simbol yang berlebihan kadang menandakan pengodean URL untuk menyamarkan isi."))
    return t


def jalankan_inspeksi():
    """Callback tombol: memproses URL sebelum halaman digambar ulang."""
    url = st.session_state.get("url_input", "").strip()
    st.session_state["pesan"] = ""
    if not url:
        st.session_state["pesan"] = "Masukkan URL terlebih dahulu."
        return
    if "." not in url or " " in url:
        st.session_state["pesan"] = "Format URL tidak valid. Gunakan alamat lengkap, misalnya https://contoh.com."
        return
    try:
        fitur = extract_features(url)
    except Exception:
        st.session_state["pesan"] = "URL tidak dapat diproses. Periksa kembali penulisannya."
        return
    df = pd.DataFrame([fitur])
    p_rf, d_rf = inferensi("Random Forest", df)
    p_xgb, d_xgb = inferensi("XGBoost", df)
    mode = st.session_state.get("mode_mesin", "Bandingkan keduanya")
    if mode == "Random Forest":
        p = p_rf
    elif mode == "XGBoost":
        p = p_xgb
    else:
        p = (p_rf + p_xgb) / 2.0
    status, badge, gauge, warna = tentukan_status(p)
    st.session_state["hasil"] = {
        "url": url, "fitur": fitur, "p_rf": p_rf, "p_xgb": p_xgb, "dummy": d_rf or d_xgb,
        "s_rf": tentukan_status(p_rf), "s_xgb": tentukan_status(p_xgb),
        "p": p, "status": status, "badge": badge, "gauge": gauge, "warna": warna,
        "mode": mode, "temuan": hitung_temuan(fitur),
    }
    st.session_state["riwayat"].append({"url": url, "status": status, "badge": badge,
                                        "waktu": datetime.now().strftime("%H:%M:%S")})


def pasang_preset(url: str):
    st.session_state["url_input"] = url


def hapus_riwayat():
    st.session_state["riwayat"] = []


for kunci, awal in {"riwayat": [], "hasil": None, "pesan": "", "url_input": ""}.items():
    st.session_state.setdefault(kunci, awal)


# ----------------------------------------------------------------------------
# Komponen HTML kecil
# ----------------------------------------------------------------------------
def fi_bar(nama: str, persen: float) -> str:
    return (f'<div class="fi"><div class="fi-head"><span><code>{nama}</code></span>'
            f'<strong>{persen:.1f}%</strong></div><div class="fi-track">'
            f'<div class="fi-fill" style="width:{min(persen, 100):.1f}%"></div></div></div>')


def cm_html(judul: str, tn: int, fp: int, fn: int, tp: int) -> str:
    return f"""<div class="card"><div class="kicker">{judul}</div>
    <table class="cm">
      <tr><td></td><td>Prediksi aman</td><td>Prediksi phishing</td></tr>
      <tr><td>Aktual aman</td><td class="good">{tn:,}<small>True Negative</small></td>
          <td class="bad">{fp:,}<small>False Positive</small></td></tr>
      <tr><td>Aktual phishing</td><td class="bad">{fn:,}<small>False Negative</small></td>
          <td class="good">{tp:,}<small>True Positive</small></td></tr>
    </table></div>""".replace(",", ".")


def kpi(label: str, nilai: str, catatan: str, warna: str = "var(--text)") -> str:
    return (f'<div class="tile" style="padding:14px"><div class="tile-label">{label}</div>'
            f'<div class="tile-val" style="font-size:24px;color:{warna}">{nilai}</div>'
            f'<div class="tile-note">{catatan}</div></div>')


# ----------------------------------------------------------------------------
# Header dan navigasi
# ----------------------------------------------------------------------------
st.markdown(
    """<div class="card" style="margin-bottom:12px"><div class="hdr">
    <div><div class="hdr-title">Detektor Phishing URL</div>
    <div class="sub">Perbandingan Random Forest dan XGBoost. Final project Keamanan Informasi, Universitas Siliwangi.</div></div>
    <div><span class="pill">RF + XGBOOST</span><span class="pill">16 FITUR</span><span class="pill">TANPA KONEKSI KE TARGET</span></div>
    </div></div>""", unsafe_allow_html=True)

MENU = ["Periksa URL", "Evaluasi Model", "Studi Kasus", "Cara Kerja"]
menu = st.radio("Menu", MENU, horizontal=True, label_visibility="collapsed")

# ----------------------------------------------------------------------------
# Menu 1: Periksa URL
# ----------------------------------------------------------------------------
if menu == "Periksa URL":
    kiri, kanan = st.columns([4, 6], gap="medium")

    with kiri:
        with st.container(border=True, key="panel_kiri"):
            st.markdown('<div class="kicker">INPUT</div><div class="judul" style="font-size:15px">URL target dan skenario uji</div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="font-size:12px;font-weight:700;margin-bottom:6px">Skenario cepat</div>',
                        unsafe_allow_html=True)
            nama_preset = list(PRESET.keys())
            c1, c2 = st.columns(2)
            for i, nama in enumerate(nama_preset):
                with (c1 if i % 2 == 0 else c2):
                    st.button(nama, key=f"preset_{i}", use_container_width=True,
                              on_click=pasang_preset, args=(PRESET[nama],))

            st.text_input("Alamat URL", key="url_input", placeholder="https://contoh.com/login",
                          help="Aplikasi hanya membaca teks URL, tidak membukanya.")
            st.selectbox("Model yang dipakai", ["Bandingkan keduanya", "Random Forest", "XGBoost"],
                         key="mode_mesin")
            st.button("Periksa URL", type="primary", use_container_width=True, on_click=jalankan_inspeksi)

            if st.session_state["pesan"]:
                st.warning(st.session_state["pesan"])

            st.markdown('<div class="kicker" style="margin-top:16px">RIWAYAT SESI</div>', unsafe_allow_html=True)
            if st.session_state["riwayat"]:
                for item in reversed(st.session_state["riwayat"][-5:]):
                    teks = escape(item["url"] if len(item["url"]) <= 30 else item["url"][:30] + "...")
                    st.markdown(
                        f'<div class="mono" style="font-size:11.5px;padding:4px 0;border-bottom:1px dashed var(--border)">'
                        f'<span class="badge {item["badge"]}">{item["status"]}</span> '
                        f'<span style="margin-left:6px;word-break:break-all">{teks}</span></div>',
                        unsafe_allow_html=True)
                st.button("Hapus riwayat", use_container_width=True, on_click=hapus_riwayat)
            else:
                st.caption("Belum ada URL yang diperiksa pada sesi ini.")

    with kanan:
        with st.container(border=True, key="panel_kanan"):
            d = st.session_state["hasil"]
            if not d:
                st.markdown(
                    """<div class="standby"><div class="ring">TEKS</div>
                    <div style="font-size:17px;font-weight:800;margin-bottom:6px">Belum ada URL yang diperiksa</div>
                    <div class="sub" style="max-width:440px;margin:0 auto">Pilih salah satu skenario cepat atau ketik
                    sebuah URL di panel kiri, lalu tekan tombol Periksa URL. Aplikasi hanya membaca teks URL
                    dan tidak membukanya.</div></div>""", unsafe_allow_html=True)
            else:
                if d["dummy"]:
                    st.info("File model tidak ditemukan atau gagal dimuat, sehingga hasil memakai penilaian sementara berbasis aturan.")

                persen = d["p"] * 100
                (s_rf, b_rf, _, _), (s_x, b_x, _, _) = d["s_rf"], d["s_xgb"]
                sepakat = s_rf == s_x
                label_mode = "Rata-rata kedua model" if d["mode"] == "Bandingkan keduanya" else d["mode"]
                st.markdown(
                    f"""<div class="gauge"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
                    <div><span class="badge {d['badge']}" style="font-size:13px;padding:4px 12px">{d['status']}</span>
                    <span class="badge {'b-safe' if sepakat else 'b-warn'}" style="margin-left:6px">
                    {'KEDUA MODEL SEPAKAT' if sepakat else 'MODEL TIDAK SEPAKAT'}</span></div>
                    <div class="mono" style="font-size:26px;font-weight:800;color:{d['warna']}">{persen:.1f}%</div></div>
                    <div class="tile-note">Peluang phishing, {label_mode}</div>
                    <div class="gauge-track"><div class="gauge-fill {d['gauge']}" style="width:{min(max(persen, 2), 100):.1f}%"></div></div>
                    <div class="gauge-scale"><span>0% aman</span><span>35% waspada</span><span>65% phishing</span><span>100%</span></div>
                    </div>""", unsafe_allow_html=True)

                if not sepakat:
                    st.markdown('<div class="note warn">Kedua model memberi penilaian berbeda untuk URL ini. '
                                'Perlakukan hasilnya dengan hati-hati dan periksa langsung ke situs resmi.</div>',
                                unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f'<div class="tile" style="padding:12px 14px;margin-bottom:12px">'
                                f'<div style="display:flex;justify-content:space-between"><span class="tile-label">Random Forest</span>'
                                f'<span class="badge {b_rf}">{s_rf}</span></div>'
                                f'<div class="tile-val" style="font-size:20px">{d["p_rf"]:.1%}</div>'
                                f'<div class="tile-note">Bobot tersebar pada HTTPS, jumlah garis miring, dan panjang path.</div></div>',
                                unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="tile" style="padding:12px 14px;margin-bottom:12px">'
                                f'<div style="display:flex;justify-content:space-between"><span class="tile-label">XGBoost</span>'
                                f'<span class="badge {b_x}">{s_x}</span></div>'
                                f'<div class="tile-val" style="font-size:20px">{d["p_xgb"]:.1%}</div>'
                                f'<div class="tile-note">Bobot terpusat pada HTTPS dan jumlah garis miring.</div></div>',
                                unsafe_allow_html=True)

                # Anatomi URL
                url_teks = d["url"]
                parsed = urlparse(url_teks if "://" in url_teks else "http://" + url_teks)
                ext = _ext_offline(url_teks)
                chips = [f'<span class="chip c-scheme"><small>SKEMA</small>{escape(parsed.scheme or "http")}://</span>']
                if ext.subdomain:
                    chips.append(f'<span class="chip c-sub"><small>SUBDOMAIN</small>{escape(ext.subdomain)}.</span>')
                chips.append(f'<span class="chip c-dom"><small>DOMAIN</small>{escape(ext.domain or parsed.netloc)}</span>')
                if ext.suffix:
                    chips.append(f'<span class="chip c-tld"><small>TLD</small>.{escape(ext.suffix)}</span>')
                if parsed.path and parsed.path != "/":
                    chips.append(f'<span class="chip c-path"><small>PATH</small>{escape(parsed.path)}</span>')
                if parsed.query:
                    chips.append(f'<span class="chip c-query"><small>QUERY</small>?{escape(parsed.query)}</span>')
                st.markdown(f'<div class="dissect"><div class="kicker">BAGIAN-BAGIAN URL</div>'
                            f'<div class="chips">{"".join(chips)}</div></div>', unsafe_allow_html=True)

                # Grid fitur
                st.markdown('<div class="kicker" style="margin-top:6px">16 FITUR YANG DIHITUNG</div>', unsafe_allow_html=True)
                f = d["fitur"]
                kartu = []
                for k, (nama, desc) in FITUR_INFO.items():
                    v = f.get(k, 0)
                    if k == "is_https":
                        teks, tag = ("HTTPS", '<span class="t-ok">SESUAI</span>') if v == 1 else ("HTTP", '<span class="t-danger">TANPA ENKRIPSI</span>')
                    elif k in ("has_ip", "has_at"):
                        teks, tag = ("Ya", '<span class="t-danger">ANOMALI</span>') if v == 1 else ("Tidak", '<span class="t-ok">NORMAL</span>')
                    elif k == "digit_ratio":
                        teks, tag = f"{v:.1%}", ('<span class="t-warn">TINGGI</span>' if v > 0.20 else '<span class="t-ok">NORMAL</span>')
                    elif k == "url_length":
                        teks, tag = f"{v}", ('<span class="t-warn">PANJANG</span>' if v > 75 else '<span class="t-ok">NORMAL</span>')
                    else:
                        teks, tag = str(v), ""
                    kartu.append(f'<div class="tile" title="{escape(desc)}"><div class="tile-label">{nama}</div>'
                                 f'<div class="tile-val">{teks}</div><div>{tag}</div></div>')
                st.markdown(f'<div class="grid16">{"".join(kartu)}</div>', unsafe_allow_html=True)

                # Temuan aturan
                st.markdown('<div class="kicker" style="margin-top:16px">TEMUAN PADA STRUKTUR URL</div>', unsafe_allow_html=True)
                if d["temuan"]:
                    for tingkat, judul, isi in d["temuan"]:
                        st.markdown(
                            f'<div class="finding {"high" if tingkat == "Tinggi" else ""}"><div class="finding-top">'
                            f'<strong style="font-size:13px">{judul}</strong>'
                            f'<span class="badge {"b-danger" if tingkat == "Tinggi" else "b-warn" if tingkat == "Sedang" else "b-info"}">{tingkat.upper()}</span></div>'
                            f'<div class="finding-body">{isi}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="finding" style="border-left-color:var(--ok)"><div class="finding-body">'
                                'Tidak ditemukan ciri mencurigakan pada struktur URL.</div></div>', unsafe_allow_html=True)

                tinggi = sum(1 for t in d["temuan"] if t[0] == "Tinggi")
                if d["status"] == "AMAN" and tinggi:
                    st.markdown('<div class="note warn">Model menilai URL ini aman, tetapi aturan menemukan ciri berisiko tinggi. '
                                'Keduanya bisa berbeda karena model belajar dari pola data latih.</div>', unsafe_allow_html=True)
                elif d["status"] == "PHISHING" and not d["temuan"]:
                    st.markdown('<div class="note warn">Model menilai URL ini phishing, tetapi tidak ada ciri yang menonjol pada aturan. '
                                'Keputusan model berasal dari kombinasi fitur, dan kasus seperti ini bisa menjadi false positive.</div>',
                                unsafe_allow_html=True)
                st.markdown('<div class="note">Keputusan akhir berasal dari model machine learning. Temuan di atas adalah penjelasan '
                            'heuristik untuk membantu memahami struktur URL, bukan dasar perhitungan model.</div>', unsafe_allow_html=True)

                saran = {
                    "AMAN": "Tetap periksa alamat situs sebelum memasukkan data pribadi. Hasil ini bukan jaminan keamanan.",
                    "WASPADA": "Jangan memasukkan kata sandi atau data pribadi. Buka situs resmi lewat pencarian atau penanda yang Anda simpan sendiri.",
                    "PHISHING": "Jangan membuka tautan ini dan jangan memasukkan data apa pun. Laporkan ke penyedia layanan atau pengelola jaringan Anda.",
                }
                st.markdown(f'<div class="kicker">SARAN TINDAKAN</div><div class="sub">{saran[d["status"]]}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Menu 2: Evaluasi Model
# ----------------------------------------------------------------------------
elif menu == "Evaluasi Model":
    st.markdown('<div class="card"><div class="kicker">EVALUASI</div><div class="judul">Kinerja Random Forest dan XGBoost</div>'
                '<div class="sub">Data uji 47.074 URL (20% dari data), terdiri atas 26.970 URL aman dan 20.104 URL phishing. '
                'Metrik dihitung untuk kelas phishing.</div></div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Precision", "99,90%", "XGBoost 99,90% | RF 99,85%", "var(--ok)"), unsafe_allow_html=True)
    k2.markdown(kpi("Recall", "99,13%", "XGBoost 99,13% | RF 99,13%", "var(--accent)"), unsafe_allow_html=True)
    k3.markdown(kpi("F1-score", "99,51%", "XGBoost 99,51% | RF 99,49%", "var(--purple)"), unsafe_allow_html=True)
    k4.markdown(kpi("Data uji", "47.074", "26.970 aman | 20.104 phishing"), unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(cm_html("CONFUSION MATRIX: RANDOM FOREST", 26940, 30, 174, 19930), unsafe_allow_html=True)
    c2.markdown(cm_html("CONFUSION MATRIX: XGBOOST", 26950, 20, 175, 19929), unsafe_allow_html=True)

    st.markdown(
        """<div class="note"><strong>False positive dan false negative.</strong><br>
        False positive (30 pada RF, 20 pada XGBoost): situs aman dinilai phishing, akibatnya situs sah terblokir dan pengguna terganggu.<br>
        False negative (174 pada RF, 175 pada XGBoost): phishing dinilai aman. Ini lebih berbahaya karena pengguna bisa percaya
        pada tautan palsu dan memasukkan data pribadi. Sebanyak 169 URL phishing yang lolos sama pada kedua model, sehingga
        penyebabnya lebih mungkin keterbatasan fitur daripada kelemahan satu algoritma.</div>""", unsafe_allow_html=True)

    st.markdown("#### Kepentingan fitur")
    f1, f2 = st.columns(2)
    f1.markdown('<div class="card"><div class="kicker">RANDOM FOREST</div>'
                + fi_bar("is_https", 43.1) + fi_bar("num_slashes", 18.1) + fi_bar("path_length", 16.4)
                + fi_bar("num_digits", 5.6) + fi_bar("digit_ratio", 4.5) + fi_bar("fitur lainnya (gabungan)", 12.3)
                + '</div>', unsafe_allow_html=True)
    f2.markdown('<div class="card"><div class="kicker">XGBOOST</div>'
                + fi_bar("is_https", 52.5) + fi_bar("num_slashes", 46.5) + fi_bar("num_dots", 0.3)
                + fi_bar("url_length", 0.2) + fi_bar("num_digits", 0.2) + fi_bar("fitur lainnya (gabungan)", 0.3)
                + '</div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="note warn"><strong>Catatan kehati-hatian.</strong> XGBoost hampir seluruhnya bergantung pada dua fitur,
        yaitu HTTPS dan jumlah garis miring. Pada dataset PhiUSIIL, URL aman umumnya berupa halaman depan tanpa path, sehingga
        model dapat mempelajari pola dataset, bukan ciri phishing yang sebenarnya. Skor di atas 99% belum tentu berlaku pada
        URL di dunia nyata, misalnya URL aman yang memiliki path panjang atau memakai HTTP.</div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Menu 3: Studi Kasus
# ----------------------------------------------------------------------------
elif menu == "Studi Kasus":
    st.markdown('<div class="card"><div class="kicker">STUDI KASUS</div><div class="judul">Contoh kesalahan klasifikasi</div>'
                '<div class="sub">Contoh diambil dari hasil evaluasi pada data uji. Semua URL di bawah hanya ditampilkan sebagai teks.</div></div>',
                unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown(
        """<div class="card" style="height:100%"><div class="kicker" style="color:var(--warn)">FALSE POSITIVE: SITUS AMAN DINILAI PHISHING</div>
        <div style="font-size:12.5px;line-height:1.7">
        <strong><code>bmvc2020-conference.com</code></strong><br>
        <span class="sub">Memuat angka tahun dan tanda hubung pada nama domain, kombinasi yang sering muncul pada phishing di data latih.</span><br><br>
        <strong><code>formula1-dictionary.net</code></strong><br>
        <span class="sub">Nama domain bertanda hubung dan berisi angka, sehingga terbaca mirip pola phishing.</span><br><br>
        <strong><code>chandra.cfa.harvard.edu</code></strong><br>
        <span class="sub">Memiliki tiga lapis subdomain dan banyak titik. Struktur ini dipelajari model sebagai ciri phishing, padahal situs ini sah.</span>
        </div></div>""", unsafe_allow_html=True)
    b.markdown(
        """<div class="card" style="height:100%"><div class="kicker" style="color:var(--danger)">FALSE NEGATIVE: PHISHING LOLOS</div>
        <div style="font-size:12.5px;line-height:1.7">
        <strong><code>verifynew--verifynewnew.repl.co</code></strong><br>
        <span class="sub">Berada di platform hosting gratis yang memakai HTTPS. Fitur kita tidak membaca kata seperti verify, sehingga teksnya tampak wajar.</span><br><br>
        <strong><code>sdgvsdvsdvs.blogspot.com</code></strong><br>
        <span class="sub">Berada di layanan blog yang sah, dengan host pendek, HTTPS, dan tanpa path, sehingga mirip situs normal.</span><br><br>
        <strong><code>continue-checksout.yourtrap.com</code></strong><br>
        <span class="sub">Teks URL bersih dan pendek dengan satu tanda hubung. Kata seperti checkout tidak terbaca oleh fitur karakter.</span>
        </div></div>""", unsafe_allow_html=True)

    st.markdown(
        """<div class="card"><div class="kicker">KETERBATASAN DAN PENGEMBANGAN</div>
        <div class="sub">1. Model hanya membaca teks URL, tanpa isi halaman, usia domain, atau reputasi.<br>
        2. Fitur berupa hitungan karakter, belum membaca kata kunci seperti login, verify, atau secure.<br>
        3. Banyak serangan phishing kini memakai HTTPS dan platform hosting gratis, sehingga struktur URL tampak wajar.<br>
        4. Situs sah yang diretas tetap bernama domain sah dan sulit dideteksi dari teks URL.<br>
        5. Dataset berasal dari satu sumber, sehingga hasil belum tentu sama pada data nyata.<br><br>
        <strong>Saran pengembangan:</strong> tambah fitur kata kunci, daftar hosting gratis, entropi nama domain, usia domain,
        dan fitur isi halaman, lalu uji pada dataset lain.</div></div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Menu 4: Cara Kerja
# ----------------------------------------------------------------------------
else:
    st.markdown('<div class="card"><div class="kicker">CARA KERJA</div><div class="judul">Dari teks URL menjadi keputusan</div>'
                '<div class="sub">Aplikasi menghitung 16 fitur dari teks URL, lalu memasukkannya ke dua model machine learning. '
                'Aplikasi tidak pernah membuka URL yang dimasukkan.</div></div>', unsafe_allow_html=True)

    tahap = [
        ("TAHAP 1", "Input URL", "Teks URL diterima dari pengguna tanpa koneksi ke target."),
        ("TAHAP 2", "Ekstraksi fitur", "16 fitur dihitung oleh fungsi extract_features di features.py."),
        ("TAHAP 3", "Penyelarasan kolom", "Urutan kolom disamakan dengan data saat training."),
        ("TAHAP 4", "Inferensi", "Random Forest dan XGBoost menghitung peluang phishing."),
        ("TAHAP 5", "Keputusan", "Peluang diubah menjadi Aman, Waspada, atau Phishing, disertai temuan struktur URL."),
    ]
    for kolom, (n, j, e) in zip(st.columns(5), tahap):
        kolom.markdown(f'<div class="tile" style="height:100%;padding:12px"><div class="step-num">{n}</div>'
                       f'<div style="font-size:13px;font-weight:700;margin:4px 0">{j}</div>'
                       f'<div class="tile-note">{e}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="note">Peluang di bawah 35% dikategorikan aman, 35% sampai 65% waspada, dan di atas 65% phishing. '
                'Pada mode Bandingkan keduanya, peluang yang ditampilkan adalah rata-rata kedua model.</div>', unsafe_allow_html=True)

    st.markdown("#### Dua model yang dibandingkan")
    m1, m2 = st.columns(2)
    m1.markdown('<div class="card"><div class="kicker">RANDOM FOREST</div><div class="sub">Gabungan banyak pohon keputusan '
                '(200 pohon) yang masing-masing dilatih pada sampel data berbeda, lalu hasilnya dirata-ratakan.</div></div>',
                unsafe_allow_html=True)
    m2.markdown('<div class="card"><div class="kicker">XGBOOST</div><div class="sub">Pohon keputusan yang dibangun berurutan, '
                'setiap pohon baru memperbaiki kesalahan pohon sebelumnya (gradient boosting).</div></div>',
                unsafe_allow_html=True)

    st.markdown("#### Katalog 16 fitur")
    daftar = list(FITUR_INFO.items())
    for kolom, bagian in zip(st.columns(2), (daftar[:8], daftar[8:])):
        for k, (nama, desc) in bagian:
            kolom.markdown(f'<div class="tile" style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;gap:8px">'
                           f'<code>{k}</code><strong style="font-size:12px">{nama}</strong></div>'
                           f'<div class="tile-note">{desc}</div></div>', unsafe_allow_html=True)


st.caption("Aplikasi ini hanya untuk edukasi dan bukan pengganti antivirus atau layanan keamanan resmi.")