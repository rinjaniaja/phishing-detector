# Detektor Phishing URL

Perbandingan Random Forest dan XGBoost untuk deteksi phishing URL.
Proyek final Keamanan Informasi, Universitas Siliwangi.

## Cara pasang
pip install -r requirements.txt

## Cara menjalankan
cd app
python -m streamlit run app.py

## Struktur folder
- app/    : aplikasi web Streamlit
- src/    : ekstraksi fitur dan training
- models/ : model terlatih (Random Forest dan XGBoost)

## Catatan
Aplikasi hanya membaca teks URL, tidak membuka halamannya.
Dibuat untuk keperluan akademik.