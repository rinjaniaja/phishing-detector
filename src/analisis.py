import pandas as pd
from features import extract_features

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

for kode, nama in [("rf", "Random Forest"), ("xgb", "XGBoost")]:
    fp = pd.read_csv(f"../data/FP_{kode}.csv")
    fn = pd.read_csv(f"../data/FN_{kode}.csv")

    print("=" * 70)
    print(nama)
    print("Jumlah FP (aman dikira phishing):", len(fp))
    print("Jumlah FN (phishing lolos)      :", len(fn))

    for judul, tabel in [("FP", fp), ("FN", fn)]:
        print(f"\n--- 10 contoh {judul} ({nama}) ---")
        contoh = tabel.sample(min(10, len(tabel)), random_state=1)
        hasil = []
        for u in contoh["url"]:
            f = extract_features(u)
            hasil.append({
                "url": u[:70],
                "panjang": f["url_length"],
                "titik": f["num_dots"],
                "hubung": f["num_hyphens"],
                "digit": f["num_digits"],
                "https": f["is_https"],
                "subdomain": f["num_subdomains"],
            })
        print(pd.DataFrame(hasil).to_string(index=False))

# URL yang salah di KEDUA model (kesalahan paling "sulit")
fp_rf = set(pd.read_csv("../data/FP_rf.csv")["url"])
fp_xgb = set(pd.read_csv("../data/FP_xgb.csv")["url"])
fn_rf = set(pd.read_csv("../data/FN_rf.csv")["url"])
fn_xgb = set(pd.read_csv("../data/FN_xgb.csv")["url"])
print("\nFP yang salah di kedua model:", len(fp_rf & fp_xgb))
print("FN yang salah di kedua model:", len(fn_rf & fn_xgb))