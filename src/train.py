import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from xgboost import XGBClassifier

df = pd.read_csv("../data/features.csv")
X = df.drop(columns=["url", "label"])
y = df["label"]
urls = df["url"]

X_train, X_test, y_train, y_test, u_train, u_test = train_test_split(
    X, y, urls, test_size=0.2, stratify=y, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             eval_metric="logloss", random_state=42),
}

hasil = []
for nama, model in models.items():
    kode = "rf" if nama == "Random Forest" else "xgb"
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("=====", nama, "=====")
    print(classification_report(y_test, pred, target_names=["Aman", "Phishing"]))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Aman", "Phishing"], yticklabels=["Aman", "Phishing"])
    plt.xlabel("Prediksi")
    plt.ylabel("Sebenarnya")
    plt.title(nama)
    plt.tight_layout()
    plt.savefig(f"../data/cm_{kode}.png", dpi=150)
    plt.close()

    hasil.append({
        "Model": nama,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    })

    tabel = pd.DataFrame({"url": u_test.values, "asli": y_test.values, "prediksi": pred})
    tabel[(tabel.asli == 0) & (tabel.prediksi == 1)].to_csv(f"../data/FP_{kode}.csv", index=False)
    tabel[(tabel.asli == 1) & (tabel.prediksi == 0)].to_csv(f"../data/FN_{kode}.csv", index=False)

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Fitur terpenting:\n", imp.head(8), "\n")

    joblib.dump({"model": model, "columns": list(X.columns)}, f"../models/model_{kode}.joblib")

tabel_hasil = pd.DataFrame(hasil).round(4)
print(tabel_hasil)
tabel_hasil.to_csv("../data/hasil_evaluasi.csv", index=False)