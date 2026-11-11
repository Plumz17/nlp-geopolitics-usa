"""Kode bersama untuk Model C (C_1, C_2, C_3, ...).

APA INI?
  Kumpulan fungsi yang dipakai semua notebook Model C (05a, 05b, 05c, ...), supaya semua varian
  diuji dengan cara yang SAMA dan hasilnya bisa dibandingkan dengan adil:
    - split yang sama dengan B_2 (data/splits_filtered)
    - fitur harga yang sama (PRICE)
    - tuning hanya di validation; test dinilai SEKALI untuk varian yang terpilih
    - hasil tiap varian disimpan ke results_c/<nama>.json

CARA PAKAI (di notebook, jalankan dari folder src/):

    # 1. Muat data
    from c_common import *
    D, text = load_all()              # default: data B_2 (hanya artikel Inggris terfilter)
    # D, text = load_all("id")        # atau: Inggris terfilter + Indonesia
    #   D["train"], D["val"], D["test"] = tabel harga + target (satu baris per hari trading)
    #   text = tabel teks per hari (kolom judul/lead/konten), index = tanggal

    # 2. Buat fitur teks kamu sendiri (embedding, skor leksikon, dll.).
    #    Hasilnya harus berupa tabel dengan kolom "date" + kolom-kolom fitur (angka).
    #    Contoh: fitur = pd.DataFrame({"date": ..., "emb_0": ..., "emb_1": ...})
    #    Simpan agar tidak perlu dihitung ulang:  fitur.to_csv(FEATURES_C / "C1.csv", index=False)

    # 3. Gabungkan fitur ke split berdasarkan tanggal
    for s in D:
        D[s] = D[s].merge(fitur, on="date", how="left").fillna(0)   # hari tanpa berita -> 0
    kolom_teks = [c for c in fitur.columns if c != "date"]

    # 4. Nilai di VALIDATION (boleh berkali-kali, untuk memilih varian/hyperparameter)
    evaluate_val("C1_LR",  "LR",  D, PRICE + kolom_teks, n_configs_tried=1)
    evaluate_val("C1_HGB", "HGB", D, PRICE + kolom_teks, n_configs_tried=1)

    # 5. Setelah SEMUA varian dicoba dan satu dipilih berdasarkan skor validation:
    final_test("C1_LR", "LR", D, PRICE + kolom_teks)     # skor test, SEKALI saja

    # 6. Lihat semua hasil dalam satu tabel
    collect_results()

ATURAN: pilih varian terbaik berdasarkan VALIDATION, bukan test. Jangan panggil final_test untuk
banyak varian lalu memilih yang paling bagus; itu membuat skor test terlalu optimis.
"""
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ lokasi file
# Notebook dijalankan dari src/, jadi data ada di ../data. (Kalau dijalankan dari root repo, pakai data/.)
DATA = Path("../data") if Path("../data").exists() else Path("data")
RESULTS_C = DATA.parent / "results_c"                      # satu file .json per varian C
FEATURES_C = DATA / "features_c"                           # tempat menyimpan fitur buatanmu (embedding, dll.)
SPLITS = DATA / "splits_filtered"                          # split yang sama dengan Model B_2
# File teks per hari, dipilih lewat parameter corpus di load_all():
TEXT_FILES = {"filtered": DATA / "processed" / "aligned_dataset_filtered.csv",  # B_2: hanya artikel Inggris terfilter
              "id": DATA / "processed" / "aligned_dataset_id.csv"}               # Inggris terfilter + Indonesia
SEED = 42                                                  # supaya hasil bisa direproduksi

# Fitur harga yang sama dengan Model A/B (lihat 03_baseline_models). Selalu dipakai bersama fitur teks,
# karena Model C = harga + teks.
PRICE = ["ret_1", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4",
         "ret_5", "ret_20", "vol_5", "vol_20", "z_20", "gap_days"]


# ------------------------------------------------------------------ data
def load_all(corpus="filtered"):
    """Muat data. Return (D, text).

    corpus  "filtered" (default) = data B_2: hanya artikel Inggris terfilter.
            "id" = artikel Inggris terfilter + artikel CNBC Indonesia (aligned_dataset_id.csv).
    D    = {"train": df, "val": df, "test": df}. Tiap df punya kolom date, target (1 = kurs naik besok,
           0 = turun), target_ret, dan semua fitur harga di PRICE. Urutannya waktu, jangan diacak.
           (Split-nya sama untuk kedua corpus, yaitu split B_2.)
    text = satu baris per hari trading (index = tanggal) dengan teks gabungan hari itu.
           Nama kolom berbeda per corpus:
             "filtered": jumlah_berita, judul_gabungan, lead_paragraph_gabungan, konten_gabungan
             "id":       jumlah_berita_en / _id, judul_gabungan_en / _id,
                         lead_paragraph_gabungan_en / _id, konten_gabungan_en / _id
           Hari tanpa berita berisi string kosong.
    Gunakan `text` sebagai bahan untuk membuat fitur; hasilnya digabung ke D lewat kolom date.
    """
    assert corpus in TEXT_FILES, f"corpus harus salah satu dari {list(TEXT_FILES)}"
    D = {s: pd.read_csv(SPLITS / f"{s}.csv", parse_dates=["date"]) for s in ["train", "val", "test"]}
    text = pd.read_csv(TEXT_FILES[corpus], parse_dates=["date"]).set_index("date")
    return D, text


# ------------------------------------------------------------------ model & metrik
def metrics(y, pred, score=None):
    """Hitung skor. y = arah sebenarnya (0/1), pred = tebakan (0/1), score = probabilitas naik.
    DA = akurasi arah. BalAcc/F1_macro/MCC = ukuran yang adil walau kelas tidak seimbang.
    AUC = seberapa baik probabilitas mengurutkan hari naik di atas hari turun (0.5 = acak)."""
    out = {"DA": accuracy_score(y, pred), "BalAcc": balanced_accuracy_score(y, pred),
           "F1_macro": f1_score(y, pred, average="macro"), "MCC": matthews_corrcoef(y, pred)}
    out["AUC"] = roc_auc_score(y, score) if score is not None else np.nan
    return out


# Kandidat hyperparameter yang dicoba saat tuning (sama dengan Tugas 2).
# LR = logistic regression (C kecil = lebih diregularisasi). HGB = gradient boosting.
GRID = {
    "LR": [{"C": c} for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]],
    "HGB": [dict(max_depth=d, learning_rate=l, max_iter=i, min_samples_leaf=m)
            for d, l, i, m in product([2, 3], [0.03, 0.1], [100, 300], [20, 50])],
}


def make_model(kind, p):
    """Buat model. kind = "LR" atau "HGB"; p = dict hyperparameter dari GRID.
    class_weight="balanced" supaya hari naik/turun diperlakukan seimbang."""
    if kind == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(**p, class_weight="balanced", max_iter=5000))
    return HistGradientBoostingClassifier(**p, l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def tune(kind, D, cols):
    """Coba semua hyperparameter di GRID: latih di train, nilai di validation, ambil yang F1-macro terbaik.
    cols = daftar kolom fitur yang dipakai (mis. PRICE + kolom_teks).
    Return (hyperparameter terbaik, skor validation, probabilitas validation)."""
    best = None
    for p in GRID[kind]:
        prob = make_model(kind, p).fit(D["train"][cols], D["train"]["target"]).predict_proba(D["val"][cols])[:, 1]
        sc = metrics(D["val"]["target"], (prob >= 0.5).astype(int), prob)
        if best is None or sc["F1_macro"] > best[1]["F1_macro"]:
            best = (p, sc, prob)
    return best


# ------------------------------------------------------------------ simpan hasil
def _save(name, record):
    """(Internal) Tulis/perbarui results_c/<name>.json. Isi lama dipertahankan, kunci baru ditambahkan."""
    RESULTS_C.mkdir(exist_ok=True)
    path = RESULTS_C / f"{name}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(record)
    path.write_text(json.dumps(old, indent=2, default=float))
    return old


def evaluate_val(name, kind, D, cols, n_configs_tried=1, note=""):
    """Tuning + skor di VALIDATION saja (test TIDAK disentuh). Aman dipanggil berkali-kali.

    name            nama varian, mis. "C1_LR". Jadi nama file results_c/C1_LR.json.
    kind            "LR" atau "HGB".
    D               hasil load_all() yang sudah digabung dengan fitur teksmu.
    cols            kolom fitur yang dipakai, mis. PRICE + kolom_teks.
    n_configs_tried berapa konfigurasi fitur yang kamu coba di balik nama ini (mis. 4 ukuran embedding
                    -> 4). Dilaporkan di laporan supaya pembaca tahu berapa banyak yang dicoba.
    note            catatan bebas (mis. "FastText 300d, rata-rata harian").
    """
    p, sc, _ = tune(kind, D, cols)
    _save(name, {"name": name, "model": kind, "n_features": len(cols), "best_params": p,
                 "val": sc, "n_configs_tried": n_configs_tried, "note": note})
    print(f"{name} [{kind}] val: " + ", ".join(f"{k}={v:.3f}" for k, v in sc.items()))
    return sc


def final_test(name, kind, D, cols):
    """Skor di TEST untuk varian yang sudah DIPILIH. Jalankan SEKALI per varian.

    Harus dipanggil setelah evaluate_val dengan name dan kind yang sama. Model dilatih ulang di
    train+val memakai hyperparameter terbaik dari validation, lalu diprediksi ke test. Hasilnya
    (skor + interval kepercayaan AUC) ditambahkan ke results_c/<name>.json.
    Kalau name sudah pernah dites, fungsi ini berhenti dengan error (supaya tidak ada pilih-pilih
    berdasarkan test).
    """
    rec = json.loads((RESULTS_C / f"{name}.json").read_text())
    assert "test" not in rec, f"{name} sudah pernah dites; jangan pilih ulang berdasarkan test."
    trval = pd.concat([D["train"], D["val"]])
    prob = make_model(kind, rec["best_params"]).fit(trval[cols], trval["target"]).predict_proba(D["test"][cols])[:, 1]
    sc = metrics(D["test"]["target"], (prob >= 0.5).astype(int), prob)
    ci = bootstrap_auc_ci(D["test"]["target"].values, prob)
    _save(name, {"test": sc, "test_auc_ci95": ci})
    print(f"{name} [{kind}] TEST: " + ", ".join(f"{k}={v:.3f}" for k, v in sc.items()) + f" | AUC 95% CI {ci}")
    return prob


def bootstrap_auc_ci(y, score, B=1000, block=5, seed=1):
    """Interval kepercayaan 95% untuk AUC. Hari dipilih ulang dalam blok 5 hari berurutan (karena hari
    yang berdekatan saling mirip). Test hanya 178 hari, jadi interval ini biasanya lebar: pakai untuk
    melihat apakah AUC benar-benar di atas 0.5."""
    g = np.random.default_rng(seed)
    n = len(y)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(B):
        ix = (g.integers(0, n - block + 1, nb)[:, None] + np.arange(block)).ravel()[:n]
        if len(set(y[ix])) == 2:  # AUC butuh kedua kelas ada
            out.append(roc_auc_score(y[ix], score[ix]))
    return [round(float(v), 3) for v in np.percentile(out, [2.5, 97.5])]


def shuffled_text_control(kind, D, price_cols, text_cols, params, n_perm=30):
    """Kontrol "teks diacak". Baris fitur teks diacak antar hari (harga tetap), lalu model dilatih dan
    dinilai di validation, diulang n_perm kali. Return rata-rata/min/maks AUC.
    Kalau AUC fitur teks aslimu tidak lebih tinggi dari kisaran ini, teksnya tidak benar-benar membantu.
    price_cols = PRICE, text_cols = kolom fitur teksmu, params = hyperparameter (dari evaluate_val's
    best_params di results_c/<name>.json)."""
    rs = np.random.default_rng(SEED)
    aucs = []
    for _ in range(n_perm):
        tr = D["train"].copy()
        va = D["val"].copy()
        tr[text_cols] = tr[text_cols].values[rs.permutation(len(tr))]
        va[text_cols] = va[text_cols].values[rs.permutation(len(va))]
        prob = make_model(kind, params).fit(tr[price_cols + text_cols], tr["target"]).predict_proba(va[price_cols + text_cols])[:, 1]
        aucs.append(roc_auc_score(va["target"], prob))
    return {"auc_mean": float(np.mean(aucs)), "auc_min": float(np.min(aucs)), "auc_max": float(np.max(aucs))}


def collect_results():
    """Gabungkan semua results_c/*.json menjadi satu tabel (satu baris per varian), dengan skor
    validation (val_*), skor test (test_*, kosong kalau belum dites), dan jumlah konfigurasi yang dicoba.
    Dipakai di 06_ablation.ipynb untuk matriks ablasi."""
    rows = []
    for f in sorted(RESULTS_C.glob("*.json")):
        r = json.loads(f.read_text())
        row = {"name": r["name"], "model": r.get("model"), "n_features": r.get("n_features"),
               "n_configs_tried": r.get("n_configs_tried")}
        row.update({f"val_{k}": v for k, v in r.get("val", {}).items()})
        row.update({f"test_{k}": v for k, v in r.get("test", {}).items()})
        row["test_auc_ci95"] = r.get("test_auc_ci95")
        rows.append(row)
    return pd.DataFrame(rows)
