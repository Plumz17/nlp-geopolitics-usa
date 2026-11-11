"""Kode bersama untuk Model C (C_1, C_2, C_3, ...).

Semua varian C memakai fungsi di sini supaya perbandingannya adil:
  - split yang sama dengan B_2 (data/splits_filtered)
  - fitur harga yang sama (PRICE)
  - tuning hanya di validation; test dinilai SEKALI untuk varian terpilih
  - hasil tiap varian disimpan ke results_c/<nama>.json

Pemakaian di notebook (jalankan dari folder src/):
    from c_common import *
    D, text = load_all()
    res = evaluate_val("C1", "LR", D, PRICE + fitur_c1, n_configs_tried=3)   # validation saja
    # ... setelah memilih varian terbaik di validation:
    final_test("C1", "LR", D, PRICE + fitur_c1)                              # test, sekali
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
DATA = Path("../data") if Path("../data").exists() else Path("data")
RESULTS_C = DATA.parent / "results_c"
FEATURES_C = DATA / "features_c"
SPLITS = DATA / "splits_filtered"  # split yang sama dengan Model B_2
TEXT_FILE = DATA / "processed" / "aligned_dataset_id.csv"  # teks Inggris (terfilter) + Indonesia per hari
SEED = 42

# Fitur harga yang sama dengan Model A/B (lihat 03_baseline_models)
PRICE = ["ret_1", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4",
         "ret_5", "ret_20", "vol_5", "vol_20", "z_20", "gap_days"]


# ------------------------------------------------------------------ data
def load_all():
    """Return (D, text). D = {"train","val","test"} berisi kolom harga + target.
    text = satu baris per hari trading: kolom teks *_en dan *_id dari aligned_dataset_id.csv."""
    D = {s: pd.read_csv(SPLITS / f"{s}.csv", parse_dates=["date"]) for s in ["train", "val", "test"]}
    text = pd.read_csv(TEXT_FILE, parse_dates=["date"]).set_index("date")
    return D, text


# ------------------------------------------------------------------ model & metrik
def metrics(y, pred, score=None):
    out = {"DA": accuracy_score(y, pred), "BalAcc": balanced_accuracy_score(y, pred),
           "F1_macro": f1_score(y, pred, average="macro"), "MCC": matthews_corrcoef(y, pred)}
    out["AUC"] = roc_auc_score(y, score) if score is not None else np.nan
    return out


GRID = {
    "LR": [{"C": c} for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]],
    "HGB": [dict(max_depth=d, learning_rate=l, max_iter=i, min_samples_leaf=m)
            for d, l, i, m in product([2, 3], [0.03, 0.1], [100, 300], [20, 50])],
}


def make_model(kind, p):
    if kind == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(**p, class_weight="balanced", max_iter=5000))
    return HistGradientBoostingClassifier(**p, l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def tune(kind, D, cols):
    """Coba semua hyperparameter: latih di train, nilai di validation, ambil F1-macro terbaik."""
    best = None
    for p in GRID[kind]:
        prob = make_model(kind, p).fit(D["train"][cols], D["train"]["target"]).predict_proba(D["val"][cols])[:, 1]
        sc = metrics(D["val"]["target"], (prob >= 0.5).astype(int), prob)
        if best is None or sc["F1_macro"] > best[1]["F1_macro"]:
            best = (p, sc, prob)
    return best


# ------------------------------------------------------------------ simpan hasil
def _save(name, record):
    RESULTS_C.mkdir(exist_ok=True)
    path = RESULTS_C / f"{name}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(record)
    path.write_text(json.dumps(old, indent=2, default=float))
    return old


def evaluate_val(name, kind, D, cols, n_configs_tried=1, note=""):
    """Tuning + skor VALIDATION saja (test tidak disentuh). Simpan ke results_c/<name>.json.
    n_configs_tried = berapa konfigurasi/varian fitur yang kamu coba untuk nama ini (dilaporkan di laporan)."""
    p, sc, _ = tune(kind, D, cols)
    _save(name, {"name": name, "model": kind, "n_features": len(cols), "best_params": p,
                 "val": sc, "n_configs_tried": n_configs_tried, "note": note})
    print(f"{name} [{kind}] val: " + ", ".join(f"{k}={v:.3f}" for k, v in sc.items()))
    return sc


def final_test(name, kind, D, cols):
    """Skor TEST untuk varian terpilih: latih ulang di train+val, prediksi test. Jalankan SEKALI."""
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
    """Interval kepercayaan 95% AUC dengan block bootstrap (blok hari berurutan)."""
    g = np.random.default_rng(seed)
    n = len(y)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(B):
        ix = (g.integers(0, n - block + 1, nb)[:, None] + np.arange(block)).ravel()[:n]
        if len(set(y[ix])) == 2:
            out.append(roc_auc_score(y[ix], score[ix]))
    return [round(float(v), 3) for v in np.percentile(out, [2.5, 97.5])]


def shuffled_text_control(kind, D, price_cols, text_cols, params, n_perm=30):
    """Kontrol: acak baris fitur teks (harga tetap), latih & nilai di validation.
    Kalau fitur teks asli tidak lebih baik dari ini, teksnya tidak membantu."""
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
    """Gabungkan semua results_c/*.json menjadi satu tabel (dipakai 06_ablation)."""
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
