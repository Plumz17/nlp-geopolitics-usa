# NLP Final Project
## Prediksi Peristiwa Geopolitik Global: Dampak Terhadap Nilai Tukar Dolar
### Deskripsi Proyek
Proyek ini membangun pipeline NLP end-to-end untuk menguji hipotesis: **berita
geopolitik global memiliki pengaruh signifikan dan dapat membantu memprediksi
fluktuasi nilai tukar Dolar AS (USD)**. Target prediksi adalah arah kurs USD/IDR
(JISDOR) pada hari perdagangan berikutnya.

Proyek dikerjakan dalam empat tahap sepanjang perkuliahan:
1. Akuisisi Data & Preprocessing Strategis
2. Proposal Pipeline & Eksperimen Baseline
3. Penyempurnaan Model & Integrasi Multimodal
4. Pengujian Hipotesis, Analisis Error, & Laporan Akhir

Progress kelompok kami berada pada **Tugas 3: Penyempurnaan Model & Integrasi Multimodal**.
Tugas 1 dan 2 sudah selesai (data, baseline, dan evaluasi awal).

### Struktur Repositori
```
data/
  raw/         berita hasil scraping dan data kurs mentah dari Bank Indonesia
  processed/   artikel bersih, kurs harian, dan dataset hasil alignment
  lexicon/     kamus Loughran-McDonald
  splits/      train.csv, val.csv, test.csv (pembagian urut waktu) dan articles/ (teks berita per bagian)
  features/    features.csv (fitur NLP harian)
  results/     predictions.csv (prediksi seluruh model)
src/           notebook pipeline, dijalankan berurutan (lihat di bawah)
notebook/      EDA dan hasil eksperimen
```

### Urutan Menjalankan Pipeline
Jalankan notebook di `src/` dari folder `src/`, sesuai urutan:

| Notebook | Fungsi |
|---|---|
| `00a_scraper` | Scraping berita (Google News RSS) ke `data/raw/` |
| `00b_preprocessing` | Pembersihan teks, deduplikasi, dan `articles_clean.csv` |
| `00c_timeseries_jisdor` | Membersihkan data kurs JISDOR |
| `00d_alignment` | Menyelaraskan berita dengan hari perdagangan |
| `01_data_split` | Membentuk target, fitur harga, dan pembagian train/val/test |
| `02_feature_extraction` | Fitur leksikon (VADER, Loughran-McDonald) dan TF-IDF |
| `03_baseline_models` | Model baseline, model harga, dan model gabungan |
| `04_evaluation` | Evaluasi dan analisis hasil |

EDA dan hasil eksperimen awal ada di `notebook/`.

### Instalasi
```
pip install -r requirements.txt
```

### Anggota Tim "Jempol Kaki Azhar"

| Nama | NIM |
|------|-----|
| Anders Emmanuel Tan | 24/541351/PA/22964 |
| Azhar Maulana | 24/533487/PA/22582 |
| Evan Razzan Adytaputra | 24/545257/PA/23166 |
| Kukuh Agus Hermawan | 24/533395/PA/22573 |
