# Struktur Lampiran

Dokumen ini menjelaskan struktur lampiran yang digunakan pada repository penelitian **Prediksi Harga Komoditas Pangan Bulanan Menggunakan Pendekatan Machine Learning Regression Berbasis Fitur Temporal**.

## Struktur Repository

```text
.
├── data/
│   ├── raw/
│   │   ├── 1778217759.csv
│   │   └── raw.md
│   │
│   └── processed/
│       ├── data_preprocessing_non_encode.csv
│       ├── data_preprocessing_encoded.csv
│       └── processed.md
│
├── notebook/
│   ├── preprocessing_template.ipynb
│   ├── eda_template.ipynb
│   └── modeling_template.ipynb
│
├── report/
│   ├── Laporan Akhir Data Mining.pdf
│   ├── Paper_Kelompok4.pdf
│   └── struktur-lampiran.md
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── utils.py
│   ├── main.py
│   └── main_notebook.ipynb
│
├── requirements.txt
├── run.sh
└── README.md
```

---

## Deskripsi Lampiran

### 1. Folder `data/`

Berisi seluruh dataset yang digunakan.

#### `data/raw/`

Berisi dataset asli (raw dataset) yang diperoleh dari sumber data sebelum dilakukan proses pengolahan.

| File | Keterangan |
|------|------------|
| `1778217759.csv` | Dataset asli harga pangan bulanan tingkat konsumen nasional. |
| `raw.md` | Dokumentasi mengenai dataset mentah. |

#### `data/processed/`

Berisi dataset hasil preprocessing yang digunakan pada proses pembangunan model machine learning.

| File | Keterangan |
|------|------------|
| `data_preprocessing_non_encode.csv` | Dataset setelah proses cleaning dan preprocessing sebelum encoding. |
| `data_preprocessing_encoded.csv` | Dataset akhir setelah encoding yang digunakan sebagai input model. |
| `processed.md` | Dokumentasi hasil preprocessing dataset. |

---

### 2. Folder `notebook/`

Berisi notebook yang digunakan selama proses penelitian.

| File | Keterangan |
|------|------------|
| `preprocessing_template.ipynb` | Notebook proses preprocessing dataset. |
| `eda_template.ipynb` | Notebook Exploratory Data Analysis (EDA). |
| `modeling_template.ipynb` | Notebook pembangunan, pelatihan, optimasi, dan evaluasi model machine learning. |

---

### 3. Folder `report/`

Berisi dokumen penelitian.

| File | Keterangan |
|------|------------|
| `Laporan Akhir Data Mining.pdf` | Laporan Akhir. |
| `Paper_Kelompok4.pdf` | Paper / jurnal. |
| `struktur-lampiran.md` | Dokumen penjelasan struktur lampiran repository. |

---

### 4. Folder `src/`

Berisi source code utama yang digunakan dalam penelitian.

| File | Keterangan |
|------|------------|
| `data_loader.py` | Modul untuk membaca dan memuat dataset. |
| `model.py` | Implementasi algoritma machine learning. |
| `utils.py` | Fungsi-fungsi pendukung penelitian. |
| `main.py` | Program utama untuk menjalankan proses penelitian. |
| `main_notebook.ipynb` | Notebook implementasi dari source code. |

---

### 5. File Pendukung

| File | Keterangan |
|------|------------|
| `requirements.txt` | Daftar library Python yang diperlukan untuk menjalankan penelitian. |
| `run.sh` | Script untuk menjalankan proyek secara otomatis. |
| `README.md` | Dokumentasi utama repository |

---

## Alur Kerja

Proses dilakukan melalui tahapan berikut:

1. Menggunakan dataset mentah pada folder `data/raw/`.
2. Melakukan preprocessing sehingga menghasilkan dataset pada folder `data/processed/`.
3. Melakukan eksplorasi data (EDA) menggunakan notebook.
4. Melatih model **Linear Regression** dan **Random Forest Regression**.
5. Melakukan optimasi hyperparameter.
6. Mengevaluasi model menggunakan metrik **MAE**, **RMSE**, **MAPE**, dan **R² Score**.
7. Menyusun laporan dan paper yang terdapat pada folder `report/`.

---
