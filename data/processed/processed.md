# Processed Dataset

Folder ini berisi dataset hasil preprocessing yang telah melalui proses pembersihan, transformasi, dan feature engineering. Dataset pada folder ini digunakan sebagai input pada tahap pemodelan machine learning.

## Struktur Folder

```text
processed/
├── data_preprocessing_encoded.csv
├── data_preprocessing_non_encoded.csv
└── processed.md
```

## Isi Folder

| File | Deskripsi |
|------|-----------|
| `data_preprocessing_non_encoded.csv` | Dataset hasil preprocessing sebelum proses encoding variabel kategorikal. |
| `data_preprocessing_encoded.csv` | Dataset hasil preprocessing yang telah melalui proses encoding dan siap digunakan untuk proses pelatihan model machine learning. |

## Tahapan Preprocessing

Dataset pada folder ini dihasilkan melalui beberapa tahapan preprocessing, yaitu:

- Pembersihan nama kolom dan nilai teks.
- Konversi kolom harga menjadi format numerik.
- Penanganan missing value menggunakan interpolasi dan forward/backward fill.
- Encoding variabel bulan.
- Pembuatan kolom tanggal.
- Deteksi outlier menggunakan metode Interquartile Range (IQR).
- Feature engineering, meliputi:
  - Lag Feature (Lag-1, Lag-2, Lag-3)
  - Moving Average (3 dan 6 periode)
  - Perubahan Harga
  - Persentase Perubahan Harga
  - Target Harga Bulan Berikutnya
- One-Hot Encoding pada variabel komoditas.
- Penentuan fitur dan target untuk proses pemodelan.
