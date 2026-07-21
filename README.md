# 📊 Prediksi Harga Komoditas Pangan Bulanan Menggunakan Pendekatan Machine Learning Regression Berbasis Fitur Temporal

Repositori ini merupakan implementasi **Tugas Besar Mata Kuliah Data Mining** Program Studi D-IV Teknik Informatika, Universitas Logistik dan Bisnis Internasional (ULBI). Proyek ini bertujuan membangun model machine learning untuk memprediksi harga komoditas pangan bulanan menggunakan pendekatan **Machine Learning Regression** yang diperkaya dengan **Feature Engineering Temporal**.

---

## 👥 Anggota Kelompok

| Nama | NIM |
|------|------|
| Ode Andi Alamsyah | 714230032 |
| Nesya Salma Ramadhani | 714230028 |
| Indah Diva Gracia | 714230039 |
| Rendy Kamaluddin | 714230030 |

---

## 📌 Deskripsi Kasus

Fluktuasi harga komoditas pangan di Indonesia dipengaruhi oleh berbagai faktor, seperti kondisi cuaca, musim, inflasi, distribusi, serta permintaan pasar. Perubahan harga yang dinamis menyebabkan perlunya model prediksi yang mampu memperkirakan harga pada periode berikutnya.

Pada proyek ini dikembangkan model prediksi menggunakan algoritma **Linear Regression** dan **Random Forest Regression**. Untuk meningkatkan performa model dilakukan **Feature Engineering** berbasis data historis serta optimasi hyperparameter menggunakan **Grid Search**, **Random Search**, dan **Bayesian Optimization**.

---

## 📂 Dataset

| Informasi | Keterangan |
|-----------|------------|
| **Nama Dataset** | Rata-rata Harga Pangan Bulanan Tingkat Konsumen Nasional |
| **Sumber** | Data Indonesia (data.go.id) |
| **Periode** | Januari 2021 – April 2026 |
| **Jumlah Data** | 1.227 Record |
| **Jumlah Komoditas** | 47 Komoditas |
| **Jumlah Atribut** | 4 Atribut |

### Atribut Dataset

| Nama Atribut | Deskripsi |
|--------------|-----------|
| Komoditas | Nama komoditas pangan |
| Tahun | Tahun pencatatan |
| Bulan | Bulan pencatatan |
| Harga | Harga rata-rata komoditas |

---

## 📁 Struktur Proyek

```
tube_data_mining/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebook/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── modeling.ipynb
│
├── report/
│   ├── laporan Akhir Data Mining.docs
│   ├── laporan Akhir Data Mining.pdf
│   ├── Paper_Kelompok4.pdf
│   └── Paper_Kelompok4.docx
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

## ⚙️ Alur Pengerjaan

Tahapan yang dilakukan pada proyek ini meliputi:

1. Pengumpulan Dataset
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Engineering
5. Pembagian Dataset (Time-Based Split)
6. Pelatihan Model
7. Optimasi Hyperparameter
8. Evaluasi Model
9. Analisis Hasil

---

## 🧹 Data Preprocessing

Tahapan preprocessing yang dilakukan antara lain:

- Membersihkan nama kolom
- Membersihkan format harga
- Mengubah tipe data
- Menangani missing value
- Encoding bulan
- Membentuk atribut tanggal
- Deteksi outlier menggunakan IQR
- Standardisasi fitur menggunakan StandardScaler

---

## 🔧 Feature Engineering

Fitur tambahan yang digunakan pada penelitian ini meliputi:

- Lag-1
- Lag-2
- Lag-3
- Moving Average (MA-3)
- Moving Average (MA-6)
- Price Difference
- Percentage Change
- Target Harga Bulan Berikutnya
- One Hot Encoding Komoditas

---

## 🤖 Algoritma yang Digunakan

### Linear Regression

Digunakan sebagai model baseline untuk mengetahui kemampuan model regresi linear dalam memprediksi harga komoditas.

### Random Forest Regression

Digunakan sebagai model utama karena mampu menangkap hubungan nonlinier dan menghasilkan prediksi yang lebih stabil.

---

## ⚡ Hyperparameter Optimization

Optimasi Random Forest dilakukan menggunakan tiga metode:

- Grid Search
- Random Search
- Bayesian Optimization

---

## 📈 Evaluasi Model

Performa model dievaluasi menggunakan metrik:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score

Model terbaik dipilih berdasarkan:

- Nilai MAE terkecil
- Nilai RMSE terkecil
- Nilai MAPE terkecil
- Nilai R² tertinggi

---

## 🚀 Cara Menjalankan

### ✅ 1. Persiapkan Environment

Install dependensi:
```bash
pip install -r requirements.txt
```

### ✅ 2. Jalankan Pipeline

#### 💻 Via Terminal:
```bash
bash run.sh
```

#### 📒 Via Jupyter Notebook:
Buka dan jalankan:
```text
src/main_notebook.ipynb
```
---
## 🛠️ Library yang Digunakan

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Scikit-Optimize
- Jupyter Notebook

---

## 📊 Output Proyek

Output yang dihasilkan meliputi:

- Dataset hasil preprocessing
- Model Machine Learning
- Hasil optimasi hyperparameter
- Visualisasi Exploratory Data Analysis (EDA)
- Grafik hasil prediksi
- Evaluasi performa model

---

## 📄 Lisensi

Repositori ini dibuat sebagai bagian dari Tugas Besar Mata Kuliah **Data Mining** Program Studi D-IV Teknik Informatika Universitas Logistik dan Bisnis Internasional (ULBI). Seluruh isi repositori ini dapat digunakan sebagai referensi pembelajaran dengan tetap mencantumkan sumber.