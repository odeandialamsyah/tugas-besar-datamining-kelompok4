# src/data_loader.py

"""
data_loader.py

Module ini digunakan untuk memuat dataset dari direktori `data/` baik dari folder raw maupun processed.
"""

import pandas as pd
import os

# Folder path relatif dari root repository
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

def load_csv(filename, processed=False):
    """
    Memuat file CSV dari folder data.
    
    Parameters:
        filename (str): Nama file (contoh: 'data.csv')
        processed (bool): Jika True, load dari folder processed, else dari raw.
    
    Returns:
        pd.DataFrame: Dataframe dari file yang dimuat
    """
    folder = PROCESSED_DATA_PATH if processed else RAW_DATA_PATH
    file_path = os.path.join(folder, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    
    return pd.read_csv(file_path)

def load_raw_data(filepath):
    """
    Memuat file CSV dari folder raw data dengan path relatif.
    
    Parameters:
        filepath (str): Path relatif file (contoh: 'data/raw/dataset.csv')
    
    Returns:
        pd.DataFrame: Dataframe dari file yang dimuat, atau None jika file tidak ada
    """
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"Warning: File tidak ditemukan di {filepath}")
            return None
    except Exception as e:
        print(f"Error membaca file: {e}")
        return None

def load_processed_data(filepath):
    """
    Memuat file CSV dari folder processed data dengan path relatif.
    
    Parameters:
        filepath (str): Path relatif file (contoh: 'data/processed/dataset.csv')
    
    Returns:
        pd.DataFrame: Dataframe dari file yang dimuat, atau None jika file tidak ada
    """
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"Warning: File tidak ditemukan di {filepath}")
            return None
    except Exception as e:
        print(f"Error membaca file: {e}")
        return None

def preview_data(df, rows=5):
    """
    Menampilkan preview awal dari dataframe
    
    Parameters:
        df (pd.DataFrame): Dataframe yang akan dipreview
        rows (int): Jumlah baris untuk ditampilkan
    """
    print(df.head(rows))

# Contoh pemanggilan (hapus saat produksi):
if __name__ == "__main__":
    try:
        # Contoh memuat data raw
        df = load_raw_data("data/raw/1778217759.csv")  # Sesuaikan nama file
        if df is not None:
            preview_data(df)
    except Exception as e:
        print(f"Error: {e}")
