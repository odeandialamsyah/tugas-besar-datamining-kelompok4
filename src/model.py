# src/model.py

"""
model.py

Modul ini digunakan untuk melatih dan mengevaluasi model Time Series Regression.
Untuk prediksi harga komoditas bulan berikutnya.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def split_data_timeseries(df, target_column="target", test_size=0.2, date_column="Tanggal"):
    """
    Memisahkan fitur dan target dengan time-based split (penting untuk time series!).
    Tidak boleh menggunakan random split karena akan menyebabkan data leakage.

    Parameters:
        df (pd.DataFrame): Dataset lengkap (sudah diproses)
        target_column (str): Nama kolom target
        test_size (float): Proporsi data uji (0.2 = 80% train, 20% test)
        date_column (str): Nama kolom tanggal untuk sorting

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Pastikan data sudah di-sort berdasarkan tanggal
    if date_column in df.columns:
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Ambil tanggal unik
        unique_dates = sorted(df[date_column].unique())
        split_index = int(len(unique_dates) * (1 - test_size))
        split_date = unique_dates[split_index]
        
        # Split berdasarkan tanggal
        train_idx = df[date_column] < split_date
        test_idx = df[date_column] >= split_date
    else:
        # Jika tidak ada kolom tanggal, gunakan index
        split_idx = int(len(df) * (1 - test_size))
        train_idx = df.index < split_idx
        test_idx = df.index >= split_idx
    
    # Pisahkan fitur dan target
    X = df.drop(columns=[target_column, date_column] if date_column in df.columns else [target_column])
    y = df[target_column]
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return X_train, X_test, y_train, y_test

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Wrapper untuk kompatibilitas backward - gunakan split_data_timeseries untuk time series!
    
    Parameters:
        df (pd.DataFrame): Dataset lengkap
        target_column (str): Nama kolom target
        test_size (float): Proporsi data uji
        random_state (int): Seed random (diabaikan untuk time series)

    Returns:
        X_train, X_test, y_train, y_test
    """
    return split_data_timeseries(df, target_column, test_size)

def train_model(X_train, y_train, n_estimators=100, random_state=42, max_depth=None, 
                min_samples_split=2, min_samples_leaf=1, max_features="sqrt"):
    """
    Melatih model regresi untuk prediksi harga (Random Forest Regressor).

    Parameters:
        X_train: Fitur data latih
        y_train: Target data latih
        n_estimators (int): Jumlah pohon dalam random forest
        random_state (int): Seed random untuk reproducibility
        max_depth (int): Kedalaman maksimal pohon (None = unlimited)
        min_samples_split (int): Minimum samples untuk split node
        min_samples_leaf (int): Minimum samples di leaf node
        max_features (str): Jumlah fitur untuk split ("sqrt", "log2", None)

    Returns:
        model terlatih (RandomForestRegressor)
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi performa model regresi pada data uji.
    Menggunakan metrik: MAE, RMSE, R²

    Parameters:
        model: model yang sudah dilatih
        X_test: fitur data uji
        y_test: target data uji

    Returns:
        dict: Dictionary berisi metrik evaluasi
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "y_pred": y_pred
    }
    
    return results

def get_feature_importance(model, feature_names, top_n=10):
    """
    Dapatkan feature importance dari trained model.
    
    Parameters:
        model: trained model
        feature_names (list): Nama-nama fitur
        top_n (int): Top N fitur yang ditampilkan
    
    Returns:
        pd.DataFrame: DataFrame dengan feature importance
    """
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    return feature_imp_df.head(top_n)
