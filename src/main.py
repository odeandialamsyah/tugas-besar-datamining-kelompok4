# src/main.py

"""
Main pipeline for Time Series Regression - Prediksi Harga Komoditas

WORKFLOW:
1. Run notebook/preprocessing_template.ipynb terlebih dahulu untuk preprocessing
2. Script ini load data yang sudah diproses dari data/processed/
3. Hyperparameter tuning dengan GridSearchCV/RandomizedSearchCV
4. Evaluate model dengan metrik regresi (MAE, RMSE, R²)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from model import split_data_timeseries, evaluate_model, get_feature_importance
from utils import print_regression_report, plot_regression_results

def main():
    # Load processed data yang sudah dibuat di preprocessing_template.ipynb
    print("\n" + "="*60)
    print("LOADING PROCESSED DATA")
    print("="*60)
    
    processed_data_path = "data/processed/data_preprocessing_encoded.csv"
    
    if not os.path.exists(processed_data_path):
        print(f"❌ Error: File {processed_data_path} tidak ditemukan!")
        print("📋 Silakan jalankan notebook/preprocessing_template.ipynb terlebih dahulu")
        return
    
    print(f"Loading data dari {processed_data_path}...")
    df = pd.read_csv(processed_data_path)
    print(f"✓ Data loaded! Shape: {df.shape}")
    
    # Cek kolom yang diperlukan
    print("\n" + "="*60)
    print("PREPARE FEATURES")
    print("="*60)
    
    print(f"Kolom tersedia: {len(df.columns)} columns")
    
    # Tentukan target column
    target_col = None
    if "Target_Harga_Bulan_Berikutnya" in df.columns:
        target_col = "Target_Harga_Bulan_Berikutnya"
    elif "target" in df.columns:
        target_col = "target"
    else:
        print("❌ Error: Kolom target tidak ditemukan!")
        return
    
    print(f"✓ Target column: {target_col}")
    
    # Tentukan fitur
    exclude_cols = [target_col, "Komoditas", "Tahun", "Bulan", "Tanggal", "Harga", "Outlier"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"✓ Total fitur: {len(feature_cols)}")
    
    # Drop missing values
    print("\n" + "="*60)
    print("TIME-BASED DATA SPLIT (80% train, 20% test)")
    print("="*60)
    
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"Data setelah drop missing: {df_clean.shape}")
    
    # Siapkan data untuk split
    date_col = "Tanggal" if "Tanggal" in df_clean.columns else None
    
    X_train, X_test, y_train, y_test = split_data_timeseries(
        df_clean[feature_cols + [target_col] + ([date_col] if date_col else [])],
        target_column=target_col,
        test_size=0.2,
        date_column=date_col
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    
    # Hyperparameter tuning dengan BayesSearchCV (lebih optimal dari RandomizedSearchCV)
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING (BayesSearchCV)")
    print("="*60)
    
    # Parameter space untuk Bayesian Search
    param_space = {
        "n_estimators": Integer(100, 500),
        "max_depth": Integer(5, 30),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2"])
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Use TimeSeriesSplit untuk time series
    tscv = TimeSeriesSplit(n_splits=5)
    
    bayes_search = BayesSearchCV(
        estimator=rf,
        search_spaces=param_space,
        n_iter=50,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )
    
    print("Tuning model dengan Bayesian Optimization (ini mungkin memakan waktu 3-5 menit)...")
    bayes_search.fit(X_train, y_train)
    
    best_params = bayes_search.best_params_
    print(f"\n✓ Tuning selesai!")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score (RMSE): {-bayes_search.best_score_:.4f}")
    
    # Train model dengan best parameters
    print("\n" + "="*60)
    print("TRAINING MODEL dengan Best Parameters")
    print("="*60)
    
    best_model = bayes_search.best_estimator_
    print("✓ Model berhasil dilatih dengan parameter optimal!")
    
    # Evaluate model
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = evaluate_model(best_model, X_test, y_test)
    y_pred = results["y_pred"]
    
    # Print metrik
    print_regression_report(y_test, y_pred)
    
    # Visualization
    print("\n" + "="*60)
    print("VISUALIZATION & FEATURE IMPORTANCE")
    print("="*60)
    
    plot_regression_results(y_test.values, y_pred)
    
    feature_importance = get_feature_importance(best_model, X_test.columns, top_n=10)
    print("\nTop 10 Feature Importance:")
    print(feature_importance)
    
    print("\n" + "="*60)
    print("✓ PIPELINE SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()
