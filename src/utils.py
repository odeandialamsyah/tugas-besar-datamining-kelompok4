# src/utils.py

"""
utils.py

Modul ini berisi fungsi-fungsi bantu seperti visualisasi dan evaluasi metrik model (regresi dan klasifikasi), serta EDA (Exploratory Data Analysis).
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(6, 4), title="Confusion Matrix"):
    """
    Menampilkan confusion matrix sebagai heatmap.
    (Untuk klasifikasi)
    
    Parameters:
        y_true (array-like): Nilai target sebenarnya
        y_pred (array-like): Nilai prediksi
        labels (list): Label klasifikasi
        figsize (tuple): Ukuran gambar
        title (str): Judul grafik
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

def plot_regression_results(y_true, y_pred, figsize=(12, 4)):
    """
    Menampilkan visualisasi hasil regresi (actual vs predicted, dan residuals).
    
    Parameters:
        y_true (array-like): Nilai sebenarnya
        y_pred (array-like): Nilai prediksi
        figsize (tuple): Ukuran gambar
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_classification_report(y_true, y_pred):
    """
    Menampilkan classification report dalam format teks.
    (Untuk klasifikasi)
    
    Parameters:
        y_true (array-like): Nilai target sebenarnya
        y_pred (array-like): Nilai prediksi
    """
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))

def print_regression_report(y_true, y_pred):
    """
    Menampilkan regression report dengan metrik evaluasi.
    (Untuk regresi)
    
    Parameters:
        y_true (array-like): Nilai target sebenarnya
        y_pred (array-like): Nilai prediksi
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print("=== REGRESSION EVALUATION METRICS ===")
    print("="*50)
    print(f"Mean Absolute Error (MAE):        {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE):   {rmse:.4f}")
    print(f"Mean Absolute Percentage Error:   {mape:.4f}%")
    print(f"R² Score:                         {r2:.4f}")
    print("="*50 + "\n")
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def print_best_model_summary(model_name, tuning_method, metrics):
    """
    Menampilkan summary best model seperti di modeling_template.
    
    Parameters:
        model_name (str): Nama model
        tuning_method (str): Metode tuning yang digunakan
        metrics (dict): Dictionary berisi MAE, RMSE, MAPE, R2
    """
    print("\n" + "="*70)
    print("BEST OVERALL MODEL")
    print("="*70)
    print(f"Model          : {model_name}")
    print(f"Tuning Method  : {tuning_method}")
    print(f"MAE            : {metrics['MAE']:.4f}")
    print(f"RMSE           : {metrics['RMSE']:.4f}")
    print(f"MAPE           : {metrics['MAPE']:.4f}%")
    print(f"R2             : {metrics['R2']:.4f}")
    print("="*70 + "\n")

def evaluate_model(y_true, y_pred, labels=None):
    """
    Mengevaluasi model dengan menampilkan confusion matrix dan classification report.
    (Untuk klasifikasi)
    
    Parameters:
        y_true (array-like): Nilai target sebenarnya
        y_pred (array-like): Nilai prediksi
        labels (list): Label klasifikasi
    """
    plot_confusion_matrix(y_true, y_pred, labels=labels)
    print_classification_report(y_true, y_pred)
