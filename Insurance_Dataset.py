# -*- coding: utf-8 -*-
"""
Skrip Analisis Data Eksplorasi (EDA) untuk Dataset Asuransi.

Skrip ini memuat data asuransi, melakukan pembersihan data dasar,
dan menghasilkan beberapa visualisasi kunci untuk memahami distribusi
dan hubungan antar variabel.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# --- KONFIGURASI ---
# Variabel global didefinisikan di sini agar mudah diubah.
FILE_PATH = 'machine_learning/insurance.csv'
TARGET_COLUMN = 'charges'
NUMERICAL_COLS = ['age', 'bmi', 'children', 'charges']
CATEGORICAL_COLS = ['sex', 'smoker', 'region']


def load_data(filepath: str) -> pd.DataFrame | None:
    """
    Memuat dataset dari file CSV yang ditentukan.

    Args:
        filepath (str): Path menuju file .csv.

    Returns:
        pd.DataFrame | None: DataFrame yang dimuat, atau None jika file tidak ditemukan.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset berhasil dimuat dari '{filepath}'.")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di '{filepath}'. Harap periksa path file.")
        return None


def summarize_dataframe(df: pd.DataFrame, df_name: str = "DataFrame") -> None:
    """
    Mencetak ringkasan informasi dan statistik dari DataFrame.

    Args:
        df (pd.DataFrame): DataFrame yang akan diringkas.
        df_name (str): Nama deskriptif untuk DataFrame (misal: "Data Awal").
    """
    print(f"\n{'='*20} RINGKASAN: {df_name.upper()} {'='*20}")
    print(f"Bentuk Data: {df.shape[0]} baris, {df.shape[1]} kolom")
    print("\n--- Info Tipe Data ---")
    df.info()
    print("\n--- Statistik Deskriptif (Numerik) ---")
    print(df.describe().round(2))
    print("\n--- Sampel 5 Data Teratas ---")
    print(df.head())
    print(f"{'='*50}\n")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memeriksa dan menangani nilai yang hilang dalam DataFrame.

    Saat ini, strategi yang digunakan adalah menghapus baris dengan nilai hilang.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame yang sudah bersih dari nilai yang hilang.
    """
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️ Ditemukan {missing_count} nilai yang hilang. Menghapus baris terkait...")
        df_cleaned = df.dropna()
        print(f"✅ Nilai yang hilang telah dihapus. Sisa baris: {len(df_cleaned)}.")
        return df_cleaned
    
    print("👍 Tidak ada nilai yang hilang dalam dataset.")
    return df


def plot_distributions(df: pd.DataFrame, columns: List[str], plot_type: str) -> None:
    """
    Membuat visualisasi distribusi untuk kolom yang diberikan.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data.
        columns (List[str]): Daftar nama kolom yang akan divisualisasikan.
        plot_type (str): Tipe plot ('numerical' atau 'categorical').
    """
    if plot_type == 'numerical':
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(columns, 1):
            plt.subplot(2, 2, i)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribusi Kolom {col.capitalize()}', fontsize=12)
        plt.suptitle('Distribusi Variabel Numerik', fontsize=16, y=1.02)
    
    elif plot_type == 'categorical':
        plt.figure(figsize=(14, 5))
        for i, col in enumerate(columns, 1):
            plt.subplot(1, 3, i)
            sns.countplot(x=col, data=df, palette='viridis', order=df[col].value_counts().index)
            plt.title(f'Distribusi Kolom {col.capitalize()}', fontsize=12)
        plt.suptitle('Distribusi Variabel Kategorikal', fontsize=16, y=1.02)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_core_analyses(df: pd.DataFrame, num_cols: List[str], target: str) -> None:
    """
    Membuat visualisasi analisis inti seperti boxplot dan heatmap korelasi.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data.
        num_cols (List[str]): Daftar kolom numerik untuk heatmap.
        target (str): Kolom target untuk analisis (misal: 'charges').
    """
    # 1. Boxplot untuk melihat pengaruh 'smoker' terhadap 'charges'
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='smoker', y=target, data=df, palette='muted')
    plt.title(f'Perbandingan {target.capitalize()} antara Perokok dan Bukan Perokok', fontsize=14)
    plt.xlabel('Status Perokok', fontsize=12)
    plt.ylabel(target.capitalize(), fontsize=12)
    plt.show()

    # 2. Heatmap korelasi untuk variabel numerik
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[num_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Heatmap Korelasi Variabel Numerik', fontsize=14)
    plt.show()


def main():
    """
    Fungsi utama untuk menjalankan alur kerja analisis data.
    """
    # Langkah 1: Memuat data
    data = load_data(FILE_PATH)
    if data is None:
        return  # Hentikan eksekusi jika data gagal dimuat

    # Langkah 2: Ringkasan data awal
    summarize_dataframe(data, "Data Awal")

    # Langkah 3: Menangani nilai yang hilang
    cleaned_data = handle_missing_values(data)

    # Langkah 4: Visualisasi distribusi
    print("\n--- MEMBUAT VISUALISASI DISTRIBUSI ---")
    plot_distributions(cleaned_data, NUMERICAL_COLS, 'numerical')
    plot_distributions(cleaned_data, CATEGORICAL_COLS, 'categorical')

    # Langkah 5: Visualisasi analisis inti
    print("\n--- MEMBUAT VISUALISASI ANALISIS INTI ---")
    plot_core_analyses(cleaned_data, NUMERICAL_COLS, TARGET_COLUMN)

    # Langkah 6: Analisis tambahan (contoh: groupby)
    print("\n--- ANALISIS TAMBAHAN ---")
    avg_charges_by_region = cleaned_data.groupby('region')[TARGET_COLUMN].mean().round(2).sort_values(ascending=False)
    print("Rata-rata Biaya Asuransi per Wilayah:")
    print(avg_charges_by_region)

    print("\n✅ Analisis selesai.")


if __name__ == "__main__":
    # Mengatur style visualisasi global
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100 # Meningkatkan resolusi plot
    
    main()