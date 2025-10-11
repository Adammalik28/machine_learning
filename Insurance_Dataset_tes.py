# ==========================================================
# 💡 PROGRAM ANALISIS DATA ASURANSI + PREDIKSI NAIVE BAYES
# ==========================================================
# Penulis: Adam Rizki, Faouza Adicha, & Update by ChatGPT
# Deskripsi:
# Program ini memuat dataset asuransi, menampilkan visualisasi interaktif,
# serta melatih berbagai varian model Naive Bayes untuk memprediksi status perokok (smoker).
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- KONFIGURASI GLOBAL ---
FILE_PATH = 'insurance.csv'  # Lokasi file CSV
TARGET_COLUMN = 'charges'  # Kolom target analisis
NUMERICAL_COLS = ['age', 'bmi', 'children', 'charges']
CATEGORICAL_COLS = ['sex', 'smoker', 'region']


# ==========================================================
# 🧩 FUNGSI PENGOLAHAN DATA
# ==========================================================
def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """Memuat dataset dari file CSV."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset berhasil dimuat dari '{filepath}'.")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di '{filepath}'. Pastikan file CSV ada di direktori yang sama.")
        return None


def summarize_dataframe(df: pd.DataFrame, df_name: str = "DataFrame") -> None:
    """Menampilkan ringkasan informasi dan statistik DataFrame."""
    print(f"\n{'='*20} RINGKASAN: {df_name.upper()} {'='*20}")
    print(f"Bentuk Data: {df.shape[0]} baris, {df.shape[1]} kolom")
    print("\n--- Informasi Tipe Data ---")
    df.info()
    print("\n--- Statistik Deskriptif (Numerik) ---")
    print(df.describe().round(2))
    print("\n--- 5 Baris Pertama ---")
    print(df.head())
    print(f"{'='*58}\n")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani nilai yang hilang dalam DataFrame."""
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️ Ditemukan {missing_count} nilai hilang. Menghapus baris terkait...")
        df_cleaned = df.dropna()
        print(f"✅ Nilai yang hilang telah dihapus. Sisa baris: {len(df_cleaned)}.")
        return df_cleaned
    print("👍 Tidak terdapat nilai hilang dalam dataset.")
    return df


# ==========================================================
# 📊 FUNGSI VISUALISASI
# ==========================================================
def plot_distributions(df: pd.DataFrame, columns: List[str], plot_type: str) -> None:
    """Membuat visualisasi distribusi untuk kolom numerik atau kategorikal."""
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


def plot_smoker_vs_charges(df: pd.DataFrame, target: str) -> None:
    """Membuat boxplot untuk membandingkan biaya antara perokok dan bukan perokok."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='smoker', y=target, data=df, palette='muted')
    plt.title(f'Perbandingan {target.capitalize()} antara Perokok dan Bukan Perokok', fontsize=14)
    plt.xlabel('Status Perokok', fontsize=12)
    plt.ylabel(target.capitalize(), fontsize=12)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: List[str]) -> None:
    """Membuat heatmap korelasi untuk variabel numerik."""
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[num_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Heatmap Korelasi Variabel Numerik', fontsize=14)
    plt.show()


def show_avg_charges_by_region(df: pd.DataFrame, target: str) -> None:
    """Menampilkan rata-rata biaya asuransi per wilayah."""
    print("\n--- Rata-rata Biaya Asuransi per Wilayah ---")
    avg_charges = df.groupby('region')[target].mean().round(2).sort_values(ascending=False)
    print(avg_charges)
    print("-" * 45)


# ==========================================================
# 🤖 MODEL NAIVE BAYES
# ==========================================================
def run_naive_bayes(df: pd.DataFrame):
    """Menjalankan berbagai varian algoritma Naive Bayes untuk prediksi 'smoker'."""
    print("\n🚀 Memulai pelatihan model Naive Bayes untuk prediksi 'smoker'...\n")

    data = df.copy()

    # Encode kolom kategorikal
    label_encoders = {}
    for col in ['sex', 'region', 'smoker']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Fitur dan target
    X = data[['age', 'bmi', 'children', 'sex', 'region']]
    y = data['smoker']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pilih tipe Naive Bayes
    print("Pilih tipe model Naive Bayes:")
    print("1. GaussianNB (default, cocok untuk data kontinu)")
    print("2. MultinomialNB (cocok untuk data diskrit / count)")
    print("3. BernoulliNB (cocok untuk data biner 0/1)")
    print("4. ComplementNB (varian Multinomial untuk data tidak seimbang)")
    choice = input("Masukkan pilihan (1–4): ")

    if choice == '2':
        model = MultinomialNB()
        model_name = "MultinomialNB"
    elif choice == '3':
        model = BernoulliNB()
        model_name = "BernoulliNB"
    elif choice == '4':
        model = ComplementNB()
        model_name = "ComplementNB"
    else:
        model = GaussianNB()
        model_name = "GaussianNB"

    # Latih model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi hasil
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model yang digunakan: {model_name}")
    print(f"🎯 Akurasi Model: {acc:.2f}\n")
    print("📊 Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=['Non-Smoker', 'Smoker']))
    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ==========================================================
# 🧭 MENU INTERAKTIF
# ==========================================================
def show_menu():
    print("\n--- MENU ANALISIS DATA ASURANSI ---")
    print("1. Tampilkan Distribusi Variabel Numerik")
    print("2. Tampilkan Distribusi Variabel Kategorikal")
    print("3. Tampilkan Perbandingan Biaya (Perokok vs. Bukan Perokok)")
    print("4. Tampilkan Heatmap Korelasi Variabel Numerik")
    print("5. Tampilkan Rata-rata Biaya per Wilayah")
    print("6. Tampilkan Ulang Ringkasan Data")
    print("7. Jalankan Model Naive Bayes untuk Prediksi 'Smoker'")
    print("8. Keluar")
    print("-" * 45)


def main():
    """Fungsi utama untuk menjalankan alur analisis."""
    data = load_data(FILE_PATH)
    if data is None:
        return

    summarize_dataframe(data, "Data Awal")
    cleaned_data = handle_missing_values(data)

    while True:
        show_menu()
        choice = input("Pilih opsi (1-8): ")

        if choice == '1':
            plot_distributions(cleaned_data, NUMERICAL_COLS, 'numerical')
        elif choice == '2':
            plot_distributions(cleaned_data, CATEGORICAL_COLS, 'categorical')
        elif choice == '3':
            plot_smoker_vs_charges(cleaned_data, TARGET_COLUMN)
        elif choice == '4':
            plot_correlation_heatmap(cleaned_data, NUMERICAL_COLS)
        elif choice == '5':
            show_avg_charges_by_region(cleaned_data, TARGET_COLUMN)
        elif choice == '6':
            summarize_dataframe(cleaned_data, "Data Bersih")
        elif choice == '7':
            run_naive_bayes(cleaned_data)
        elif choice == '8':
            print("\nTerima kasih telah menggunakan program analisis ini. 👋")
            break
        else:
            print("\n❌ Pilihan tidak valid. Masukkan angka 1–8.")


# ==========================================================
# 🚀 EKSEKUSI PROGRAM
# ==========================================================
if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    main()
