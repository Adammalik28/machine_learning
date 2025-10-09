import pandas as pd  # Library untuk manipulasi dan analisis data
import matplotlib.pyplot as plt  # Library untuk visualisasi data
import seaborn as sns  # Library untuk visualisasi statistik yang estetis
from typing import List, Optional  # Mengganti | dengan Optional untuk kompatibilitas


# --- KONFIGURASI GLOBAL ---
FILE_PATH = 'insurance.csv'  # Lokasi file CSV
TARGET_COLUMN = 'charges'  # Kolom target untuk analisis
NUMERICAL_COLS = ['age', 'bmi', 'children', 'charges']  # Kolom numerik
CATEGORICAL_COLS = ['sex', 'smoker', 'region']  # Kolom kategorikal


def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath (str): Path menuju file CSV.

    Returns:
        Optional[pd.DataFrame]: DataFrame jika berhasil dimuat, atau None jika file tidak ditemukan.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset berhasil dimuat dari '{filepath}'.")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di '{filepath}'. Harap pastikan file CSV berada di direktori yang sama.")
        return None


def summarize_dataframe(df: pd.DataFrame, df_name: str = "DataFrame") -> None:
    """
    Menampilkan ringkasan informasi dan statistik DataFrame.

    Args:
        df (pd.DataFrame): DataFrame yang akan diringkas.
        df_name (str): Nama deskriptif untuk DataFrame.
    """
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
    """
    Menangani nilai yang hilang dalam DataFrame.

    Strategi saat ini: menghapus baris dengan nilai hilang.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame yang telah bersih dari nilai hilang.
    """
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️ Ditemukan {missing_count} nilai hilang. Menghapus baris terkait...")
        df_cleaned = df.dropna()
        print(f"✅ Nilai yang hilang telah dihapus. Sisa baris: {len(df_cleaned)}.")
        return df_cleaned
    
    print("👍 Tidak terdapat nilai hilang dalam dataset.")
    return df


def plot_distributions(df: pd.DataFrame, columns: List[str], plot_type: str) -> None:
    """
    Membuat visualisasi distribusi untuk kolom yang dipilih.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data.
        columns (List[str]): Daftar kolom yang akan divisualisasikan.
        plot_type (str): Jenis plot ('numerical' atau 'categorical').
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
    """Menampilkan dan menganalisis rata-rata biaya per wilayah."""
    print("\n--- Rata-rata Biaya Asuransi per Wilayah ---")
    avg_charges = df.groupby('region')[target].mean().round(2).sort_values(ascending=False)
    print(avg_charges)
    print("-" * 45)


def show_menu():
    """Menampilkan menu pilihan untuk pengguna."""
    print("\n--- MENU VISUALISASI DATA ASURANSI ---")
    print("1. Tampilkan Distribusi Variabel Numerik")
    print("2. Tampilkan Distribusi Variabel Kategorikal")
    print("3. Tampilkan Perbandingan Biaya (Perokok vs. Bukan Perokok)")
    print("4. Tampilkan Heatmap Korelasi Variabel Numerik")
    print("5. Tampilkan Rata-rata Biaya per Wilayah")
    print("6. Tampilkan Ulang Ringkasan Data")
    print("7. Keluar")
    print("-" * 41)


def main():
    """
    Fungsi utama untuk menjalankan alur analisis data secara interaktif.
    """
    data = load_data(FILE_PATH)
    if data is None:
        return

    summarize_dataframe(data, "Data Awal")
    cleaned_data = handle_missing_values(data)

    while True:
        show_menu()
        choice = input("Pilih opsi (1-7): ")

        if choice == '1':
            print("\nMembuat plot distribusi variabel numerik...")
            plot_distributions(cleaned_data, NUMERICAL_COLS, 'numerical')
        elif choice == '2':
            print("\nMembuat plot distribusi variabel kategorikal...")
            plot_distributions(cleaned_data, CATEGORICAL_COLS, 'categorical')
        elif choice == '3':
            print("\nMembuat plot perbandingan biaya perokok...")
            plot_smoker_vs_charges(cleaned_data, TARGET_COLUMN)
        elif choice == '4':
            print("\nMembuat heatmap korelasi...")
            plot_correlation_heatmap(cleaned_data, NUMERICAL_COLS)
        elif choice == '5':
            show_avg_charges_by_region(cleaned_data, TARGET_COLUMN)
        elif choice == '6':
            summarize_dataframe(cleaned_data, "Data Bersih")
        elif choice == '7':
            print("\nTerima kasih telah menggunakan program analisis ini. Sampai jumpa!")
            break
        else:
            print("\n❌ Pilihan tidak valid. Silakan masukkan angka antara 1 hingga 7.")


if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    main()
