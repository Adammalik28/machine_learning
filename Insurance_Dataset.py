import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --- KONFIGURASI ---
FILE_PATH = "insurance.csv"
TARGET_COLUMN = "charges"
NUMERICAL_COLS = ["age", "bmi", "children"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]


# --- FUNGSI PERSIAPAN DATA ---
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Mengubah target menjadi kategori klasifikasi
    df["charges_category"] = pd.qcut(df[TARGET_COLUMN], q=3, labels=["Rendah", "Sedang", "Tinggi"])
    df = df.drop(TARGET_COLUMN, axis=1)

    print(f"✅ Data dimuat ({len(df)} baris). Distribusi kategori:")
    print(df["charges_category"].value_counts(), "\n")

    return df


# --- PIPELINE ---
def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_COLS),
        ]
    )

<<<<<<< HEAD
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_naive_bayes(df: pd.DataFrame):
    """
    Menerapkan algoritma Naive Bayes untuk memprediksi status perokok.
    """
    print("\n🚀 Memulai pelatihan model Naive Bayes untuk prediksi 'smoker'...")

    # Salin data agar aman
    data = df.copy()

    # Encode kolom kategorikal
    label_encoders = {}
    for col in ['sex', 'region', 'smoker']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Pisahkan fitur (X) dan target (y)
    X = data[['age', 'bmi', 'children', 'sex', 'region']]
    y = data['smoker']

    # Bagi data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat model dan latih
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi hasil
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Akurasi Model: {acc:.2f}")
    print("\n📊 Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=['Non-Smoker', 'Smoker']))
    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def show_menu():
    """Menampilkan menu pilihan untuk pengguna."""
    print("\n--- MENU VISUALISASI DATA ASURANSI ---")
    print("1. Tampilkan Distribusi Variabel Numerik")
    print("2. Tampilkan Distribusi Variabel Kategorikal")
    print("3. Tampilkan Perbandingan Biaya (Perokok vs. Bukan Perokok)")
    print("4. Tampilkan Heatmap Korelasi Variabel Numerik")
    print("5. Tampilkan Rata-rata Biaya per Wilayah")
    print("6. Tampilkan Ulang Ringkasan Data")
    print("7. Jalankan Model Naive Bayes untuk Prediksi 'Smoker'")
    print("8. Keluar")
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
        choice = input("Pilih opsi (1-8): ")

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
            run_naive_bayes(cleaned_data)
        elif choice == '8':
            print("\nTerima kasih telah menggunakan program analisis ini. Sampai jumpa!")
            break
        else:
            print("\n❌ Pilihan tidak valid. Silakan masukkan angka antara 1 hingga 7.")


if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    main()
=======
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB()),
        ]
    )
    return pipeline
>>>>>>> 44cd5949a1b24767971fbfbd6a1238a1e455e7b5
