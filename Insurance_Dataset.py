import pandas as pd

# Baca CSV
data = pd.read_csv('machine_learning/insurance.csv')

# --- Step 1: Tampilkan info & tipe data ---
print("=== INFO DATA ===")
print(data.info())
print("\n=== DESKRIPSI NUMERIK ===")
print(data.describe())

# --- Step 2: Set multi-index (misal date + cash_type kalau ada) ---
# Kalau CSV ada kolom 'date' dan 'cash_type', aktifkan baris ini:
# data.set_index(['date', 'cash_type'], inplace=True)

# --- Step 3: Handle missing values ---
if data.isnull().sum().sum() > 0:
    print("\n=== MISSING VALUES TERDETEKSI ===")
    print(data.isnull().sum())
    # Contoh: hapus baris yang punya missing
    data.dropna(inplace=True)
    print("Missing values sudah dihapus.")

# --- Step 4: Filter / contoh agregasi ---
# Misal lihat total money per cash_type
if 'cash_type' in data.columns and 'money' in data.columns:
    print("\n=== TOTAL MONEY PER CASH TYPE ===")
    total_money = data.groupby('cash_type')['money'].sum()
    print(total_money)

# --- Step 5: Tampilkan 5 baris pertama yang sudah bersih & siap dipakai ---
print("\n=== DATA SAMPLE ===")
print(data.head())
