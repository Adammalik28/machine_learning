import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Baca CSV ---
data = pd.read_csv('machine_learning/insurance.csv')

# --- Step 1: Info & Statistik ---
print("=== INFO DATA ===")
print(data.info())
print("\n=== DESKRIPSI NUMERIK ===")
print(data.describe())
print("\n=== DATA SAMPLE ===")
print(data.head())

# --- Step 2: Handle missing values (jika ada) ---
if data.isnull().sum().sum() > 0:
    print("\n=== MISSING VALUES TERDETEKSI ===")
    print(data.isnull().sum())
    data.dropna(inplace=True)
    print("Missing values sudah dihapus.")

# --- Step 3: Visualisasi distribusi kolom numerik ---
numerical_cols = ['age', 'bmi', 'children', 'charges']
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()

# --- Step 4: Visualisasi kolom kategorikal ---
categorical_cols = ['sex', 'smoker', 'region']
plt.figure(figsize=(12, 4))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 3, i)
    sns.countplot(x=col, data=data, palette='pastel')
    plt.title(f'Count {col}')
plt.tight_layout()
plt.show()

# --- Step 5: Boxplot charges per smoker ---
plt.figure(figsize=(6,4))
sns.boxplot(x='smoker', y='charges', data=data, palette='Set2')
plt.title('Charges berdasarkan status smoker')
plt.show()

# --- Step 6: Korelasi numerik ---
plt.figure(figsize=(6,5))
sns.heatmap(data[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasi antar kolom numerik')
plt.show()

# --- Step 7: Groupby sederhana (opsional) ---
total_charges_region = data.groupby('region')['charges'].sum()
print("\n=== TOTAL CHARGES PER REGION ===")
print(total_charges_region)