# ==========================================================
# 🤖 NAIVE BAYES PRO+ EDITION — DENGAN GRID SEARCH & VISUALISASI
# ==========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# --- KONFIGURASI GLOBAL ---
FILE_PATH = "insurance.csv"
TARGET_COLUMN = "charges"
NUMERICAL_COLS = ["age", "bmi", "children"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


# ==========================================================
# 📦 FUNGSI: LOAD & PERSIAPAN DATA
# ==========================================================
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Memuat dan menyiapkan dataset Insurance untuk klasifikasi"""
    df = pd.read_csv(filepath)

    # Buat label kategori berdasarkan distribusi charges
    df["charges_category"] = pd.qcut(
        df[TARGET_COLUMN], q=3, labels=["Rendah", "Sedang", "Tinggi"]
    )
    df = df.drop(columns=[TARGET_COLUMN])

    print(f"✅ Data dimuat ({len(df)} baris, {len(df.columns)} kolom)")
    print("📊 Distribusi kategori:")
    print(df["charges_category"].value_counts(normalize=True).round(2) * 100, "\n")

    return df


# ==========================================================
# 🧱 PIPELINE: PREPROCESSING + MODEL
# ==========================================================
def build_pipeline() -> Pipeline:
    """Membangun pipeline preprocessing + GaussianNB"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_COLS),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB())
        ]
    )
    return pipeline


# ==========================================================
# 🔍 GRID SEARCH: TUNING PARAMETER
# ==========================================================
def tune_hyperparameters(model: Pipeline, X_train, y_train) -> Pipeline:
    """Melakukan tuning var_smoothing GaussianNB menggunakan GridSearchCV"""
    param_grid = {"classifier__var_smoothing": np.logspace(-9, -3, 10)}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n🏆 Parameter terbaik: {grid_search.best_params_}")
    print(f"📈 Akurasi Cross-Validation: {grid_search.best_score_:.2%}\n")

    return grid_search.best_estimator_
