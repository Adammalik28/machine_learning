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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB()),
        ]
    )
    return pipeline