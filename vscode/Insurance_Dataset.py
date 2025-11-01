# ==========================================================
# Contributors:
# - Adam Rizki Putra Khoiri
# - M Krisna Nugraha
# - Faouza Dio Ardicha
# ==========================================================

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


# --- GRID SEARCH TUNING ---
def tune_hyperparameters(model: Pipeline, X_train, y_train):
    """
    Melakukan pencarian parameter terbaik menggunakan GridSearchCV
    """
    # Parameter GaussianNB yang bisa di-tune: var_smoothing
    # (semakin besar -> smoothing lebih tinggi, lebih stabil tapi bisa menurunkan akurasi)
    param_grid = {
        "classifier__var_smoothing": np.logspace(-9, -3, 7)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n🏆 Parameter terbaik: {grid_search.best_params_}")
    print(f"📈 Akurasi cross-val terbaik: {grid_search.best_score_:.2%}\n")

    return grid_search.best_estimator_


# --- EVALUASI MODEL ---
def evaluate_model(model: Pipeline, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Akurasi Model (Test): {acc:.2%}")
    print("\n📊 Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=["Rendah", "Sedang", "Tinggi"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rendah", "Sedang", "Tinggi"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Gaussian Naive Bayes (Tuned)")
    plt.show()


# --- MAIN ---
def main():
    df = load_and_prepare_data(FILE_PATH)

    X = df.drop("charges_category", axis=1)
    y = df["charges_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_pipeline()

    print("⚙️  Melakukan tuning hyperparameter...")
    best_model = tune_hyperparameters(model, X_train, y_train)

    print("✅ Model terbaik ditemukan dan siap dievaluasi!")
    evaluate_model(best_model, X_test, y_test)


if __name__ == "__main__":
    sns.set(style="whitegrid")
    plt.rcParams["figure.dpi"] = 100
    main()
