# utils.py (Updated)

from sklearn import datasets
import pandas as pd

# function for load dataset scikitlearn / file csv
def load_dataset():
    print("\n=== Pilihan Sumber Dataset ===")
    sumber = input("0. Keluar dari program\n1. Dari scikit-learn (built-in)\n2. Dari file CSV lokal\nMasukkan pilihan [0/1/2]: ")

    if sumber == '0':
        print("Keluar dari program. Sampai jumpa!")
        exit()

    if sumber == '1':
        print("\n=== Pilihan Dataset Scikit-learn ===")
        print("0. Keluar")
        print("1. Iris (klasifikasi)")
        print("2. Wine (klasifikasi)")
        print("3. Breast Cancer (klasifikasi)")
        print("4. Diabetes (regresi)")
        print("5. Digits (klasifikasi)")
        print("6. California Housing (regresi)")
        print("7. MNIST (klasifikasi)")
        pilihan_dataset = int(input("Masukkan nomor dataset [0-7]: "))

        if pilihan_dataset == 0:
            print("Keluar dari program. Sampai jumpa!")
            exit()

        if pilihan_dataset == 1:
            data = datasets.load_iris()
            task_type = "classification"
        elif pilihan_dataset == 2:
            data = datasets.load_wine()
            task_type = "classification"
        elif pilihan_dataset == 3:
            data = datasets.load_breast_cancer()
            task_type = "classification"
        elif pilihan_dataset == 4:
            data = datasets.load_diabetes()
            task_type = "regression"
        elif pilihan_dataset == 5:
            data = datasets.load_digits()
            task_type = "classification"
        elif pilihan_dataset == 6:
            data = datasets.fetch_california_housing()
            task_type = "regression"
        elif pilihan_dataset == 7:
            print("[WARNING] Loading MNIST dataset may take a long time. Please be patient...")
            data = datasets.fetch_openml('mnist_784', version=1, as_frame=True)
            task_type = "classification"
        else:
            print("Pilihan dataset tidak valid.")
            exit()

        X = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, 'feature_names') else None)
        y = pd.Series(data.target)
        target_names = data.target_names if hasattr(data, 'target_names') else None

        print(f"\nMemuat dataset {data.DESCR.splitlines()[0]}...")

        return X, y, target_names, task_type, pilihan_dataset

    elif sumber == '2':
        path = input("Masukkan path file CSV: ")
        df = pd.read_csv(path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        target_names = None
        task_type = input("Masukkan jenis task [classification/regression]: ").lower()
        pilihan_dataset = 0
        return X, y, target_names, task_type, pilihan_dataset

    else:
        print("Pilihan sumber dataset tidak valid.")
        exit()


# function AutoML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None
    print("[WARNING] XGBoost not installed. XGBoost models will not be available.")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None
    print("[WARNING] LightGBM not installed. LightGBM models will not be available.")

from sklearn.model_selection import cross_val_score
import numpy as np

def jalankan_automl(X_train, y_train, task_type):
    # Split data train/test kecil untuk keperluan evaluasi AutoML
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print("\n[INFO] Menjalankan AutoML untuk mencari model terbaik secara otomatis...")

    # Daftar model sesuai task
    if task_type == "classification":
        model_options = {
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Support Vector Machine": SVC(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier() if XGBClassifier else None,
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1) if LGBMClassifier else None,
        }
    else:
        model_options = {
            "Linear Regression": LinearRegression(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Support Vector Regression": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor() if XGBRegressor else None,
            "LightGBM": LGBMRegressor(random_state=42, verbose=-1) if LGBMRegressor else None,
        }

    best_score = -np.inf
    best_model = None
    best_model_name = None

    print("\n[INFO] Evaluasi model AutoML dengan cross-validation (cv=5):")
    for name, model in model_options.items():
        if model is None:
            continue
        try:
            scores = cross_val_score(model, X_tr, y_tr, cv=5)
            mean_score = np.mean(scores)
            print(f"  > {name}: Mean CV Score = {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"  > {name}: Error saat evaluasi - {str(e)}")

    print(f"\n[INFO] Model terbaik dari AutoML: {best_model_name} (CV Score = {best_score:.4f})")
    best_model.fit(X_tr, y_tr)

    return best_model, best_model_name

# function simpan model
import joblib
import os
def simpan_model(model, nama_file):
    if not os.path.exists('models'):
        os.makedirs('models')

    # Simpan model ke file
    path = os.path.join('models', nama_file)
    joblib.dump(model, path)
    print(f"Model berhasil disimpan ke {path}")

# function tanya model
def tanya_simpan_model(models_dict, simpan_model_func):
    if not models_dict:
        print("\n[INFO] Tidak ada model untuk disimpan.")
        return
    print("\n=== Simpan Model ===")
    print("Model yang tersedia untuk disimpan:")
    for i, (key, (_, name)) in enumerate(models_dict.items(), start=1):
        print(f"{i}. {name} [{key}]")

    simpan = input("Apakah ingin menyimpan model? (y/n): ").strip().lower()
    if simpan != 'y':
        print("Membatalkan penyimpanan model.")
        return

    pilihan = input(f"Pilih nomor model yang ingin disimpan (1-{len(models_dict)}): ").strip()
    try:
        pilihan_int = int(pilihan)
        if 1 <= pilihan_int <= len(models_dict):
            key_pilih = list(models_dict.keys())[pilihan_int - 1]
            model_terpilih, model_terpilih_name = models_dict[key_pilih]
            print(f"Kamu memilih menyimpan model: {model_terpilih_name}")
            nama_file = input("Masukkan nama file untuk menyimpan model (contoh: model_terlatih.joblib): ").strip()
            if not nama_file:
                nama_file = "model_terlatih.joblib"
            simpan_model_func(model_terpilih, nama_file)
            print(f"[INFO] Model '{model_terpilih_name}' berhasil disimpan di '{nama_file}'")
        else:
            print("Pilihan nomor tidak valid. Tidak menyimpan model.")
    except ValueError:
        print("Input tidak valid. Tidak menyimpan model.")


# Contoh penggunaan:
if __name__ == "__main__":
    X, y, target_names = load_dataset()
