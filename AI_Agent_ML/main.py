# main.py

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Using a more conservative value to prevent performance issues

import pandas as pd
from sklearn.impute import SimpleImputer
from utils import load_dataset, jalankan_automl, simpan_model, tanya_simpan_model
from pipeline import (
    split_data,
    pilih_model,
    latih_dan_evaluasi,
    apply_scaling
)
from feature_engineering import automated_feature_engineering
from mod_hyperparameter import hyperparameter_tuning, custom_tuning, handle_imbalance
from mod_visualization import tampilkan_eda, plot_learning_curve

def main():
    print("=== 🤖AI AGENT FOR MACHINE LEARNING🤖 ===\n")

    # Load dataset & dapatkan pilihan user
    X, y, target_names, task_type, pilihan_dataset = load_dataset()

    # Cek dan handle missing values (sama seperti sebelumnya)
    if X.isnull().sum().sum() > 0:
        print("[INFO] Ditemukan missing values di dataset.")
        print("[WARNING] Model tidak bisa jalan kalau ada missing values. Akan ditangani otomatis.")
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        print("[INFO] Missing values sudah ditangani dengan mean.")
    else:
        print("[INFO] Tidak ditemukan missing values. Data aman.")

    # Tampilkan Exploratory Data Analysis
    tampilkan_eda(X, y, target_names)

    # AUTOMATED FEATURE ENGINEERING
    print("\n=== Feature Engineering ===")
    feature_eng = input("Apakah ingin melakukan feature engineering? [y/n]: ")
    if feature_eng.lower() == 'y':
        print("[INFO] Mulai feature engineering...")
        X = automated_feature_engineering(X)
        # Tampilkan Exploratory Data Analysis
        tampilkan_eda(X, y, target_names)
    else:
        print("[INFO] Melewati feature engineering.")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, task_type=task_type)

    # Handle imbalance
    print("\n=== Balancing data ===")
    print("[INFO] Mengecek dan menangani imbalance pada data training...")
    X_train, y_train, use_class_weight = handle_imbalance(X_train, y_train, method="auto")
    print(f"[INFO] use_class_weight = {use_class_weight}")

    # Tanya scaling
    print("\n=== Scaling data ===")
    scale = input("Apakah ingin melakukan scaling? [y/n]: ")
    if scale.lower() == 'y':
        X_train, X_test, scaler_name = apply_scaling(X_train, X_test)
        print(f"\nData sudah discaling dengan {scaler_name}.")
    else:
        scaler_name = "Tanpa Scaling"

    # Pilih model
    print("\n=== Pilih model ===")
    model, model_name = pilih_model(task_type)
    print("\n=== Custom Hyperparameter Tuning ===")
    custom_tune = input("Apakah ingin melakukan custom hyperparameter tuning? [y/n]: ").lower()

    if custom_tune == 'y':
        model = custom_tuning(model_name, X_train, y_train)

        latih_dan_evaluasi(
            model, X_train, X_test, y_train, y_test,
            task_type=task_type,
            target_names=target_names,
            judul=f"{model_name} ({scaler_name})",
            use_cv=True,
            cv=5
        )
    else:
        print("[INFO] Melewati custom hyperparameter tuning.\n")

    # Variabel model hasil tuning otomatis
    best_model = None
    tuning_done = False

    # Tanya hyperparameter tuning otomatis
    print('=== Hyperparameter Tuning Otomatis ===')
    tuning = input("Apakah ingin melakukan hyperparameter tuning otomatis? [y/n]: ").lower()
    if tuning == 'y':
        metode_tuning = input("Pilih metode tuning: 1. GridSearchCV  2. RandomizedSearchCV\nMasukkan nomor [1/2]: ").strip()
        if metode_tuning == "1":  # GridSearch
            best_model = hyperparameter_tuning(
                model_name,
                X_train,
                y_train,
                task_type=task_type,
                search_type="grid",
                class_weight_handling=use_class_weight  # ===== UPDATE =====
            )
            tuning_done = True
        elif metode_tuning == "2":  # RandomizedSearch
            best_model = hyperparameter_tuning(
                model_name,
                X_train,
                y_train,
                task_type=task_type,
                search_type="random",
                class_weight_handling=use_class_weight  # ===== UPDATE =====
            )
            tuning_done = True
        else:
            print("[WARNING] Pilihan metode tuning tidak valid, melewati tuning.")

        if best_model is not None:
            latih_dan_evaluasi(
                best_model, X_train, X_test, y_train, y_test,
                task_type=task_type,
                target_names=target_names,
                judul=f"{model_name} (Tuned)"
            )
            model = best_model  # Update model jika berhasil tuning
        else:
            print("[INFO] Hyperparameter tuning gagal atau tidak dilakukan.")
    else:
        print("\n[INFO] Tidak melakukan hyperparameter tuning otomatis.")

    # Cek learning curve
    print("\n=== Learning Curve For Tuned Model ===")
    cek_lc = input("Mau cek overfitting atau underfitting gak? [y/n]: ").lower()
    if cek_lc == 'y':
        print("[INFO] Membuat learning curve...")
        plot_learning_curve(
            model, X_train, y_train, task_type,
            title=f"Learning Curve - {model_name}"
        )
        # Setelah cek learning curve, tanya lagi opsi tuning otomatis (loop)
        tuning2 = input("Apakah ingin melakukan hyperparameter tuning otomatis? [y/n]: ").lower()
        if tuning2 == 'y':
            metode_tuning2 = input("Pilih metode tuning: 1. GridSearchCV  2. RandomizedSearchCV\nMasukkan nomor [1/2]: ").strip()
            if metode_tuning2 == '2':
                best_model2 = hyperparameter_tuning(model_name, X_train, y_train, task_type, search_type="random", class_weight_handling=use_class_weight)
            else:
                best_model2 = hyperparameter_tuning(model_name, X_train, y_train, task_type, search_type="grid", class_weight_handling=use_class_weight)

            if best_model2 is not None:
                latih_dan_evaluasi(
                    best_model2, X_train, X_test, y_train, y_test,
                    task_type=task_type,
                    target_names=target_names,
                    judul=f"{model_name} (Tuned - After LC)"
                )
                model = best_model2  # Update model jika berhasil tuning
                best_model = best_model2
                tuning_done = True
            else:
                print("[INFO] Hyperparameter tuning gagal atau tidak dilakukan.")
        else:
            print("\nTidak melakukan hyperparameter tuning otomatis setelah learning curve.")

    # AutoML
    best_automl_model = None
    automl_model_name = None

    print("\n=== AutoML ===")
    automl = input("Mau coba AutoML juga? [y/n]: ").strip().lower()
    if automl == "y":
        print("[INFO] Menjalankan AutoML untuk mencari model terbaik secara otomatis...")
        best_automl_model, automl_model_name = jalankan_automl(X_train, y_train, task_type)

        # Evaluate AutoML model
        latih_dan_evaluasi(
            best_automl_model, X_train, X_test, y_train, y_test,
            task_type=task_type,
            target_names=target_names,
            judul=f"{automl_model_name} (AutoML)"
        )
        model = best_automl_model  # Update main model with AutoML result

        # learning curve for AutoML model
        cek_lc_automl = input("\nMau cek learning curve dari model AutoML ini? [y/n]: ").strip().lower()
        if cek_lc_automl == 'y':
            print("[INFO] Membuat learning curve untuk model AutoML...")
            plot_learning_curve(
                model, X_train, y_train, task_type,
                title=f"Learning Curve - {automl_model_name} (AutoML)"
            )
        else:
            print("Melanjutkan tanpa menampilkan learning curve AutoML.")

    # simpan model
    models_dict = {
        "initial": (model, f"{model_name} ({scaler_name})"),
    }

    if tuning_done and best_model is not None:
        models_dict["tuned"] = (best_model, f"{model_name} (Tuned)")

    if best_automl_model is not None:
        models_dict["automl"] = (best_automl_model, f"{automl_model_name} (AutoML)")
    tanya_simpan_model(models_dict, simpan_model)

    print("\n=== 🤖Thank you using AI Agent🤖 ===\n")

if __name__ == "__main__":
    main()
