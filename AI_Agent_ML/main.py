# main.py
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from utils import load_dataset, jalankan_automl, simpan_model, tanya_simpan_model
from pipeline import split_data, pilih_model, latih_dan_evaluasi, apply_scaling
from feature_engineering import automated_feature_engineering
from mod_hyperparameter import hyperparameter_tuning, custom_tuning, handle_imbalance, get_model_instance
from mod_visualization import tampilkan_eda, plot_learning_curve

def set_loky_max_cpu(max_limit=16):
    cpu_count = os.cpu_count() or 1  # fallback kalau os.cpu_count() None
    max_cpu = min(cpu_count, max_limit)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max_cpu)
    print(f"Detected CPU cores: {cpu_count}, setting LOKY_MAX_CPU_COUNT to {max_cpu}")

def main():
    print("=== 🤖 AI AGENT FOR MACHINE LEARNING 🤖 ===\n")
    set_loky_max_cpu(max_limit=8)  # Set max CPU cores for parallel processing

    # Load dataset and get user choices
    X, y, target_names, task_type, pilihan_dataset = load_dataset()

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("[INFO] Ditemukan missing values di dataset.")
        print("[WARNING] Model tidak bisa jalan kalau ada missing values. Akan ditangani otomatis.")
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        print("[INFO] Missing values sudah ditangani dengan mean.")
    else:
        print("[INFO] Tidak ditemukan missing values. Data aman.")

    # Exploratory Data Analysis
    tampilkan_eda(X, y, target_names)

    # Feature Engineering
    print("\n=== Feature Engineering ===")
    feature_eng = input("Apakah ingin melakukan feature engineering? [y/n]: ")
    if feature_eng.lower() == 'y':
        print("[INFO] Mulai feature engineering...")
        X = automated_feature_engineering(X)
        tampilkan_eda(X, y, target_names)
    else:
        print("\n[INFO] Melewati feature engineering.")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, task_type=task_type)

    # Handle class imbalance
    print("\n=== Balancing data ===")
    print("[INFO] Mengecek dan menangani imbalance pada data training...")
    X_train, y_train, use_class_weight = handle_imbalance(X_train, y_train, method="auto")
    print(f"[INFO] use_class_weight = {use_class_weight}")

    # Data scaling
    print("\n=== Scaling data ===")
    scale = input("Apakah ingin melakukan scaling? [y/n]: ")
    if scale.lower() == 'y':
        X_train, X_test, scaler_name = apply_scaling(X_train, X_test)
        print(f"\nData sudah discaling dengan {scaler_name}.")
    else:
        scaler_name = "Tanpa Scaling"
        print("\n[INFO] Melewati scaling, data tetap dalam bentuk aslinya.")

    # Model selection and training
    print("\n=== Pilih model ===")
    model, model_name = pilih_model(task_type)
    # Langsung evaluasi model baseline
    print(f"\n=== Evaluasi Model {model_name} (Baseline) ===")
    latih_dan_evaluasi(
        model, X_train, X_test, y_train, y_test,
        task_type=task_type,
        target_names=target_names,
        judul=f"{model_name} (Baseline)"
    )

    # Automatic hyperparameter tuning
    best_model = None
    tuning_done = False

    print('\n=== Hyperparameter Tuning Otomatis ===')
    tuning = input("Apakah ingin melakukan hyperparameter tuning otomatis? [y/n]: ").lower()
    if tuning == 'y':
        metode_tuning = input(
            "Pilih metode tuning: 1. GridSearchCV  2. RandomizedSearchCV\nMasukkan nomor [1/2]: ").strip()

        if metode_tuning in ["1", "2"]:
            search_type = "grid" if metode_tuning == "1" else "random"
            best_model = hyperparameter_tuning(
                model_name,
                X_train,
                y_train,
                task_type=task_type,
                search_type=search_type,
                class_weight_handling=use_class_weight
            )
            tuning_done = True
        else:
            print("[WARNING] Pilihan metode tuning tidak valid, melewati tuning.")

        if best_model is not None:
            # Evaluate tuned model
            latih_dan_evaluasi(
                best_model, X_train, X_test, y_train, y_test,
                task_type=task_type,
                target_names=target_names,
                judul=f"{model_name} (Tuned)"
            )
            model = best_model  # Update model with tuned version

            # Learning curve analysis
            print("\n=== Learning Curve Analysis ===")
            cek_lc = input("Mau cek learning curve untuk analisis overfitting/underfitting? [y/n]: ").lower()
            if cek_lc == 'y':
                print("[INFO] Membuat learning curve...")
                plot_learning_curve(
                    model, X_train, y_train, task_type,
                    title=f"Learning Curve - {model_name} (Tuned)"
                )
        else:
            print("[INFO] Hyperparameter tuning gagal atau tidak dilakukan.")
    else:
        print("\n[INFO] Tidak melakukan hyperparameter tuning otomatis.")


    # Custom hyperparameter tuning
    print("\n=== Custom Hyperparameter Tuning ===")
    custom_tune = input("Apakah ingin melakukan custom hyperparameter tuning? [y/n]: ").lower()
    while custom_tune not in ['y', 'n']:
        custom_tune = input("Mohon masukkan hanya 'y' atau 'n': ").lower()
    if custom_tune == 'y':
        tuned_model = custom_tuning(model_name, X_train, y_train)
        if tuned_model is None:
            print("[INFO] Model tuning untuk model ini belum tersedia, menggunakan model baseline.")
            tuned_model = get_model_instance(model_name, y_train)
        latih_dan_evaluasi(
            tuned_model, X_train, X_test, y_train, y_test,
            task_type=task_type,
            target_names=target_names,
            judul=f"{model_name} ({scaler_name}) - Tuning",
            use_cv=True,
            cv=5
        )
    else:
        print("\n[INFO] Melewati custom hyperparameter tuning.\n")

    # AutoML option
    best_automl_model = None
    automl_model_name = None

    print("=== AutoML ===")
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

        # Learning curve for AutoML model
        cek_lc_automl = input("\nMau cek learning curve dari model AutoML ini? [y/n]: ").strip().lower()
        if cek_lc_automl == 'y':
            print("[INFO] Membuat learning curve untuk model AutoML...")
            plot_learning_curve(
                model, X_train, y_train, task_type,
                title=f"Learning Curve - {automl_model_name} (AutoML)"
            )
    else:
        print("\n[INFO] Melewati AutoML.")

    # Prepare models for saving
    models_dict = {
        "initial": (model, f"{model_name} ({scaler_name})"),
    }

    if tuning_done and best_model is not None:
        models_dict["tuned"] = (best_model, f"{model_name} (Tuned)")

    if best_automl_model is not None:
        models_dict["automl"] = (best_automl_model, f"{automl_model_name} (AutoML)")

    # Ask about model saving
    tanya_simpan_model(models_dict, simpan_model)

    print("\n=== 🤖 Thank you for using AI Agent 🤖 ===\n")


if __name__ == "__main__":
    main()