# pipeline.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# function split data
def split_data(X, y, task_type="classification"):
    if task_type == "classification":
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        return train_test_split(X, y, test_size=0.2, random_state=42)

# def scaling
def apply_scaling(X_train, X_test):
    print("\nPilih metode scaling:")
    print("1. StandardScaler")
    print("2. MinMaxScaler")
    print("3. RobustScaler")
    scaler_choice = input("Masukkan pilihan [1/2/3]: ")

    if scaler_choice == '1':
        scaler = StandardScaler()
    elif scaler_choice == '2':
        scaler = MinMaxScaler()
    elif scaler_choice == '3':
        scaler = RobustScaler()
    else:
        print("Pilihan tidak valid. Default ke StandardScaler.")
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler.__class__.__name__

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from tpot import TPOTClassifier, TPOTRegressor

# function latih dan evaluasi model
def latih_dan_evaluasi(model, X_train, X_test, y_train, y_test,
                       task_type="classification", target_names=None, judul="",
                       use_cv=False, cv=5,
                       automl=False):
    if automl:
        print("[INFO] Memulai AutoML dengan TPOT, mohon tunggu...")
        if task_type == "classification":
            automl_model = TPOTClassifier(generations=5, population_size=20, cv=cv, verbosity=2, random_state=42)
        else:
            automl_model = TPOTRegressor(generations=5, population_size=20, cv=cv, verbosity=2, random_state=42)

        automl_model.fit(X_train, y_train)
        print(f"[INFO] AutoML selesai, score terbaik di training: {automl_model.score(X_train, y_train):.4f}")

        best_pipeline = automl_model.fitted_pipeline_
        model = best_pipeline  # ganti model ke pipeline terbaik TPOT

        print("\nPipeline terbaik dari AutoML:")
        print(best_pipeline)
    else:
        if use_cv and task_type == "classification":
            print(f"[INFO] Melakukan cross-validation dengan cv={cv} ...")
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            print(f"Accuracy CV: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

            # Cross val prediksi untuk confusion matrix
            y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
            target_names = [f"class {label}" for label in np.unique(y_train)]
            if target_names is not None:
                print(classification_report(y_train, y_pred_cv, target_names=target_names))
            else:
                print(classification_report(y_train, y_pred_cv))

            cm_cv = confusion_matrix(y_train, y_pred_cv)
            disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=target_names)
            disp_cv.plot(cmap=plt.cm.Greens)
            plt.title(f"Confusion Matrix CV - {judul}")
            plt.show()

        # Fit model ke data training
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n=== Evaluasi Model {judul} (Test Set) ===")
    if task_type == "classification":
        if target_names is not None:
            print(classification_report(y_test, y_pred, target_names=target_names))
        else:
            print(classification_report(y_test, y_pred))

        # Tampilkan confusion matrix test set
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {judul}")
        plt.show()

    else:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")


# function pilih model
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

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None

def pilih_model(task_type):
    print("Pilih model:")

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
    elif task_type == "regression":
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
    else:
        raise ValueError("Jenis task tidak dikenali.")

    while True:
        # Tampilkan pilihan
        for i, name in enumerate(model_options.keys(), start=1):
            print(f"{i}. {name}")

        pilihan_str = input(f"Masukkan nomor model [1-{len(model_options)}]: ")
        try:
            pilihan = int(pilihan_str)
        except ValueError:
            print("[ERROR] Input harus berupa angka.")
            continue

        if 1 <= pilihan <= len(model_options):
            model_name = list(model_options.keys())[pilihan - 1]
            model = model_options[model_name]
            if model is None:
                print(f"[ERROR] {model_name} tidak tersedia. Pastikan library terkait sudah di-install.")
                continue  # ulangi input
            return model, model_name
        else:
            print("[ERROR] Pilihan tidak valid. Silakan coba lagi.")
