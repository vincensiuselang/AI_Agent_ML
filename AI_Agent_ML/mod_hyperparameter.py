# mod_hyperparameter.py

# function param grid
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

def get_param_grid(model_name, y=None, use_class_weight=False):
    is_classification = None
    if y is not None:
        # Tentukan classification jika jumlah kelas <= 20 dan tipe data integer/unsigned
        is_classification = len(np.unique(y)) <= 20 and y.dtype.kind in {'i', 'u'}

    if model_name == "K-Nearest Neighbors":
        if is_classification:
            return KNeighborsClassifier(), {
                'n_neighbors': list(range(3, 16, 2)),
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'leaf_size': [20, 30, 40, 50],
                'algorithm': ['auto']
            }
        else:
            return KNeighborsRegressor(), {
                'n_neighbors': list(range(3, 16, 2)),
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'leaf_size': [20, 30, 40, 50],
                'algorithm': ['auto']
            }

    elif model_name == "Random Forest":
        if is_classification:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2'],
            }
            param_grid['class_weight'] = ['balanced'] if use_class_weight else [None]
            return RandomForestClassifier(random_state=42), param_grid
        else:
            return RandomForestRegressor(random_state=42), {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }

    elif model_name == "Logistic Regression":
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': np.logspace(-4, 4, 5),
            'l1_ratio': [0, 0.5, 1],  # hanya dipakai jika penalty='elasticnet'
        }
        param_grid['class_weight'] = ['balanced'] if use_class_weight else [None]
        return LogisticRegression(max_iter=10000, solver='saga', random_state=42), param_grid

    elif model_name == "Support Vector Machine":
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3],
        }
        param_grid['class_weight'] = ['balanced'] if use_class_weight else [None]
        return SVC(probability=True, random_state=42), param_grid

    elif model_name == "Support Vector Regression":
        return SVR(), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3]
        }

    elif model_name == "Decision Tree":
        if is_classification:
            param_grid = {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini', 'entropy']
            }
            param_grid['class_weight'] = ['balanced'] if use_class_weight else [None]
            return DecisionTreeClassifier(random_state=42), param_grid
        else:
            return DecisionTreeRegressor(random_state=42), {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['squared_error', 'absolute_error']
            }

    elif model_name == "Naive Bayes":
        return GaussianNB(), {
            'var_smoothing': np.logspace(-9, -6, 4)
        }

    elif model_name == "Gradient Boosting":
        if is_classification:
            return GradientBoostingClassifier(random_state=42), {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        else:
            return GradientBoostingRegressor(random_state=42), {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }

    elif model_name == "XGBoost":
        if is_classification:
            scale_pos = 1
            if y is not None:
                unique, counts = np.unique(y, return_counts=True)
                if len(unique) == 2:
                    scale_pos = counts[0] / counts[1]
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [1, scale_pos]
            }
        else:
            return XGBRegressor(random_state=42), {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

    elif model_name == "LightGBM":
        if is_classification:
            param_grid = {
                'verbose': -1,
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'min_child_weight': [31, 83],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            if use_class_weight:
                param_grid['class_weight'] = ['balanced']
            return LGBMClassifier(random_state=42), param_grid
        else:
            return LGBMRegressor(random_state=42), {
                'verbose': -1,
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'min_child_weight': [31, 83],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
    elif model_name == "Linear Regression":
        return LinearRegression(), {}

    else:
        raise ValueError(f"Model '{model_name}' belum didukung.")

# function handle imbalance
from sklearn.utils.multiclass import type_of_target
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter

def handle_imbalance(X_train, y_train, method="auto", force_balancing=None):
    print("[INFO] Cek apakah kelas pada data training tidak seimbang...")

    target_type = type_of_target(y_train)
    if target_type not in ['binary', 'multiclass']:
        print("  > Target bukan klasifikasi, jadi tidak perlu balancing.")
        return X_train, y_train, False

    class_counts = Counter(y_train)
    print(f"  > Jumlah sampel per kelas: {dict(class_counts)}")
    if force_balancing is None:
        jawab = input("Mau lakukan balancing data supaya kelas seimbang? (y/n): ").strip().lower()
        if jawab != 'y':
            print("  > Oke, kita lanjut tanpa balancing.")
            return X_train, y_train, True
    elif force_balancing is False:
        print("  > Balancing tidak dilakukan (force_balancing=False).")
        return X_train, y_train, True
    else:
        print("  > Melanjutkan balancing (force_balancing=True).")

    if method == "auto":
        min_class_count = min(class_counts.values())

        if min_class_count <= 1:
            print("  > Ada kelas dengan sangat sedikit data, SMOTE tidak bisa dipakai.")
            print("  > Jadi kita pakai RandomOverSampler untuk menyeimbangkan data.")
            ros = RandomOverSampler()
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
            print("  > Balancing selesai dengan RandomOverSampler.")
            return X_resampled, y_resampled, False
        else:
            print("  > Balancing data dengan SMOTE, sabar ya...")
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print("  > Balancing selesai dengan SMOTE.")
            return X_resampled, y_resampled, False

    print("  > Metode balancing tidak dikenali, lanjut tanpa balancing.")
    return X_train, y_train, True



# hyperparameter tuning function
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
def hyperparameter_tuning(model_name, X_train, y_train, task_type="classification", search_type="grid", class_weight_handling=False):
    print(f"\n[INFO] Melakukan {search_type.title()}SearchCV untuk model: {model_name}")
    try:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Ambil model dan param_grid, dengan opsi class_weight jika diaktifkan
        model, param_grid = get_param_grid(model_name, y_train, use_class_weight=class_weight_handling)

        # Skor tergantung task
        scoring = make_scorer(accuracy_score) if task_type == "classification" else make_scorer(mean_squared_error, greater_is_better=False)

        # Use a more conservative value for n_jobs to prevent performance issues
        if search_type == "grid":
            search = GridSearchCV(model, param_grid, cv=cv, n_jobs=2, verbose=1, scoring=scoring)
        elif search_type == "random":
            search = RandomizedSearchCV(model, param_grid, cv=cv, n_jobs=2, verbose=1, scoring=scoring, n_iter=20, random_state=42)
        else:
            raise ValueError(f"search_type '{search_type}' tidak didukung. Pilih 'grid' atau 'random'.")

        search.fit(X_train, y_train)
        print(f"[INFO] Best params: {search.best_params_}")
        print(f"[INFO] Best score: {search.best_score_:.4f}")

        return search.best_estimator_

    except Exception as e:
        print(f"[ERROR] Gagal melakukan tuning: {e}")
        return None

# custom tuning function
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
def custom_tuning(model_name, X_train, y_train, task_type="classification"):
    print(f"\n[INFO] Custom Tuning untuk model: {model_name}")
    try:
        base_model, param_grid = get_param_grid(model_name, y_train)
        best_model = base_model
        best_score = -np.inf if task_type == "classification" else np.inf

        for param_name, param_values in param_grid.items():
            if param_name == 'class_weight' and param_values == [None]:
                continue  # skip if no class_weight options

            for val in param_values:
                setattr(best_model, param_name, val)
                best_model.fit(X_train, y_train)
                if task_type == "classification":
                    score = best_model.score(X_train, y_train)
                    if score > best_score:
                        best_score = score
                        best_params = {param_name: val}
                else:
                    # regression -> MSE
                    y_pred = best_model.predict(X_train)
                    score = -mean_squared_error(y_train, y_pred)
                    if score > best_score:
                        best_score = score
                        best_params = {param_name: val}

        print(f"[INFO] Best params (custom): {best_params}")
        print(f"[INFO] Best score (custom): {best_score:.4f}")
        return best_model

    except Exception as e:
        print(f"[ERROR] Gagal custom tuning: {e}")
        return None
