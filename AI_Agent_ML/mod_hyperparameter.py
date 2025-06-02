# mod_hyperparameter.py
"""
This module provides functions for hyperparameter tuning and model selection.
It includes utilities for handling imbalanced datasets and custom parameter tuning.
"""

# Standard library imports
from collections import Counter

# Third-party imports
import numpy as np
from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error

# Model imports
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Imbalanced data handling
from imblearn.over_sampling import SMOTE, RandomOverSampler

def get_param_grid(model_name, y=None, use_class_weight=False):
    """
    Get a model instance and its parameter grid for hyperparameter tuning.

    Args:
        model_name (str): Name of the model to get parameters for
        y (numpy.ndarray, optional): Target variable to determine if classification or regression
        use_class_weight (bool, optional): Whether to use class_weight parameter for imbalanced data

    Returns:
        tuple: (model_instance, parameter_grid_dict)
    """
    is_classification = None
    if y is not None:
        # Determine if classification based on number of unique classes and data type
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

def handle_imbalance(X_train, y_train, method="auto", force_balancing=None):
    """
    Handle imbalanced datasets by applying oversampling techniques.

    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target variable
        method (str, optional): Method to use for balancing ('auto' or custom)
        force_balancing (bool, optional): Force balancing without asking user

    Returns:
        tuple: (X_resampled, y_resampled, is_imbalanced)
            - X_resampled: Resampled features
            - y_resampled: Resampled target
            - is_imbalanced: Boolean indicating if data is still imbalanced
    """
    print("[INFO] Checking if training data classes are imbalanced...")

    target_type = type_of_target(y_train)
    if target_type not in ['binary', 'multiclass']:
        print("  > Target is not classification, no balancing needed.")
        return X_train, y_train, False

    class_counts = Counter(y_train)
    print(f"  > Samples per class: {dict(class_counts)}")

    # Determine if balancing should be performed
    if force_balancing is None:
        user_response = input("Do you want to balance the data for equal class distribution? (y/n): ").strip().lower()
        if user_response != 'y':
            print("  > Continuing without balancing.")
            return X_train, y_train, True
    elif force_balancing is False:
        print("  > Balancing not performed (force_balancing=False).")
        return X_train, y_train, True
    else:
        print("  > Proceeding with balancing (force_balancing=True).")

    # Apply balancing method
    if method == "auto":
        min_class_count = min(class_counts.values())

        if min_class_count <= 1:
            print("  > Some classes have very few samples, SMOTE cannot be used.")
            print("  > Using RandomOverSampler for balancing instead.")
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
            print("  > Balancing completed with RandomOverSampler.")
            return X_resampled, y_resampled, False
        else:
            print("  > Balancing data with SMOTE, please wait...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print("  > Balancing completed with SMOTE.")
            return X_resampled, y_resampled, False

    print("  > Balancing method not recognized, continuing without balancing.")
    return X_train, y_train, True

def hyperparameter_tuning(model_name, X_train, y_train, task_type="classification", search_type="grid", class_weight_handling=False):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Args:
        model_name (str): Name of the model to tune
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target variable
        task_type (str, optional): Type of task ('classification' or 'regression')
        search_type (str, optional): Type of search ('grid' or 'random')
        class_weight_handling (bool, optional): Whether to use class_weight parameter

    Returns:
        object: Best estimator after hyperparameter tuning or None if tuning fails
    """
    print(f"\n[INFO] Performing {search_type.title()}SearchCV for model: {model_name}")
    try:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Get model and param_grid, with class_weight option if enabled
        model, param_grid = get_param_grid(model_name, y_train, use_class_weight=class_weight_handling)

        # Scoring depends on task type
        scoring = make_scorer(accuracy_score) if task_type == "classification" else make_scorer(mean_squared_error, greater_is_better=False)

        # Use a more conservative value for n_jobs to prevent performance issues
        if search_type == "grid":
            search = GridSearchCV(model, param_grid, cv=cv, n_jobs=2, verbose=1, scoring=scoring)
        elif search_type == "random":
            search = RandomizedSearchCV(model, param_grid, cv=cv, n_jobs=2, verbose=1, scoring=scoring, n_iter=20, random_state=42)
        else:
            raise ValueError(f"search_type '{search_type}' is not supported. Choose 'grid' or 'random'.")

        search.fit(X_train, y_train)
        print(f"[INFO] Best params: {search.best_params_}")
        print(f"[INFO] Best score: {search.best_score_:.4f}")

        return search.best_estimator_

    except Exception as e:
        print(f"[ERROR] Failed to perform tuning: {e}")
        return None

def get_model_instance(model_name, y=None):
    """
    Get a model instance based on model_name and target type.

    Args:
        model_name (str): Name of the model to instantiate
        y (numpy.ndarray, optional): Target variable to determine if classification or regression

    Returns:
        object: Model instance appropriate for the task type

    Raises:
        ValueError: If the model name is not supported
    """
    is_classification = None
    if y is not None:
        # Assume classification if number of classes <= 20 and integer/unsigned type
        is_classification = (len(np.unique(y)) <= 20) and (y.dtype.kind in {'i', 'u'})

    if model_name == "K-Nearest Neighbors":
        return KNeighborsClassifier() if is_classification else KNeighborsRegressor()

    elif model_name == "Random Forest":
        return RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)

    elif model_name == "Logistic Regression":
        return LogisticRegression(max_iter=10000, solver='saga', random_state=42)

    elif model_name == "Support Vector Machine":
        return SVC(probability=True, random_state=42)

    elif model_name == "Support Vector Regression":
        return SVR()

    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42) if is_classification else DecisionTreeRegressor(random_state=42)

    elif model_name == "Naive Bayes":
        return GaussianNB()

    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42) if is_classification else GradientBoostingRegressor(random_state=42)

    elif model_name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) if is_classification else XGBRegressor(random_state=42)

    elif model_name == "LightGBM":
        return LGBMClassifier(random_state=42) if is_classification else LGBMRegressor(random_state=42)

    elif model_name == "Linear Regression":
        return LinearRegression()

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")


# Dictionary of parameters that will be requested from the user for manual tuning
CUSTOM_PARAM_SPACE = {
    'K-Nearest Neighbors': ['n_neighbors', 'weights', 'p'],
    'Random Forest': ['n_estimators', 'max_depth', 'min_samples_split'],
    'Logistic Regression': ['C', 'penalty', 'solver'],
    'Support Vector Machine': ['C', 'kernel', 'gamma'],
    'Decision Tree': ['max_depth', 'min_samples_split'],
    'Naive Bayes': [],  # Usually doesn't have many parameters to tune
    'Gradient Boosting': ['n_estimators', 'learning_rate', 'max_depth'],
    'XGBoost': ['n_estimators', 'learning_rate', 'max_depth'],
    'LightGBM': ['n_estimators', 'learning_rate', 'num_leaves']
}


def parse_weights_input(weights_input):
    """
    Parse user input for weights parameter in KNN models.

    Args:
        weights_input (str): User input string with comma-separated values

    Returns:
        list: List of valid weight options ('uniform', 'distance')
    """
    # Helper function to parse weights input (1=uniform, 2=distance)
    mapping = {'1': 'uniform', '2': 'distance'}
    weights_list = []
    for w in weights_input.split(','):
        w = w.strip()
        if w in mapping:
            weights_list.append(mapping[w])
        elif w in ['uniform', 'distance']:
            weights_list.append(w)
    return list(set(weights_list)) or ['uniform']


def custom_tuning(model_name, X_train, y_train):
    """
    Custom hyperparameter tuning for specific models with user input.

    Currently supports K-Nearest Neighbors (KNN) and XGBoost (classification).

    Args:
        model_name (str): Name of the model to tune
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target variable

    Returns:
        object: Best estimator after hyperparameter tuning or None if model not supported
    """
    if model_name.lower() in ['k-nearest neighbors', 'knn']:
        print("[INFO] Custom tuning for model: K-Nearest Neighbors")

        # Get manual parameter input from user
        n_neighbors_input = input("Enter values for 'n_neighbors' (comma-separated): ")
        weights_input = input("Enter values for 'weights' (comma-separated, 1=uniform, 2=distance): ")
        p_input = input("Enter values for 'p' (comma-separated): ")

        # Parse input into lists with appropriate types
        try:
            n_neighbors_list = [int(x.strip()) for x in n_neighbors_input.split(',') if x.strip().isdigit()]
            if not n_neighbors_list:
                n_neighbors_list = [5]  # default fallback
        except Exception:
            n_neighbors_list = [5]

        weights_list = parse_weights_input(weights_input)

        try:
            p_list = [int(x.strip()) for x in p_input.split(',') if x.strip().isdigit()]
            if not p_list:
                p_list = [2]  # default fallback
        except Exception:
            p_list = [2]

        param_grid = {
            'n_neighbors': n_neighbors_list,
            'weights': weights_list,
            'p': p_list,
        }

        print(f"[INFO] Parameter grid: {param_grid}")

        try:
            knn = KNeighborsClassifier()
            search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=2)
            search.fit(X_train, y_train)

            print(f"[INFO] Best params: {search.best_params_}")
            print(f"[INFO] Best score: {search.best_score_:.4f}")

            return search.best_estimator_
        except Exception as e:
            print(f"[ERROR] Failed to perform KNN tuning: {e}")
            return KNeighborsClassifier()

    elif model_name.lower() == 'xgboost':
        print("[INFO] Custom tuning for model: XGBoost")

        # Get manual parameter input from user
        n_estimators_input = input("Enter values for 'n_estimators' (comma-separated): ")
        max_depth_input = input("Enter values for 'max_depth' (comma-separated): ")
        learning_rate_input = input("Enter values for 'learning_rate' (comma-separated, example: 0.01,0.1): ")
        subsample_input = input("Enter values for 'subsample' (comma-separated, example: 0.8,1.0): ")
        colsample_bytree_input = input("Enter values for 'colsample_bytree' (comma-separated, example: 0.8,1.0): ")

        # Parse input
        try:
            n_estimators_list = [int(x.strip()) for x in n_estimators_input.split(',') if x.strip().isdigit()]
            if not n_estimators_list:
                n_estimators_list = [100]
        except Exception:
            n_estimators_list = [100]

        try:
            max_depth_list = [int(x.strip()) for x in max_depth_input.split(',') if x.strip().isdigit()]
            if not max_depth_list:
                max_depth_list = [3]
        except Exception:
            max_depth_list = [3]

        try:
            learning_rate_list = [float(x.strip()) for x in learning_rate_input.split(',') if x.strip()]
            if not learning_rate_list:
                learning_rate_list = [0.1]
        except Exception:
            learning_rate_list = [0.1]

        try:
            subsample_list = [float(x.strip()) for x in subsample_input.split(',') if x.strip()]
            if not subsample_list:
                subsample_list = [1.0]
        except Exception:
            subsample_list = [1.0]

        try:
            colsample_bytree_list = [float(x.strip()) for x in colsample_bytree_input.split(',') if x.strip()]
            if not colsample_bytree_list:
                colsample_bytree_list = [1.0]
        except Exception:
            colsample_bytree_list = [1.0]

        param_grid = {
            'n_estimators': n_estimators_list,
            'max_depth': max_depth_list,
            'learning_rate': learning_rate_list,
            'subsample': subsample_list,
            'colsample_bytree': colsample_bytree_list,
        }

        print(f"[INFO] Parameter grid: {param_grid}")

        try:
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
            search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=2)
            search.fit(X_train, y_train)

            print(f"[INFO] Best params: {search.best_params_}")
            print(f"[INFO] Best score: {search.best_score_:.4f}")

            return search.best_estimator_
        except Exception as e:
            print(f"[ERROR] Failed to perform XGBoost tuning: {e}")
            return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    else:
        print(f"[WARNING] Custom tuning for model '{model_name}' is not available.")
        return None
