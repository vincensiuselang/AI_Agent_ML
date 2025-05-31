# mod_visualization.py

# function for displaying EDA
from sklearn.utils.multiclass import type_of_target
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import numpy as np

# function for tampilkan EDA
def tampilkan_eda(X, y, target_names):
    print("\n=== EDA (Exploratory Data Analysis) ===")

    # 1. Informasi Dasar
    print("\n1. Informasi Dasar:")
    print(f"Dimensi Data: {X.shape}")
    print("\n5 Data Teratas:")
    print(X.head())

    # 2. Distribusi Target
    print("\n2. Distribusi Target:")
    target_type = type_of_target(y)

    if target_type in ['binary', 'multiclass']:
        target_counts = y.value_counts()
        if target_names is not None and len(target_names) == target_counts.shape[0]:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=y, hue=y, palette='viridis', legend=False)
            plt.title('Distribusi Kelas Target')

            # Hanya set xticks jika jumlah label dan label-nya cocok
            plt.xticks(ticks=range(len(target_counts)), labels=target_names, rotation=45)

            plt.show()
        else:
            # fallback kalau target_names tidak cocok, tapi masih klasifikasi
            plt.figure(figsize=(8, 4))
            sns.countplot(x=y, palette='viridis')
            plt.title('Distribusi Kelas Target')
            plt.show()
    else:  # Untuk regresi
        print(y.describe())
        plt.figure(figsize=(8, 4))
        sns.histplot(y, kde=True, color='skyblue')
        plt.title('Distribusi Target (Regresi)')
        plt.xlabel('Target')
        plt.ylabel('Frekuensi')
        plt.show()

    # 3. Analisis Fitur
    print("\n3. Analisis Fitur:")
    print(X.describe())

    # 4. Distribusi Fitur
    print("\n4. Distribusi Fitur:")
    num_cols = X.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        X[num_cols].hist(bins=20, figsize=(15, 10), edgecolor='black')
        plt.suptitle("Distribusi Fitur Numerik", fontsize=16)
        plt.tight_layout()
        plt.show()

    # 5. Analisis Korelasi
    print("\n5. Analisis Korelasi:")
    if len(num_cols) > 1:
        corr = X[num_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, linewidths=.5)
        plt.title("Heatmap Korelasi Antar Fitur")
        plt.show()

    # 6. Visualisasi PCA (hanya untuk klasifikasi)
    if target_type in ['binary', 'multiclass'] and len(num_cols) >= 2:
        print("\n6. Visualisasi PCA (2D):")
        plot_pca(X[num_cols], y, target_names)


# function for plot by PCA
def plot_pca(X, y, target_names):
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['target'] = y.values

        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(data=df_pca, x='PC1', y='PC2',
                                  hue='target', palette='viridis',
                                  alpha=0.7, s=100)

        plt.title(f"Visualisasi PCA (Variansi: {pca.explained_variance_ratio_.sum():.2%})")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

        handles, labels = scatter.get_legend_handles_labels()
        if target_names is not None and len(target_names) == len(labels):
            plt.legend(handles, target_names, title="Kelas",
                       bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend(title="Kelas", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Gagal membuat plot PCA: {str(e)}")

# function confusion matrix
def tampilkan_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (angka):")
    print(cm)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.show()

# function plot learning curve
def plot_learning_curve(model, X, y, task_type, title="Learning Curve {model}"):
    print("[INFO] Plotting Learning Curve...")

    if task_type == "regression":
        scoring = "neg_mean_squared_error"
    else:
        scoring = "accuracy"

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model, X=X, y=y, cv=5, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    if scoring == "neg_mean_squared_error":
        train_scores_mean = -train_scores_mean
        test_scores_mean = -test_scores_mean

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (" + ("MSE" if task_type == "regression" else "Accuracy") + ")")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
