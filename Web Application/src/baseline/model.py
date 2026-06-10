import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import function yang sudah kita buat di features dan preprocessing file
from preprocessing import load_and_preprocess_dataset
from features import extract_orb_features, build_visual_vocabulary, extract_bovw_histograms

def train_and_evaluate():
    DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=== FASE 1: PERSIAPAN DATA & EKSTRAKSI FITUR ===")
    # Menyimpan label (y) dan histogramnya (X) ke dalam variabel
    images, labels = load_and_preprocess_dataset(DATASET_DIR)
    
    if len(images) == 0:
        print("[ERROR] Dataset tidak ditemukan.")
        return

    descriptors = extract_orb_features(images)
    
    kmeans_model = build_visual_vocabulary(descriptors, num_clusters=150)
    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, 'bovw_dictionary.pkl'))
    
    X = extract_bovw_histograms(descriptors, kmeans_model)
    y = np.array(labels)

    print("\n=== FASE 2: PELATIHAN MODEL (Sesuai Metodologi) ===")
    # Membagi dataset 80% untuk Training, 20% untuk Testing
    # stratify=y memastikan proporsi setiap pecahan uang seimbang di data test & train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[*] Data Latih (Training): {len(X_train)} gambar")
    print(f"[*] Data Uji (Testing): {len(X_test)} gambar")

    # 1. Baseline Model KNN
    print("\n[*] Melatih Baseline Model (K-Nearest Neighbors)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_preds)
    print(f" -> Akurasi Baseline (KNN): {knn_acc * 100:.2f}%")

    # 2. Baseline Model Multi-class SVM
    print("\n[*] Melatih Model Utama (Support Vector Machine)...")
    # Kernel RBF (Radial Basis Function) adalah standar untuk data BoVW yang non-linear
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    print(f" -> Akurasi Model Utama (SVM): {svm_acc * 100:.2f}%")

    print("\n=== FASE 3: EVALUASI SVM (Sesuai Success Criteria) ===")
    print("Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, svm_preds))

    svm_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    joblib.dump(svm, svm_path)
    print(f"[*] Model SVM berhasil disimpan di: {svm_path}")

if __name__ == "__main__":
    train_and_evaluate()