import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer

from preprocessing import load_and_preprocess_dataset
from features import extract_orb_and_color_features, build_visual_vocabulary, extract_bovw_histograms

def train_proposed_model():
    DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'proposed')
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=== PROPOSED MODEL: PERSIAPAN DATA ===")
    images, labels = load_and_preprocess_dataset(DATASET_DIR)
    
    # Ekstrak ORB dan warna sekaligus
    descriptors, color_features = extract_orb_and_color_features(images)
    
    # Mencegah data leakage dengan split data dulu
    X_desc_train, X_desc_test, X_color_train, X_color_test, y_train, y_test = train_test_split(
        descriptors, color_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"[*] Data Latih (Train): {len(y_train)} | Data Uji (Test): {len(y_test)}")

    print("\n=== PROPOSED MODEL: MEMBANGUN VOCABULARY (Hanya dari Train Data) ===")
    # 1. Bangun Kamus Visual hanya dari X_desc_train
    kmeans_model = build_visual_vocabulary(X_desc_train, num_clusters=800)
    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, 'bovw_dictionary.pkl'))
    
    # 2. Ekstrak Histogram BoVW untuk Train dan Test secara terpisah
    bovw_train_raw = extract_bovw_histograms(X_desc_train, kmeans_model)
    bovw_test_raw = extract_bovw_histograms(X_desc_test, kmeans_model)
    
    print("\n=== PROPOSED MODEL: FEATURE FUSION & SCALING ===")
    # 3. TF-IDF hanya di-fit pada data Train
    tfidf = TfidfTransformer()
    bovw_train_tfidf = tfidf.fit_transform(bovw_train_raw).toarray()
    bovw_test_tfidf = tfidf.transform(bovw_test_raw).toarray()
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_scaler.pkl'))
    
    # 4. Gabungkan fitur BoVW + fitur Warna
    X_train_fused = np.hstack((bovw_train_tfidf, X_color_train))
    X_test_fused = np.hstack((bovw_test_tfidf, X_color_test))

    # 5. Normalisasi (StandardScaler) hanya di-fit pada Data Train
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_fused)
    X_test_final = scaler.transform(X_test_fused) # Cuma di-transform!
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    print("\n=== PROPOSED MODEL: PELATIHAN SVM ===")
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', random_state=42)
    svm.fit(X_train_final, y_train)
    
    # 1. Tebak Data Testing
    svm_preds = svm.predict(X_test_final)
    svm_acc = accuracy_score(y_test, svm_preds)
    
    # 2. Tebak Data Training (Train) - untuk cek overfit
    svm_train_preds = svm.predict(X_train_final)
    train_acc = accuracy_score(y_train, svm_train_preds)
    
    print("\n=== CEK OVERFITTING ===")
    print(f" -> Akurasi Data Latih (Train) : {train_acc * 100:.2f}%")
    print(f" -> Akurasi Data Uji (Test)   : {svm_acc * 100:.2f}%")
    
    print(f"\n -> Akurasi PROPOSED MODEL (ORB + Warna + TF-IDF): {svm_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, svm_preds))

    svm_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    joblib.dump(svm, svm_path)
    print(f"[*] Model PROPOSED berhasil disimpan di: {svm_path}")

if __name__ == "__main__":
    train_proposed_model()