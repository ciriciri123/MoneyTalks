import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

# Import fungsi dari file proyekmu
from preprocessing import load_and_preprocess_dataset
from features import extract_orb_and_color_features, build_visual_vocabulary, extract_bovw_histograms

def plot_svm_decision_boundary():
    DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    
    print("=== 1. PERSIAPAN DATA (Versi Ringan untuk Visualisasi) ===")
    # Kita ambil data aslinya (proses ini mungkin butuh 1-2 menit)
    images, labels = load_and_preprocess_dataset(DATASET_DIR)
    
    # Encode label string ('idr_1000') menjadi angka (0, 1, 2, dst) untuk pewarnaan grafik
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    
    print("=== 2. EKSTRAKSI FITUR ===")
    descriptors, color_features = extract_orb_and_color_features(images)
    
    # Supaya cepat, kita buat kamus visual kecil saja untuk visualisasi ini
    kmeans_model = build_visual_vocabulary(descriptors, num_clusters=100)
    bovw_raw = extract_bovw_histograms(descriptors, kmeans_model)
    
    # TF-IDF dan Pembobotan (Sesuai Proposed Model)
    tfidf = TfidfTransformer()
    bovw_tfidf = tfidf.fit_transform(bovw_raw).toarray() * 3.0
    
    # Fitur Gabungan (Tekstur + Warna)
    X_final = np.hstack((bovw_tfidf, color_features))
    
    print("=== 3. DIMENSIONALITY REDUCTION (PCA) ===")
    print(f"Bentuk data awal: {X_final.shape} -> Kita peras jadi 2 Dimensi!")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_final)
    
    print("=== 4. MELATIH SVM 2D ===")
    # Kita latih SVM khusus pada data 2D ini untuk keperluan gambar
    # Kamu bisa ubah kernel='linear' jika ingin garis lurus seperti di contohmu
    # Atau biarkan kernel='rbf' untuk garis meliuk-liuk
    svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_2d.fit(X_2d, y_encoded)
    
    print("=== 5. MENGGAMBAR GRAFIK (Mungkin butuh beberapa detik) ===")
    # Menyiapkan kanvas grafik
    plt.figure(figsize=(10, 8))
    
    # Membuat grid poin untuk menggambar warna latar (Area Keputusan SVM)
    h = .02  # Ukuran step grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Prediksi setiap titik di grid untuk mewarnai latar belakangnya
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Skema Warna (7 Warna untuk 7 Pecahan Uang)
    colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', '#AAFFFF', '#D3D3D3']
    cmap_background = ListedColormap(colors)
    
    # Menggambar Area Keputusan (Garis Batas)
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.5)
    
    # Menggambar Titik Data (Sebaran Data Uang)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_encoded, cmap=ListedColormap(colors), 
                          edgecolors='k', s=50)
    
    # Membuat Keterangan (Legend)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(le.classes_), 
               title="Pecahan Uang", loc="best")
    
    plt.title("Visualisasi 2D Decision Boundary SVM (Proposed Model + PCA)")
    plt.xlabel("Principal Component 1 (Fitur Dominan 1)")
    plt.ylabel("Principal Component 2 (Fitur Dominan 2)")
    plt.tight_layout()
    
    # Tampilkan!
    print("[*] Selesai! Menampilkan grafik...")
    plt.show()

if __name__ == "__main__":
    plot_svm_decision_boundary()