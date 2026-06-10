import cv2
import numpy as np
import os
import joblib
from sklearn.cluster import MiniBatchKMeans
from preprocessing import load_and_preprocess_dataset

def extract_orb_features(images, max_features=1000):
    # Mengekstrak deskriptor ORB dari sekumpulan gambar.
    orb = cv2.ORB_create(nfeatures=max_features)
    descriptor_list = []
    
    print(f"[*] Memulai ekstraksi fitur ORB dari {len(images)} gambar...")
    for i, img in enumerate(images):
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            descriptor_list.append(descriptors)
        else:
            print(f" [WARNING] Gambar index ke-{i} tidak memiliki fitur.")
            descriptor_list.append(np.array([])) # Masukkan array kosong jika gagal
            
    print(f"[*] Ekstraksi selesai!")
    return descriptor_list

def build_visual_vocabulary(descriptor_list, num_clusters=800):
    # Mengelompokkan descriptors menjadi 'Visual Words' menggunakan K-Means.
    # Gabungkan semua descriptor jadi satu array
    all_descriptors = []
    for des in descriptor_list:
        if len(des) > 0:
            all_descriptors.extend(des)
    all_descriptors = np.array(all_descriptors)
    
    print(f"[*] Melatih K-Means dengan {len(all_descriptors)} keypoints untuk membuat {num_clusters} 'Visual Words'...")
    # Kita pakai MiniBatchKMeans agar komputasinya jauh lebih cepat dan hemat RAM
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=2048)
    kmeans.fit(all_descriptors)
    
    print("[*] Visual Vocabulary (Kamus Visual) berhasil dibuat!")
    return kmeans

def extract_bovw_histograms(descriptor_list, kmeans_model):
    # Mengonversi descriptor setiap gambar menjadi Histogram berdasarkan Kamus Visual.
    num_clusters = kmeans_model.n_clusters
    histograms = []
    
    print("[*] Mengonversi setiap gambar menjadi Histogram BoVW...")
    for des in descriptor_list:
        if len(des) > 0:
            # Prediksi setiap titik di gambar ini masuk ke 'kata' (cluster) yang mana
            words = kmeans_model.predict(des)
            # Hitung frekuensinya (buat histogram)
            hist, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
        else:
            hist = np.zeros(num_clusters)
            
        histograms.append(hist)
        
    return np.array(histograms)

if __name__ == "__main__":
    DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Preprocessing
    images, labels = load_and_preprocess_dataset(DATASET_DIR)
    
    if len(images) > 0:
        # 2. Ekstraksi ORB
        descriptors = extract_orb_features(images)
        
        # 3. Bangun Kamus Visual (BoVW)
        kmeans_model = build_visual_vocabulary(descriptors, num_clusters=150)
        
        # 4. Ubah gambar jadi Histogram
        X_features = extract_bovw_histograms(descriptors, kmeans_model)
        
        # 5. Simpan Kamus Visual (KMeans Model) ke file .pkl
        dict_path = os.path.join(MODELS_DIR, 'bovw_dictionary.pkl')
        joblib.dump(kmeans_model, dict_path)
        
        print(f"\n[INFO] Pipeline Fitur Selesai!")
        print(f"[INFO] Bentuk fitur akhir (X) yang siap masuk SVM: {X_features.shape}")
        print(f"[INFO] Kamus visual berhasil disimpan di: {dict_path}")