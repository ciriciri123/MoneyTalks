import cv2
import numpy as np
import os
import joblib
from sklearn.cluster import MiniBatchKMeans
from preprocessing import load_and_preprocess_dataset

def extract_orb_and_color_features(images, max_features=2000):
    orb = cv2.ORB_create(nfeatures=max_features)
    descriptor_list = []
    color_histograms = []
    
    print(f"[*] Memulai ekstraksi ORB & Warna dari {len(images)} gambar...")
    for img in images:
        # 1. Fitur warna (HSV Histogram)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Hitung histogram 3D HSV (Hue, Saturation, Value)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        color_histograms.append(hist.flatten())

        # 2. Fitur tektur (ORB + CLAHE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        keypoints, descriptors = orb.detectAndCompute(blurred, None)
        if descriptors is not None:
            descriptor_list.append(descriptors)
        else:
            descriptor_list.append(np.array([]))
            
    return descriptor_list, np.array(color_histograms)

def build_visual_vocabulary(descriptor_list, num_clusters=800):
    all_descriptors = []
    for des in descriptor_list:
        if len(des) > 0:
            all_descriptors.extend(des)
    all_descriptors = np.array(all_descriptors)
    
    print(f"[*] Melatih K-Means dengan {len(all_descriptors)} keypoints...")
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=2048)
    kmeans.fit(all_descriptors)
    return kmeans

def extract_bovw_histograms(descriptor_list, kmeans_model):
    num_clusters = kmeans_model.n_clusters
    histograms = []
    for des in descriptor_list:
        if len(des) > 0:
            words = kmeans_model.predict(des)
            hist, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
        else:
            hist = np.zeros(num_clusters)
        histograms.append(hist)
    return np.array(histograms)