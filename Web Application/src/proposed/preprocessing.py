import cv2
import numpy as np
import os
import glob

def preprocess_single_image(image_path, target_size=(800, 400)):
    img = cv2.imread(image_path)
    if img is None: return None
    # Hanya resize dan kembalikan gambar berwarna
    return cv2.resize(img, target_size)

def load_and_preprocess_dataset(data_dir, target_size=(800, 400)):
    processed_images, labels = [], []
    valid_classes = ['idr_1000', 'idr_2000', 'idr_5000', 'idr_10000', 'idr_20000', 'idr_50000', 'idr_100000']
    print(f"[*] Membaca dataset dari: {data_dir} dengan format Warna (BGR)...")
    for label in valid_classes:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir): continue
        image_paths = glob.glob(os.path.join(class_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(class_dir, "*.[pP][nN][gG]"))
        for img_path in image_paths:
            img_preprocessed = preprocess_single_image(img_path, target_size)
            if img_preprocessed is not None:
                processed_images.append(img_preprocessed)
                labels.append(label)
    return processed_images, labels