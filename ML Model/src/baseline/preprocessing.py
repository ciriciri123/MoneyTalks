import cv2
import numpy as np
import os
import glob

def preprocess_single_image(image_path, target_size=(600, 300)):
    # Memuat dan memproses satu gambar untuk ekstraksi ORB.
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Preprocess umum: Resize, Grayscale, dan Blur
    img_resized = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred

def load_and_preprocess_dataset(data_dir, target_size=(600, 300)):
    # Melakukan looping ke seluruh folder uang dan memproses gambar
    processed_images = []
    labels = []
    
    valid_classes = [
        'idr_1000', 'idr_2000', 'idr_5000', 'idr_10000', 'idr_20000', 'idr_50000', 'idr_100000'
    ]
    
    print(f"[*] Membaca dataset dari: {data_dir}")
    
    for label in valid_classes:
        class_dir = os.path.join(data_dir, label)
        
        if not os.path.isdir(class_dir):
            continue
            
        # Cari semua gambar jpg/png di folder tersebut
        image_paths = glob.glob(os.path.join(class_dir, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(class_dir, "*.[pP][nN][gG]"))
        
        print(f" -> Memproses {len(image_paths)} gambar untuk '{label}'...")
        
        for img_path in image_paths:
            img_preprocessed = preprocess_single_image(img_path, target_size)
            if img_preprocessed is not None:
                processed_images.append(img_preprocessed)
                labels.append(label)

    print(f"[*] Selesai! Total gambar diproses: {len(processed_images)}")
    return processed_images, labels

if __name__ == "__main__":
    DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    images, labels = load_and_preprocess_dataset(DATASET_DIR)