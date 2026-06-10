import cv2
import numpy as np
import os
import glob
import random

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def generate_single_augmentation(img_path, unique_id):
    img = cv2.imread(img_path)
    if img is None: return False
    
    dir_name = os.path.dirname(img_path)
    name, ext = os.path.splitext(os.path.basename(img_path))
    
    # KITA TAMBAHKAN 'rotate180' KE DALAM PILIHAN ACAK
    effect = random.choice(['dark', 'bright', 'blur', 'noise', 'rotate180'])
    
    if effect == 'dark':
        new_img = cv2.convertScaleAbs(img, alpha=0.75, beta=0) 
    elif effect == 'bright':
        new_img = cv2.convertScaleAbs(img, alpha=1.1, beta=15)
    elif effect == 'blur':
        new_img = cv2.GaussianBlur(img, (5, 5), 0)
    elif effect == 'noise': 
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        new_img = cv2.add(img, noise)
    else: # effect == 'rotate180'
        new_img = cv2.rotate(img, cv2.ROTATE_180) # Memutar gambar 180 derajat
        
    new_filename = f"{name}_{effect}_{unique_id}{ext}"
    cv2.imwrite(os.path.join(dir_name, new_filename), new_img)
    return True

def run_balanced_augmentation():
    valid_classes = ['idr_1000', 'idr_2000', 'idr_5000', 'idr_10000', 'idr_20000', 'idr_50000', 'idr_100000']
    print(f"[*] Memindai dataset di: {DATASET_DIR}")
    
    original_files_map = {}
    for label in valid_classes:
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir): continue
        all_images = glob.glob(os.path.join(class_dir, "*.*"))
        originals = [img for img in all_images if not any(x in img for x in ['_dark', '_bright', '_blur', '_noise'])]
        original_files_map[label] = originals
        
    if not original_files_map: return
        
    max_class = max(original_files_map, key=lambda k: len(original_files_map[k]))
    max_count = len(original_files_map[max_class])
    target_total = max_count * 2 # Balanced Oversampling
    
    print(f"[*] TARGET TOTAL disetel ke: {target_total} gambar per kelas.\n")
    
    total_generated = 0
    for label, originals in original_files_map.items():
        current_count = len(originals)
        if current_count == 0: continue
            
        needed = target_total - current_count
        print(f" -> '{label}': Membutuhkan {needed} augmentasi...")
        
        for i in range(needed):
            img_to_augment = random.choice(originals)
            if generate_single_augmentation(img_to_augment, i):
                total_generated += 1
                
    print(f"\n[*] Selesai! Semua kelas BALANCED di {target_total} gambar.")

if __name__ == "__main__":
    run_balanced_augmentation()