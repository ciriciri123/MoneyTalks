import os
import glob

# Pastikan path ini sama dengan yang ada di augment.py kamu
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def remove_augmented_data():
    valid_classes = ['idr_1000', 'idr_2000', 'idr_5000', 'idr_10000', 'idr_20000', 'idr_50000', 'idr_100000']
    
    # Target akhiran file yang akan dihapus
    target_suffixes = ['_dark', '_bright', '_blur', '_noise']
    
    print(f"[*] Memulai pembersihan data augmentasi di folder: {DATASET_DIR}")
    total_deleted = 0
    
    for label in valid_classes:
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir):
            continue
            
        print(f" -> Memeriksa folder '{label}'...")
        
        for suffix in target_suffixes:
            # MAGIC FIX: Tambahkan "_*" setelah suffix untuk menangkap angka ID di belakangnya!
            # Contoh: *_{suffix}_*.* akan sukses menangkap file seperti "IMG_123_blur_181.jpg"
            pattern = os.path.join(class_dir, f"*{suffix}_*.*")
            files_to_delete = glob.glob(pattern)
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    total_deleted += 1
                except Exception as e:
                    print(f"    [ERROR] Gagal menghapus {file_path}: {e}")
                    
    print(f"\n[*] Selesai! Berhasil menghapus total {total_deleted} file augmentasi.")
    print("[*] Ruang penyimpanan (Harddisk) laptopmu sudah lega kembali! 🚀")

if __name__ == "__main__":
    # Peringatan keamanan sebelum mengeksekusi
    print("WARNING: Script ini akan menghapus semua gambar augmentasi secara permanen.")
    konfirmasi = input("Ketik 'Y' untuk melanjutkan penghapusan: ")
    
    if konfirmasi.lower() == 'y':
        remove_augmented_data()
    else:
        print("[*] Dibatalkan. Tidak ada file yang dihapus.")