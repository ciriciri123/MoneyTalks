import matplotlib.pyplot as plt

def create_donut_chart():
    # 1. Persiapan Data
    labels = ['Training Data\n(80%)', 'Testing Data\n(20%)']
    sizes = [80, 20]
    
    # 2. Skema Warna (Bisa disesuaikan dengan tema PPT kamu)
    # Gunakan warna Biru Profesional dan Oranye/Kuning sebagai aksen
    colors = ['#4A90E2', '#F5A623'] 
    
    # Membuat jarak sedikit (explode) agar terlihat dinamis
    explode = (0.05, 0) 

    # 3. Membuat Pie Chart Dasar
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        colors=colors, 
        autopct='%1.0f%%', 
        startangle=90, 
        pctdistance=0.85,
        textprops=dict(color="black", fontsize=12, weight="bold")
    )

    # 4. MAGIC TRICK: Mengubah Pie Chart menjadi Donut Chart
    # Menggambar lingkaran putih di tengah
    centre_circle = plt.Circle((0,0), 0.65, fc='white')
    fig.gca().add_artist(centre_circle)

    # Menambahkan teks di tengah lubang Donut
    plt.text(0, 0, 'Stratified\nSplit', ha='center', va='center', fontsize=14, weight='bold', color='#333333')

    # Memastikan bentuknya lingkaran sempurna
    ax.axis('equal')  
    
    plt.title("SVM Data Splitting Strategy", fontsize=16, weight="bold", pad=20)
    plt.tight_layout()

    # 5. Menyimpan dengan Resolusi Tinggi dan Background Transparan
    output_filename = 'data_split_donut.png'
    plt.savefig(output_filename, transparent=True, dpi=300)
    
    print(f"[*] Selesai! Gambar berhasil disimpan sebagai '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    create_donut_chart()