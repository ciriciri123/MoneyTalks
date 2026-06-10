# 💵 MoneyTalks: Project Master Documentation

**MoneyTalks** adalah aplikasi teknologi asistif berbasis web yang memungkinkan penyandang tunanetra untuk mengidentifikasi denominasi uang kertas Rupiah secara mandiri dan *real-time*

---

## 📄 Bagian 1: Product Requirements Document (PRD) v2.0

### 1. Ringkasan Eksekutif & Masalah

* **Visi:** Menyediakan deteksi  *real-time* **, audio instan, dan tanpa hambatan bagi pengguna tunanetra**^^.
* **Masalah:** Lebih dari 2,2 miliar orang di seluruh dunia mengalami gangguan penglihatan, dan identifikasi mata uang kertas merupakan tantangan harian yang signifikan yang berisiko pada eksploitasi finansial^^.
* **Solusi:** Pendekatan *browser-first* yang agnostik perangkat, bebas instalasi, tanpa akun, dan berbasis audio ( *audio-first* **)**^^.

### 2. Stack Teknologi Terintegrasi

**Berdasarkan revisi v2.0, berikut adalah komponen teknis yang digunakan**^^:

| **Layer**          | **Teknologi**   | **Justifikasi**                                                                         |
| ------------------------ | --------------------- | --------------------------------------------------------------------------------------------- |
| **Web Framework**  | Flask (Python)        | Kontrol penuh atas routing dan integrasi alami dengan codebase ML Python.                     |
| **Camera Access**  | MediaDevices API      | Native browser API; tanpa library eksternal, beban server nol.                                |
| **Text-to-Speech** | Web Speech API        | Latensi nol (client-side) dengan gTTS sebagai fallback server-side.                           |
| **ML Inference**   | scikit-learn + joblib | Deserialisasi model `.pkl`. Berjalan di server menggunakan pola *thread-safe singleton* . |
| **Database**       | Supabase (PostgreSQL) | Managed service untuk menyimpan data scan, admin, dan versi model.                            |
| **File Storage**   | Supabase Storage      | Penyimpanan objek untuk gambar scan dan file model `.pkl`.                                  |

### 3. Arsitektur Sistem (Four-Tier)

1. **Browser Tier:** Jinja2, Vanilla JS, MediaDevices API, Web Speech API^^.
2. **Application Tier:** Flask, Gunicorn, Nginx, Flask-Login, gTTS^^.
3. **Inference Tier:**`<span class="citation-318">inference.py</span>`, joblib, scikit-learn, Pillow^^.
4. **Data Tier:** Supabase PostgreSQL & Storage^^.

### 4. Alur Kerja Deteksi (Guest User)

1. **Akses Kamera:** Browser meminta izin melalui `<span class="citation-316">getUserMedia</span>`^^.
2. **Capture Frame:** JavaScript mengambil frame video setiap 1 detik (1000ms) melalui elemen `<span class="citation-315"><canvas></span>`^^.
3. **Inference:** Frame dikirim ke `/api/detect`. **Flask melakukan preprocessing (Pillow) dan menjalankan **`<span class="citation-314">model.predict()</span>`^^.
4. **Threshold:** Jika kepercayaan ( *confidence* ) **$\ge 0.75$**, label dikembalikan. **Jika **$< 0.75$**, sistem meminta user mendekatkan uang**^^.
5. **Output Audio:** Hasil dibacakan via Web Speech API atau fallback gTTS^^.

---

## 👁️ Bagian 2: Panduan Proyek Computer Vision (COMP7116001)

### 1. Learning Outcomes (LO)

* **LO 3 & 4:** Membangun sistem pengenalan gambar dan menggunakan fitur untuk korespondensi antar gambar^^.

### 2. Komponen Penilaian (Bobot 100%)

* **Aplikasi (40%):** Harus fungsional, stabil, dan *user-friendly*^^^^^^^^.
* **Laporan Tertulis (40%):** 10–15 halaman dengan struktur akademik (Metodologi, Analisis, Hasil)^^^^^^^^.
* **Video Demo (20%):** Durasi < 5 menit, penjelasan alur kerja dan hasil yang profesional^^^^^^^^.

### 3. Persyaratan Teknis & Etika

* **Dataset:** Harus bersumber secara legal dan etis. **Dokumentasikan proses pengumpulan dan preprocessing**^^.
* **Responsible AI:** Hindari bias, pastikan transparansi, dan lindungi privasi data sensitif^^^^^^^^.

---

## 🛠️ Bagian 3: Panduan Software Engineering (COMP6100001)

### 1. Kriteria Utama Penilaian (SO 3, 4, 5)

* **Manajemen Proyek:** Penggunaan alat seperti Gantt Chart atau Trello untuk identifikasi tugas dan lini masa^^.
* **Desain Sistem:** Pengembangan model arsitektur yang detail (seperti diagram UML)^^.
* **Version Control:** Penggunaan Git (GitHub/GitLab) secara efektif untuk kolaborasi tim, manajemen branch, dan penanganan konflik^^^^^^^^.

### 2. Quality Assurance & Risiko

* **Pengujian Komprehensif:** Wajib melakukan Unit, Integration, System, dan Acceptance Testing beserta dokumentasi hasilnya^^.
* **Analisis Risiko:** Identifikasi kerentanan keamanan dan strategi mitigasinya^^.

### 3. Ketentuan Laporan (Project Portfolio)

**Laporan harus dibuat dalam bentuk ****Notion** atau proposal **PKM-KC** yang mencakup^^:

* Model SDLC yang digunakan dan penjelasan tiap fasenya.
* Detail persyaratan ( *requirements* ), desain, pengembangan, dan pengujian.

---

## 🛡️ Keamanan & Kepatuhan Data

| **Kategori**      | **Standar yang Diterapkan**                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Enkripsi**      | **HTTPS/TLS 1.2+ dipaksakan oleh Nginx**.                                                                                           |
| **Password**      | **Hashing menggunakan****bcrypt**(cost factor 12) dengan kolom `<span class="citation-298">VARCHAR(255)</span>`.                  |
| **Akses DB**      | Row Level Security (RLS) pada Supabase;**API key hanya di sisi server**.                                                            |
| **Aksesibilitas** | **Kepatuhan** **WCAG 2.1 AA** **, teks hasil deteksi**$\ge 48pt$**, dan dukungan***screen reader*(aria-live). |

---

### 💡 Catatan Technical Lead:

* **VCS:** Pastikan setiap anggota tim berkontribusi secara konsisten di GitHub. **Saya akan memantau ***commit history* untuk memastikan penilaian SO 5 (Kerja Tim) terpenuhi^^.
* **CV Pipeline:** Mengingat batasan  *Classical CV* **, pastikan pipeline ****ORB -> BoVW -> SVM** dioptimalkan untuk latensi di bawah 2 detik agar memenuhi NFR-01^^.
* **Testing:** Jangan lupa mendokumentasikan hasil *Unit Test* untuk modul `inference.py` dan `preprocessing` secara terpisah.
