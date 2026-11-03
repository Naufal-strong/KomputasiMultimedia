# Tugas 2: Transformasi Gambar, Image Compression, dan Watermarking
**Mata Kuliah:** Komputasi Multimedia  
**Topik:** Augmentasi Citra Digital (DAiSEE), Kompresi Citra Multi-Format, dan Digital Watermarking  
**Tahun Akademik:** 2026

---

## 📌 Ringkasan Proyek & Struktur Deliverables

Proyek ini menyelesaikan seluruh instruksi **Tugas 2 Komputasi Multimedia** secara tuntas, mencakup:
1. **Laporan Resmi PDF:** [`Laporan_Tugas2_KomputasiMultimedia.pdf`](./Laporan_Tugas2_KomputasiMultimedia.pdf) (12 Halaman komprehensif, dilengkapi cover resmi, pembahasan teoretis, tabel komparasi 10 teknik kompresi, metrik empiris, grafik beresolusi tinggi, dan sitasi IEEE/APA dengan DOI).
2. **Interactive Jupyter Notebook:** [`Tugas2_KomputasiMultimedia.ipynb`](./Tugas2_KomputasiMultimedia.ipynb) (18 sel interaktif terstruktur, mandiri, siap dijalankan langsung di VS Code / JupyterLab / Google Colab).
3. **Folder Data & Hasil Eksperimen:**
   - `data/video/`: Berkas video e-learning representatif DAiSEE (300 frame, 30 FPS).
   - `data/extracted_frames/`: 300 frame hasil ekstraksi video.
   - `data/sample_10_frames/`: 10 frame yang dipilih secara acak menggunakan *reproducible seed* (`seed = 42`).
   - `data/augmented_images/`: Folder terpisah per teknik augmentasi (Horizontal Flip, Slight Rotation, Random Crop & Scale, Color Jitter, Gaussian Noise, Gaussian Blur).
   - `data/compression_inputs/`: Citra benchmark multi-format (`.bmp`, `.jpg`, `.png`, `.svg`, dan hasil rasterisasi SVG).
   - `data/compression_results/`: Berkas terkompresi (RLE, JPEG, WebP Lossy & Lossless) serta `compression_benchmark_summary.csv`.
   - `data/watermark_results/`: Logo watermark biner, citra ter-watermark (Visible, LSB, DCT), hasil ekstraksi, berkas serangan manipulasi, serta `watermark_robustness_summary.csv`.
   - `output/`: Seluruh visualisasi grafis resolusi tinggi (grid augmentasi, grafik rasio kompresi, grafik *space saving*, grafik kualitas vs waktu, visualisasi residual heatmap, dan matriks ketahanan watermarking).
4. **Dependensi:** [`requirements.txt`](./requirements.txt).

---

## 🚀 Cara Menjalankan (Reproducibility Guide)

### 1. Persiapan Environment
Gunakan Python 3.10+ (atau Python 3.14). Buat virtual environment dan pasang dependensi:
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Menjalankan Seluruh Pipeline Eksperimen Ulang
Untuk mengeksekusi ulang seluruh simulasi, perhitungan metrik, dan pembuatan grafik:
```bash
python run_experiments.py
```

### 3. Membuka dan Menjalankan Jupyter Notebook
```bash
jupyter notebook Tugas2_KomputasiMultimedia.ipynb
```
Atau buka langsung file `Tugas2_KomputasiMultimedia.ipynb` melalui ekstensi Jupyter di VS Code.

### 4. Melakukan Kompilasi Ulang Laporan PDF
```bash
python generate_report_pdf.py
```

---

## 📊 Ringkasan Temuan Utama

### Bagian A: Augmentasi Data (DAiSEE)
- **Horizontal Flipping:** Sangat aman dan valid karena simetri bilateral ekspresi afektif wajah manusia (Shorten & Khoshgoftaar, 2019).
- **Rotasi Kecil ($\pm 8^\circ$) vs Rotasi Besar ($>30^\circ$):** Rotasi kecil merefleksikan *head tilt* saat siswa berpikir/bingung. Rotasi besar merusak prior gravitasi posisi duduk siswa di depan laptop (Kumar et al., 2016; Dewan et al., 2019).
- **Random Crop & Scale, Color Jitter, Noise, Blur:** Sangat sesuai untuk memodelkan variasi jarak kamera, fluktuasi pencahayaan ruangan kamar, serta keterbatasan sensor kamera web murah.

### Bagian B: Image Compression
- **Citra Fotografis Natural (BMP):** WebP Lossy mencapai Compression Ratio tertinggi **126.5:1** (*Space Saving* **99.21%**, PSNR **41.96 dB**), sementara JPEG DCT konvensional menawarkan kualitas rekonstruksi lebih baik (CR 59.8:1, PSNR **45.15 dB**). RLE mengalami kegagalan kompresi negatif (*space saving* **-83.93%**) akibat ketiadaan run piksel identik panjang.
- **Citra Vektor / Grafis (SVG):** SVG wajib dirasterisasi terlebih dahulu ke matriks piksel sebelum dikompresi. Pada citra grafis, WebP Lossless paling superior (CR **2.07:1**, *Space Saving* **51.70%**), sedangkan JPEG mengalami pemborosan bit (*space saving* -80.75%) akibat artefak frekuensi tinggi DCT pada batas tepi tajam (fenomena Gibbs).

### Bagian C: Digital Watermarking
- **Imperceptibility:** LSB menghasilkan PSNR sangat tinggi (**73.90 dB**, SSIM 0.99998), sedangkan DCT menghasilkan PSNR **43.00 dB** (tak kasatmata).
- **Robustness (Uji Ketahanan):**
  - **Kompresi JPEG ($Q=50$ dan $Q=30$):** LSB terdegradasi parah (BER **61.28%** dan **58.47%**, logo hilang). DCT relatif lebih tahan (BER **26.05%** dan **22.88%**), tetapi tetap terdegradasi signifikan akibat kuantisasi koefisien DCT yang agresif.
  - **Derau Gaussian ($\sigma=10$):** LSB hancur (BER **50.95%**, identik tebakan acak), sedangkan DCT sangat tangguh (BER hanya **2.61%**, NCC **0.9195**).
  - **Gaussian Blur ($3\times 3$):** LSB terdegradasi (BER **28.54%**), sedangkan DCT terekstrak hampir sempurna (BER **0.02%**, NCC **0.9992**).

