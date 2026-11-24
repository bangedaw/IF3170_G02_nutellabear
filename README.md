# IF3170 G02 · Nutellabear

Repositori Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027. Proyek ini memprediksi pembatalan pemesanan penginapan (`is_canceled`) dengan lima model yang diimplementasikan dari awal menggunakan NumPy: KNN, pohon keputusan C4.5, regresi logistik, SVM, dan Naive Bayes. Notebook membandingkan model pada data validasi yang sama dan menyediakan pembanding scikit-learn.

| Lokasi | Isi |
| --- | --- |
| [Notebook pengumpulan](nb/Notebook_Template_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb) | Eksplorasi data, prapemrosesan, pelatihan, evaluasi, penyimpanan model, dan pembuatan CSV. |
| [`src/`](src/) | Implementasi model, prapemrosesan, evaluasi, dan modul eksperimen tambahan. |
| [`tests/`](tests/) | Pengujian implementasi. |
| [`output/`](output/) | CSV prediksi dan gambar hasil eksperimen. |
| [Laporan](docs/laporan.pdf) | Metode, hasil, dan pembagian tugas. [Sumber LaTeX](docs/laporan.tex). |

## Persiapan dan cara menjalankan

1. Dari akar repositori, buat lingkungan Python 3.12 dan pasang dependensi. Contoh berikut menggunakan PowerShell.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt ipykernel
   .\.venv\Scripts\python.exe -m ipykernel install --user --name if3170 --display-name "Python (IF3170)"
   ```

2. Letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` dari kompetisi dalam satu folder bernama `T2-IF3170-2026-2027/` di akar repositori. Jika dataset berada di lokasi lain, atur `IF3170_DATA_DIR` sebelum memulai kernel notebook.
3. Buka notebook pengumpulan, pilih kernel **Python (IF3170)**, lalu pilih **Run All**. Folder keluaran bawaan adalah `output/`. Atur `IF3170_OUTPUT_DIR` ke folder lain jika ingin membandingkan hasil baru dengan artefak yang sudah ada. Di Kaggle, tambahkan dataset kompetisi sebagai input, lalu jalankan notebook. Keluaran disimpan di `/kaggle/working/output/`.
4. Jalankan pengujian dari akar repositori dengan `.\.venv\Scripts\python.exe -m pytest -q`.

Notebook menghasilkan `submission.csv` dari KNN dasar serta CSV untuk empat keluarga model lainnya. CSV eksperimen tambahan di `output/`, seperti DTL dengan penalti MDL dan KNN dengan *pipeline* kombinasi, berasal dari jalur eksperimen di `src/` yang dibahas dalam laporan.

## Pembagian tugas

| Anggota | NIM | Kontribusi |
| --- | --- | --- |
| Samuelson Dharmawan Tanuraharja | 13524001 | SVM dengan SMO dan tiga kernel, strategi subsampel, kalibrasi Platt, pembanding SVC, serta visualisasi bidang pemisah. |
| Edward David Rumahorbo | 13524036 | Naive Bayes, eksplorasi data, pembanding `CategoricalNB`, graf Naive Bayes, klasifikasi berbasis biaya, serta koordinasi laporan. Eksperimen tambahan meliputi KNN berbobot dan kombinasi, pemangkasan dan MDL pada DTL, FISTA dan Adam, diskretisasi MDL pada Naive Bayes, serta seleksi *working set* orde kedua pada SMO. |
| Narendra Dharma Wistara M. | 13524044 | Regresi logistik dengan L1, L2, dan Elastic Net, video kontur *log-loss*, struktur repositori, serta antarmuka penyimpanan dan pemuatan model. |
| Reinhard Alfonzo Hutabarat | 13524056 | DTL C4.5 untuk fitur numerik, kategorikal, dan nilai hilang, visualisasi pohon dengan `top_n`, serta perapian konfigurasi dan evaluasi notebook. |
| Miguel Rangga Deardo Sinaga | 13524069 | Prapemrosesan, modul evaluasi dan validasi silang berstrata, KNN, integrasi notebook, pengumpulan Kaggle, serta pemeriksaan reproduksibilitas. |
