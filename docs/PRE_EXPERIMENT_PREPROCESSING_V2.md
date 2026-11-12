# Pre-experiment preprocessing V2 for EndekGAN-T2I

**Notebook utama:** `01_EndekGAN_T2I_Preprocessing_BEFORE_Training_V2.ipynb`.

Notebook baru telah dieksekusi dengan kernel Python 3.11 dari drive D. Semua 13 sel kode berhasil; 4 visualisasi tersimpan di notebook.

## Alur preprocessing
1. Audit dan inventaris koleksi Endek lokal beserta tiga sumber eksternal yang diusulkan.
2. Deduplikasi SHA-256, audit kemiripan visual dHash, dan penjagaan provenance.
3. Pelabelan 4 kelas serta pemetaan metadata prompt untuk Text-to-Image.
4. Source- dan visual-group-aware split train/validation/test agar gambar dan turunannya tidak bocor lintas split.
5. Full image resize 512x512 ke 256x256, serta perbandingan crop kontekstual, crop tekstur, dan patch lokal.
6. Augmentasi fotometrik ringan untuk data training dan penyeimbangan jumlah file per kelas.
7. Ekspor dua perlakuan berupa manifest tanpa menggandakan citra sumber.
8. QA integritas data, penulisan flowchart dan gerbang pemeriksaan sebelum pelatihan GAN.

## Perlakuan dataset yang sudah tersedia
- A: `ENDEK_PREPROCESSING_V2/arms/full_image_augmented_A/` — 944 train, 37 validation, 37 testing.
- B: `ENDEK_PREPROCESSING_V2/arms/full_plus_crop_ablation_B/` — 1280 train, 37 validation, 37 testing.
- B menambahkan 336 kandidat crop yang tetap mewarisi prompt asal. B adalah perlakuan ablation yang belum terbukti memiliki kecocokan semantik untuk setiap crop.
- Kedua arm menggunakan gambar sumber dan set evaluasi yang sama; angka tersebut merupakan jumlah file, bukan jumlah motif Endek independen hasil penambahan sumber eksternal.

## Batasan ilmiah dan status
- Ketiga folder penerimaan dataset Mendeley/Roboflow masih kosong pada saat notebook dijalankan.
- Dataset gabungan FINAL belum ditetapkan: setiap gambar eksternal harus diduplikasi/diuji kemiripannya, diklasifikasikan, dan dipasangkan dengan metadata teks yang benar.
- Eksekusi preprocessing tidak otomatis memulai training GAN atau menghasilkan metrik FID/KID.
- Evaluasi pakar Endek ditempatkan setelah proses pembangkitan gambar AI.
- Ketika sumber eksternal sudah ada, audit, split sumber dan preprocessing perlu dijalankan ulang untuk menyusun dataset final penelitian baru.

## Cara membuka di VS Code
- Workspace: `D:\DISERTASI_S3\WIWIK\EXPERIMEN_2\ENDEK_T2I_EXPERIMENT_V2\EndekGAN-T2I-V2.code-workspace`.
- Launcher: `D:\VSCode-MultiAgent-AI\START_ENDEK_T2I_V2.cmd` sudah diarahkan ke notebook preprocessing baru.
- Kernel: **Python 3.11 (EndekGAN-T2I V2 | D:)**.
- Interpreter: `D:\VSCode-MultiAgent-AI\envs\endekgan-t2i-v2-py311\Scripts\python.exe`.

## Output yang dihasilkan
- `ENDEK_PREPROCESSING_V2/arms/` berisi manifest data loader A/B.
- `ENDEK_PREPROCESSING_V2/pretraining_quality_report.json` berisi ringkasan audit dua arm.
- `ENDEK_PREPROCESSING_V2/PRE_EXPERIMENT_STATUS.json` berisi status gerbang kesiapan.
- `ENDEK_PREPROCESSING_V2/FLOWCHART_PREPROCESSING_V2.png` adalah diagram alur.
