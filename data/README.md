# 📊 Dataset Pengaduan Publik Kota Makassar

Direktori ini berisi subset dataset aduan publik Pemerintah Kota Makassar yang telah melalui tahapan sanitasi teks, pembersihan tag metadata, penanganan singkatan, serta normalisasi leksikal dialek lokal Makassar (*ki, ji, mi, tawwa*, serta akhiran *-x*).

## Ringkasan Partisi Data (Stratified Split 80:10:10)
Dataset dibagi dengan metode *Stratified Sampling* untuk mempertahankan proporsi kelas minoritas yang seimbang di seluruh subset:

| Nama Berkas | Jumlah Rekaman | Persentase | Fungsi Pembagian |
| :--- | :---: | :---: | :--- |
| `data_clean_train.csv` | **12.063** baris | 80,0% | Data pelatihan parameter model |
| `data_clean_val.csv` | **1.508** baris | 10,0% | Data validasi tuning & checkpointing |
| `data_clean_test.csv` | **1.508** baris | 10,0% | Data uji independen untuk evaluasi komparatif |
| **Total** | **15.090** baris | **100,0%** | Populasi 5 Kategori Teratas |

## Distribusi 5 Kategori Aduan
1. **Lampu Jalan (PJU):** 10.630 aduan (70,44%) $\rightarrow$ *Kelas Mayoritas*
2. **Drainase & Saluran Air:** 1.398 aduan (9,26%)
3. **Pangkas / Tebang Pohon:** 1.253 aduan (8,30%)
4. **Sampah & Kebersihan:** 962 aduan (6,38%)
5. **Jalan Rusak & Berlubang:** 840 aduan (5,57%) $\rightarrow$ *Kelas Minoritas*
