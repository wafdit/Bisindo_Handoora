# BISINDO Handoora — Real-Time Sign Language Recognition

Sistem pengenalan Bahasa Isyarat Indonesia (BISINDO) berbasis computer vision (**MediaPipe Hands**) dan machine learning (**Scikit-Learn**) yang terbagi menjadi dua tingkat pembelajaran:
1. **Level 1 (Abjad Statis):** Deteksi 26 huruf A–Z berbasis 1 frame (127 fitur).
2. **Level 2 (Kata Dinamis):** Deteksi frasa/kata berbasis *sliding window* 30 frame ($30 \times 127 = 3.810$ fitur) dilengkapi *Motion Filter* untuk membedakan tangan diam dan gerakan aktif.

---

## Ringkasan Perbedaan File (Agar Tidak Tertukar)

| Kategori | Level 1 (Abjad Statis) | Level 2 (Kata Dinamis - Aktif) | Versi Lama / Eksperimen (Jangan Dipakai) |
| :--- | :--- | :--- | :--- |
| **Rekam & Uji Kamera** | `collect_alphabet.py` & `test_realtime.py` | `app_level2.py` *(gabungan rekam + deteksi)* | `collect_words.py`, `test_words.py` |
| **Training Model** | `train_model.py` | `train_level2.py` | `train_words.py` |
| **File Dataset** | `dataset_abjad_optimal.csv` | `dataset_kata_level2.csv` | `dataset_abjad_2hands.csv` |
| **File Model Hasil** | `model_abjad.pkl` | `model_kata_level2.pkl` | — |

---

## Penjelasan Detail Setiap File

### 1. Pipeline Utama Level 2 — Kata Dinamis (Gunakan Ini untuk Kata)
* **`app_level2.py`** — Aplikasi utama kamera untuk **Level 2**. Memiliki dua mode:
  * **Mode Inference (`N`):** Mendeteksi gerakan kata secara *real-time* dari buffer 30 frame, lengkap dengan pelacakan ujung jari telunjuk (*Motion Filter*) agar tangan yang diam terdeteksi sebagai `Diam / Bersiap...`.
  * **Mode Rekam (`K`):** Merekam 30 frame berurutan ke `dataset_kata_level2.csv` dengan menekan tombol angka `0`–`8`.
* **`train_level2.py`** — Melatih algoritma `MLPClassifier` dari `dataset_kata_level2.csv` dan menyimpan hasilnya ke `model_kata_level2.pkl`.
* **`kata_label.csv`** — Daftar 9 kelas target Level 2 sesuai urutan index (`0` = *Halo / Hai* hingga `8` = *Diam / Bersiap*).
* **`dataset_kata_level2.csv`** — Dataset mentah berisi rangkaian 30 frame per sampel gerakan kata.
* **`model_kata_level2.pkl`** — Model terlatih siap pakai untuk inferensi Level 2.

### 2. Pipeline Utama Level 1 — Abjad Statis (Gunakan Ini untuk Huruf A–Z)
* **`collect_alphabet.py`** — Script perekaman koordinat tangan untuk huruf abjad BISINDO (1 dan 2 tangan).
* **`clean_dataset.py`** — Menghapus semua dataset di abjad tertentu pada csv
* **`train_model.py`** — Melatih model klasifikasi abjad dari `dataset_abjad_optimal.csv` dan menghasilkan `model_abjad.pkl`.
* **`test_realtime.py`** — Aplikasi kamera untuk menguji deteksi huruf A–Z secara *real-time* menggunakan `model_abjad.pkl`.
* **`dataset_abjad_optimal.csv`** — Dataset bersih yang digunakan untuk training model abjad.
* **`model_abjad.pkl`** — Model terlatih siap pakai untuk inferensi Level 1.

### 3. Utilitas Manajemen Dataset Level 2
* **`hapus_label.py`** — Menghapus **seluruh baris sampel** milik satu ID kata tertentu dari `dataset_kata_level2.csv` (berguna saat ingin merekam ulang satu kata dari nol).
* **`undo_dataset.py`** — Menghapus **X baris terakhir** yang baru saja direkam ke `dataset_kata_level2.csv` (berguna jika salah gerakan saat sesi perekaman).
* **`reset_kata.py`** — Mengatur ulang / mengosongkan data kata saat ingin memulai perekaman baru.

### 4. File Arsip / Versi Awal (Legacy)
* **`collect_words.py`, `train_words.py`, `test_words.py`** — Purwarupa awal untuk deteksi kata sebelum digabung dan ditingkatkan menjadi arsitektur 30-frame di `app_level2.py` dan `train_level2.py`. **Tidak perlu dijalankan lagi.**
* **`dataset_abjad_2hands.csv`** — Dataset abjad mentah sebelum diproses oleh `clean_dataset.py`.

---

## Prasyarat & Setup Environment

* **Versi Python Direkomendasikan:** **Python 3.10.11** (kompatibel pada rentang Python `3.9` – `3.11`; hindari Python `3.12+` karena ketidakcocokan *wheel* MediaPipe dan Protobuf).

```powershell
# 1. Buat virtual environment menggunakan Python 3.10.11
py -3.10 -m venv venv

# 2. Aktifkan virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 3. Install dependensi dengan versi terkunci agar tidak bentrok
pip install numpy==1.26.4 protobuf==3.20.3 opencv-python==4.10.0.84 mediapipe==0.10.9 scikit-learn pandas
