import pandas as pd

DATASET_CSV = 'dataset_kata_level2.csv'
# Ubah angka ini sesuai dengan urutan kata di kata_label.csv yang ingin dihapus
# (Ingat: baris pertama = 0, baris kedua = 1, dst.)
TARGET_ID = 5  

print(f"Membaca {DATASET_CSV}...")
df = pd.read_csv(DATASET_CSV, header=None)

# Hitung jumlah awal
jumlah_awal = len(df)
jumlah_target = len(df[df[0] == TARGET_ID])

if jumlah_target == 0:
    print(f"Label {TARGET_ID} tidak ditemukan di dataset.")
else:
    # Buang semua baris yang kolom pertamanya sama dengan TARGET_ID
    df_bersih = df[df[0] != TARGET_ID]
    
    # Timpa file CSV lama dengan data yang sudah bersih
    df_bersih.to_csv(DATASET_CSV, index=False, header=False)
    
    print(f"BERHASIL! {jumlah_target} baris untuk label {TARGET_ID} telah dihapus.")
    print(f"Total data sekarang: {len(df_bersih)} baris (Awalnya {jumlah_awal}).")