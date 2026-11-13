import pandas as pd

DATASET_CSV = 'dataset_kata_level2.csv'
TARGET_ID = 7  # ID untuk data "Apa" yang lama

print(f"Membaca {DATASET_CSV}...")
try:
    df = pd.read_csv(DATASET_CSV, header=None)
    jumlah_awal = len(df)
    
    # Saring data, buang yang berlabel 7
    df_bersih = df[df[0] != TARGET_ID]
    jumlah_dihapus = jumlah_awal - len(df_bersih)
    
    if jumlah_dihapus == 0:
        print(f"Tidak ada data berlabel {TARGET_ID} yang ditemukan.")
    else:
        # Simpan kembali tanpa index 7
        df_bersih.to_csv(DATASET_CSV, index=False, header=False)
        print(f"BERHASIL! {jumlah_dihapus} baris rekaman 'Apa' (Label 7) telah dihapus.")
        print(f"Sisa data (termasuk Halo): {len(df_bersih)} baris.")
except FileNotFoundError:
    print("Dataset belum dibuat atau tidak ditemukan.")